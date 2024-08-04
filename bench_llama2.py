from dataclasses import dataclass
from typing import Callable
import time
import numpy as np

import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
import optax
from safetensors import safe_open

from jax_flash_attn import run_mha

BSIZE = 1
USE_SAFETENSORS = True
SEQLEN = 4096

@dataclass
class Config:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    rope_theta: float
    use_flash_attn: bool

    def v2_7b(use_flash_attn: bool):
        return Config(
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
            num_hidden_layers=32,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            rope_theta=1e4,
            use_flash_attn=use_flash_attn,
        )

    def flops(self, bsize, seqlen):
        # Attention flops
        flops = 4 * bsize * seqlen ** 2 * self.hidden_size # b.q.k.h.d
        flops += 2 * bsize * seqlen * self.hidden_size ** 2
        # MLP flops
        flops += 2 * bsize * seqlen * self.hidden_size * self.intermediate_size * 3
        flops *= self.num_hidden_layers
        return flops

class RmsNorm(nn.Module):
    sz: int
    eps: float = 1e-5
    w_init: Callable = nn.initializers.ones_init()

    def setup(self):
        self.ws = self.param('weight', self.w_init, (self.sz,))

    def __call__(self, xs):
        return xs / jnp.sqrt((xs * xs).mean(-1, keepdims=True) + self.eps) * self.ws

class RotaryEmbeddings(nn.Module):
    cfg: Config

    def setup(self):
        head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
        theta = (1 / jnp.power(self.cfg.rope_theta, jnp.arange(0, head_dim, 2) / head_dim)).reshape((1, -1))
        idx = jnp.arange(SEQLEN).astype('float32').reshape((-1, 1))
        idx_theta = idx @ theta
        idx_theta = jnp.concatenate((idx_theta, idx_theta), -1)
        self.cos = jnp.cos(idx_theta).reshape((1, SEQLEN, 1, head_dim))
        self.sin = jnp.sin(idx_theta).reshape((1, SEQLEN, 1, head_dim))

    def __call__(self, xs):
        head_dim = self.cfg.hidden_size // self.cfg.num_attention_heads
        xs1 = xs[:, :, :, :head_dim // 2]
        xs2 = xs[:, :, :, head_dim // 2:]
        rotate_x = jnp.concatenate((-xs2, xs1), -1)
        rope = xs * self.cos.astype(xs.dtype) + rotate_x * self.sin.astype(xs.dtype)
        return rope


class CausalSelfAttention(nn.Module):
    hidden_size: int
    num_attention_heads: int
    use_flash_attn: bool
    rotary_embeddings: RotaryEmbeddings
    mask: jax.Array

    def setup(self):
        hidden_size = self.hidden_size
        self.q_proj = nn.Dense(hidden_size, use_bias=False)
        self.k_proj = nn.Dense(hidden_size, use_bias=False)
        self.v_proj = nn.Dense(hidden_size, use_bias=False)
        self.o_proj = nn.Dense(hidden_size, use_bias=False)

    def __call__(self, xs):
        b_sz, seq_len, hidden_size = xs.shape
        q = self.q_proj(xs)
        k = self.k_proj(xs)
        v = self.v_proj(xs)

        nha = self.num_attention_heads
        head_size = hidden_size // nha

        q = q.reshape((b_sz, seq_len, nha, head_size))
        k = k.reshape((b_sz, seq_len, nha, head_size))
        v = v.reshape((b_sz, seq_len, nha, head_size))

        q = self.rotary_embeddings(q)
        k = self.rotary_embeddings(k)

        softmax_scale = 1.0 / head_size ** 0.5

        if self.use_flash_attn:
            output = run_mha(q, k, v, is_causal=True, softmax_scale=softmax_scale)
        else:
            q = q.astype('float32')
            k = k.astype('float32')
            v = v.astype('float32')
            q = q.transpose((0, 2, 1, 3))
            attn = q @ k.transpose((0, 2, 3, 1)) + self.mask
            attn = jax.nn.softmax(attn * softmax_scale)
            output = attn @ v.transpose((0, 2, 1, 3))
            output = output.transpose((0, 2, 1, 3))
            output = output.astype(xs.dtype)
        return self.o_proj(output.reshape(xs.shape))

class MLP(nn.Module):
    hidden_size: int
    intermediate_size: int

    def setup(self):
        h_size = self.hidden_size
        i_size = self.intermediate_size
        self.fc1 = nn.Dense(i_size, use_bias=False)
        self.fc2 = nn.Dense(i_size, use_bias=False)
        self.c_proj = nn.Dense(h_size, use_bias=False)

    def __call__(self, xs):
        xs = jax.nn.silu(self.fc1(xs)) * self.fc2(xs)
        return self.c_proj(xs)

class Block(nn.Module):
    cfg: Config
    rotary_embeddings: RotaryEmbeddings
    mask: jax.Array

    def setup(self):
        c = self.cfg
        self.input_layernorm = RmsNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RmsNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = MLP(c.hidden_size, c.intermediate_size)
        self.self_attn = CausalSelfAttention(
            c.hidden_size,
            c.num_attention_heads,
            c.use_flash_attn,
            self.rotary_embeddings,
            self.mask,
        )

    def __call__(self, xs):
        xs = self.self_attn(self.input_layernorm(xs)) + xs
        xs = self.mlp(self.post_attention_layernorm(xs)) + xs
        return xs

class Llama(nn.Module):
    cfg: Config

    def setup(self):
        c = self.cfg
        self.wte = nn.Embed(
            num_embeddings=c.vocab_size,
            features=c.hidden_size,
        )
        rotary_embeddings = RotaryEmbeddings(c)
        mask = jnp.log(jnp.tril(jnp.ones((SEQLEN, SEQLEN))))
        layers = []
        for _layer_idx in range(c.num_hidden_layers):
            block = Block(c, rotary_embeddings, mask)
            layers.append(block)
        self.layers = layers
        self.ln_f = RmsNorm(c.hidden_size)
        self.lm_head = nn.Dense(c.vocab_size, use_bias=False)

    def __call__(self, xs):
        _b_sz, _seq_len = xs.shape
        xs = self.wte(xs)
        for layer in self.layers:
            xs = layer(xs)
        xs = self.ln_f(xs)
        return self.lm_head(xs)

tokens = jnp.array([[i % 1000 for i in range(SEQLEN)]] * BSIZE)

def to_bfloat16(params):
    def _to_bfloat16(param):
        if isinstance(param, jnp.ndarray):
            return param.astype(jnp.bfloat16)
        else:
            return param
    return jax.tree_util.tree_map(_to_bfloat16, params)

rng = jax.random.PRNGKey(0)
model_fa = Llama(Config.v2_7b(True))

model = Llama(Config.v2_7b(False))

if USE_SAFETENSORS:
    from huggingface_hub import hf_hub_download
    rename_paths = {
        "model.embed_tokens.weight": "params.wte.embedding",
        "lm_head.weight": "params.lm_head.kernel",
        "model.norm.weight": "params.ln_f.weight",
    }
    for layer_id in range(100):
        rename_paths[f"model.layers.{layer_id}.input_layernorm.weight"] = (
                f"params.layers_{layer_id}.input_layernorm.weight")
        rename_paths[f"model.layers.{layer_id}.post_attention_layernorm.weight"] = (
                f"params.layers_{layer_id}.post_attention_layernorm.weight")
        rename_paths[f"model.layers.{layer_id}.self_attn.q_proj.weight"] = (
                f"params.layers_{layer_id}.self_attn.q_proj.kernel")
        rename_paths[f"model.layers.{layer_id}.self_attn.k_proj.weight"] = (
                f"params.layers_{layer_id}.self_attn.k_proj.kernel")
        rename_paths[f"model.layers.{layer_id}.self_attn.v_proj.weight"] = (
                f"params.layers_{layer_id}.self_attn.v_proj.kernel")
        rename_paths[f"model.layers.{layer_id}.self_attn.o_proj.weight"] = (
                f"params.layers_{layer_id}.self_attn.o_proj.kernel")
        rename_paths[f"model.layers.{layer_id}.mlp.down_proj.weight"] = (
                f"params.layers_{layer_id}.mlp.c_proj.kernel")
        rename_paths[f"model.layers.{layer_id}.mlp.gate_proj.weight"] = (
                f"params.layers_{layer_id}.mlp.fc1.kernel")
        rename_paths[f"model.layers.{layer_id}.mlp.up_proj.weight"] = (
                f"params.layers_{layer_id}.mlp.fc2.kernel")
        rename_paths[f"model.layers.{layer_id}.self_attn.rotary_emb.inv_freq"] = "params.TODO"
    params = {}
    filenames = [
            hf_hub_download(
                repo_id="meta-llama/Llama-2-7b-hf",
                filename="model-00001-of-00002.safetensors",
            ),
            hf_hub_download(
                repo_id="meta-llama/Llama-2-7b-hf",
                filename="model-00002-of-00002.safetensors",
            ),
    ]
    for filename in filenames:
        with safe_open(filename, framework="flax") as f_obj:
            for orig_path in f_obj.keys():
                current = params
                path = rename_paths[orig_path]
                path_split = path.split('.')
                for i, k in enumerate(path_split):
                    if i == len(path_split) - 1:
                        tensor = f_obj.get_tensor(orig_path)
                        if k == "kernel": tensor = tensor.transpose((1, 0))
                        current[k] = tensor
                        break
                    if k not in current:
                        current[k] = {}
                    current = current[k]
else:
    params = model.init(rng, tokens)

params = to_bfloat16(params)

forward = jax.jit(model.apply)
forward_fa = jax.jit(model_fa.apply)

ca = forward.lower(params, tokens).compile().cost_analysis()
flops = ca[0]["flops"]
print(flops/1e12)

flops = Config.v2_7b(False).flops(BSIZE, SEQLEN)
print(flops / 1e12)

logits = forward(params, tokens)
logits_fa = forward_fa(params, tokens)

print("FA", logits_fa.shape, logits_fa.dtype)
print(logits_fa)
print("NO FA", logits.shape, logits.dtype)
print(logits)

def bench(label, fwd, n_run=8, n_warmup=2, bwd=False):
    if bwd:
        def loss(params, xs):
            return jnp.sum(fwd(params, xs))
        f = jax.grad(loss, 0)
        f = jax.jit(f)
    else:
        f = fwd
    dts = []
    for _ in range(n_warmup + n_run):
        start_time = time.perf_counter()
        res = f(params, tokens)
        if bwd:
          res = res[0]
        res = res.block_until_ready()
        res = float(res.sum())
        dt = time.perf_counter() - start_time
        dts.append(dt)
    #print(dts)
    dts = dts[n_warmup:]
    dts = np.array(dts)
    min_ms = np.min(dts) * 1000
    max_ms = np.max(dts) * 1000
    mean_ms = np.mean(dts) * 1000
    std_ms = np.std(dts) * 1000
    tflops = flops / np.mean(dts) / 1e12
    print(f"{label:16} {mean_ms:5.2f}ms {tflops:.2f} TFLOPS (std {std_ms:.2f}ms, min {min_ms:.2f}ms, max {max_ms:.2f}ms)")

bench("fwd no-flash-attn  ", forward)
bench("fwd with-flash-attn", forward_fa)

# Note that the backward step requires multiple H100 GPUs.
bench("bwd no-flash-attn  ", forward, bwd=True)
bench("bwd with-flash-attn", forward_fa, bwd=True)
