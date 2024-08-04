from functools import partial
import argparse
import time

import sys
import numpy as np
import jax
import jax.numpy as jnp
import flax
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh
from jax.experimental.shard_map import shard_map

parser = argparse.ArgumentParser()
parser.add_argument("--bindings", type=str, default="cpp")
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()

if args.bindings == "cpp":
    import jax_flash_attn

    print(jax_flash_attn.__file__)
    from jax_flash_attn import run_mha
elif args.bindings == "rust":
    import jflash_attn

    print(jflash_attn.__file__)
    from jflash_attn import run_mha
else:
    raise ValueError('unsupported bindings "{args.bindings}", use "cpp" or "rust"')


def attn_einsum(q, k, v, mask=None):
    softmax_scale = q.shape[-1] ** -0.5
    qk = jnp.einsum("bqhd,bkhd->bhqk", q, k)

    if mask is not None:
        qk = qk + jnp.log(mask)
    attn_weights = jax.nn.softmax(qk * softmax_scale, axis=-1)
    attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    return attn


def bench(label, fwd, b_sz, seq_len, n_heads, dim, n_run=20, n_warmup=4, bwd=False):
    # the flops below only include the matmul of the forward pass
    flops = 4 * b_sz * seq_len * seq_len * n_heads * dim  # b.q.k.h.d
    if bwd:
        flops *= 3.5
    if bwd:

        def loss(q, k, v):
            return jnp.sum(fwd(q, k, v))

        f = jax.grad(loss, (0, 1, 2))
    else:
        f = fwd
    qkv_shape = b_sz, seq_len, n_heads, dim

    def normal(seed):
        rng = jax.random.PRNGKey(seed)
        return jax.random.normal(rng, qkv_shape, dtype=jnp.bfloat16)

    dts = []
    for i in range(n_warmup + n_run):
        q = normal(3 * i)
        k = normal(3 * i + 1)
        v = normal(3 * i + 2) / seq_len
        start_time = time.perf_counter()
        res = f(q, k, v)
        if bwd:
            res = res[0]
        res = res.block_until_ready()
        res = float(res.sum())
        dt = time.perf_counter() - start_time
        dts.append(dt)
    # print(dts)
    dts = dts[n_warmup:]
    dts = np.array(dts)
    min_ms = np.min(dts) * 1000
    max_ms = np.max(dts) * 1000
    mean_ms = np.mean(dts) * 1000
    std_ms = np.std(dts) * 1000
    gflops = flops / np.mean(dts) / 1e12
    print(
        f"{label:16} {seq_len:7} {mean_ms:5.2f}ms {gflops:8.1f} TFLOPS (std {std_ms:.2f}ms, min {min_ms:.2f}ms, max {max_ms:.2f}ms)"
    )


mesh = Mesh(jax.local_devices(), ("q",))
sh_q = P("q")

@partial(shard_map, mesh=mesh, in_specs=(sh_q, sh_q, sh_q), out_specs=sh_q, check_rep=False)
def run_mha_shmap(q, k, v, is_causal=False, softmax_scale=1.):
    return run_mha(q, k, v, is_causal=is_causal, softmax_scale=softmax_scale)

if True:
    run_mha_jit = jax.jit(run_mha_shmap)
    attn_einsum_jit = jax.jit(attn_einsum)
    attn_flax_jit = jax.jit(flax.linen.dot_product_attention)

    # Values taken from:
    # https://github.com/Dao-AILab/flash-attention/blob/2c3baba4a63c4007c8a132c5380edc9430f88a22/benchmarks/benchmark_flash_attention.py#L74C1-L77C11
    BSIZE_SEQLEN_VALS = [
        (32, 512),
        (16, 1024),
        (8, 2048),
        (4, 4096),
        (2, 8192),
        (2, 16384),
    ]
    HEADDIM = 128
    DIM = 2048
    n_heads = DIM // HEADDIM

    for b_sz, seqlen in BSIZE_SEQLEN_VALS:
        bench("flash-attn ", run_mha_jit, b_sz, seqlen, n_heads, HEADDIM)
        # bench("attn-einsum ", attn_einsum_jit, b_sz, seqlen, n_heads, HEADDIM)
        bench("attn-flax ", attn_flax_jit, b_sz, seqlen, n_heads, HEADDIM)

    for b_sz, seqlen in BSIZE_SEQLEN_VALS:
        bench("bwd flash-attn ", run_mha_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True)
        # bench("bwd attn-einsum", attn_einsum_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True)
        bench("bwd attn-flax", attn_flax_jit, b_sz, seqlen, n_heads, HEADDIM, bwd=True)

