from functools import partial
from jax.sharding import Mesh, PartitionSpec

import argparse

import jax
import jax.numpy as jnp
import flax

from jax.experimental.pjit import pjit

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--bindings", type=str, default="cpp")
args = parser.parse_args()

if args.bindings == "cpp":
    import jax_flash_attn

    print(jax_flash_attn.__file__)
    from jax_flash_attn import xmap_run_mha
elif args.bindings == "rust":
    import jflash_attn

    print(jflash_attn.__file__)
    from jflash_attn import xmap_run_mha
else:
    raise ValueError('unsupported bindings "{args.bindings}", use "cpp" or "rust"')


mesh = Mesh(jax.local_devices(), ("q",))


def attn_einsum(q, k, v, mask=None):
    softmax_scale = q.shape[-1] ** -0.5
    qk = jnp.einsum("bqhd,bkhd->bhqk", q, k)

    if mask is not None:
        qk = qk + jnp.log(mask)
    attn_weights = jax.nn.softmax(qk * softmax_scale, axis=-1)
    attn = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    return attn


def test_fwd(qkv_shape, max_err, is_causal):
    _b_size, seqlen, _num_heads, head_dim = qkv_shape
    rng_q = jax.random.PRNGKey(0)
    q = jax.random.normal(rng_q, qkv_shape, dtype=jnp.bfloat16)
    rng_k = jax.random.PRNGKey(1)
    k = jax.random.normal(rng_k, qkv_shape, dtype=jnp.bfloat16)
    rng_v = jax.random.PRNGKey(2)
    v = jax.random.normal(rng_v, qkv_shape, dtype=jnp.bfloat16) / seqlen

    mask = None
    if is_causal:
        mask = jnp.tril(jnp.ones((seqlen, seqlen)))

    softmax_scale = head_dim**-0.5
    # attn_mha  = xmap_run_mha(q, k, v, is_causal=is_causal, softmax_scale=softmax_scale, device_count=jax.local_device_count())
    pjitted = pjit(
        partial(
            xmap_run_mha,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            device_count=jax.local_device_count(),
        ),
        # Shard x by batch dimension and replicate weight on all devices.
        in_shardings=(
            PartitionSpec("q", None, None, None),
            PartitionSpec("q", None, None, None),
            PartitionSpec("q", None, None, None),
        ),
        # Shard the output by batch dimension.
        out_shardings=PartitionSpec("q", None, None, None),
    )
    attn_mha = pjitted(q, k, v)
    attn_ein = attn_einsum(q, k, v, mask=mask)
    attn_flax = flax.linen.dot_product_attention(q, k, v, mask=mask)

    diff_mha_ein = (attn_mha - attn_ein).max()
    diff_mha_flax = (attn_mha - attn_flax).max()
    diff_ein_flax = (attn_ein - attn_flax).max()
    if args.verbose:
        print("fwd", diff_mha_ein, diff_mha_flax, diff_ein_flax)
    if not (diff_mha_ein <= max_err):  # be cautious about handling nans
        print(
            "FAIL    fwd",
            qkv_shape,
            diff_mha_ein,
            diff_mha_flax,
            diff_ein_flax,
            is_causal,
        )


def test_bwd(qkv_shape, max_err, is_causal):
    _b_size, seqlen, _num_heads, head_dim = qkv_shape
    rng_q = jax.random.PRNGKey(0)
    q = jax.random.normal(rng_q, qkv_shape, dtype=jnp.float16)
    rng_k = jax.random.PRNGKey(1)
    k = jax.random.normal(rng_k, qkv_shape, dtype=jnp.float16)
    rng_v = jax.random.PRNGKey(2)
    v = jax.random.normal(rng_v, qkv_shape, dtype=jnp.float16)

    mask = None
    if is_causal:
        mask = jnp.tril(jnp.ones((seqlen, seqlen)))

    def loss_mha(q, k, v):
        softmax_scale = head_dim**-0.5
        predictions = xmap_run_mha(
            q,
            k,
            v,
            is_causal=is_causal,
            softmax_scale=softmax_scale,
            device_count=jax.local_device_count(),
        )
        return jnp.sum(predictions)

    loss_mha_grad = jax.grad(loss_mha, (0, 1, 2))

    def loss_flax(q, k, v):
        predictions = flax.linen.dot_product_attention(q, k, v, mask=mask)
        return jnp.sum(predictions)

    loss_flax_grad = jax.grad(loss_flax, (0, 1, 2))

    dq_mha, dk_mha, dv_mha = loss_mha_grad(q, k, v)
    dq_flax, dk_flax, dv_flax = loss_flax_grad(q, k, v)

    dq_diff = ((dq_mha - dq_flax) ** 2).mean()
    dk_diff = ((dk_mha - dk_flax) ** 2).mean()
    dv_diff = ((dv_mha - dv_flax) ** 2).mean()
    if args.verbose:
        print("bwd", dq_diff, dk_diff, dv_diff)
    if not (
        dq_diff <= max_err and dk_diff <= max_err and dv_diff <= max_err
    ):  # be cautious about nans.
        print("FAIL    bwd", qkv_shape, dq_diff, dk_diff, dv_diff, is_causal)


TEST_CASES = [
    ((1, 20, 16, 32), 1e-3),
    ((16, 100, 28, 64), 2e-4),
    ((16, 512, 32, 128), 1e-4),
    ((21, 50, 17, 160), 5e-4),
]

with mesh:
    for _qkv, _max_err in TEST_CASES:
        test_fwd(_qkv, _max_err, is_causal=False)
        test_fwd(_qkv, _max_err, is_causal=True)
        test_bwd(_qkv, _max_err, is_causal=False)
        test_bwd(_qkv, _max_err, is_causal=True)
