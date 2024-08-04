# jax-flash-attn

This repo contains bindings for [FlashAttention2](https://github.com/Dao-AILab/flash-attention)
in JAX. There are two versions for these bindings, a C++ version
`jax_flash_attn` and a Rust version `jflash_attn`.

The BSD-3 license that holds for the flash-attention repo also applies here.

## Building the C++ Version

Build a wheel file. `-j32` will compile 32 cuda kernels in parallel which could exhaust memory on boxes with
less than 100GB.
```bash
python setup.py bdist_wheel -- -- -j32
```

Build locally for development.
```bash
python setup.py build_ext -i -- -- -j32
python test.py # run some tests and benchmarks
```

This may require you to install the two following pip packages:
```bash
pip install scikit_build
pip install "pybind11[global]"
```

## Building the Rust Version

In order to build a python package as a wheel, run `maturin build --release`.
In order to build a python package and install it in the current virtual
enviroment, run `maturin develop`.

## Running the Tests and Benchmarks

First compile the C++ and/or Rust package and install them locally. Use the
following to run the tests.
```bash
python test.py --bindings cpp
python test.py --bindings rust
```

And use the `--bench` flag to run the benchmarks instead of the tests.

```bash
python test.py --bindings cpp --bench True
python test.py --bindings rust --bench True
```

## Benchmarks (H100 80G HBM3)

This measures the time spent in the attention layer for three different implementations.
- `flash-attn`: uses the optimized flash-attention kernel. 
- `attn-einsum`: uses a simple attention implementation based on einsum.
- `attn-flax`: uses `flax.linen.dot_product_attention`.
Timings include the forward pass only for the first lines and both the forward
and backward passes for the lines that start with `bwd`. The second column is the
sequence length (the batch size is adapted so as to have a reasonable amount of
computation).

```
flash-attn           512  1.23ms     55.8 TFLOPS (std 0.54ms, min 0.79ms, max 2.38ms)
attn-flax            512  1.83ms     37.6 TFLOPS (std 0.58ms, min 1.54ms, max 3.88ms)
flash-attn          1024  1.24ms    110.7 TFLOPS (std 0.38ms, min 0.89ms, max 2.14ms)
attn-flax           1024  2.40ms     57.2 TFLOPS (std 0.49ms, min 1.81ms, max 3.58ms)
flash-attn          2048  1.59ms    173.2 TFLOPS (std 0.34ms, min 1.37ms, max 2.44ms)
attn-flax           2048  3.46ms     79.4 TFLOPS (std 0.30ms, min 3.04ms, max 4.42ms)
flash-attn          4096  2.40ms    229.2 TFLOPS (std 0.22ms, min 2.23ms, max 3.24ms)
attn-flax           4096  6.08ms     90.4 TFLOPS (std 0.45ms, min 5.76ms, max 7.32ms)
flash-attn          8192  4.26ms    258.3 TFLOPS (std 0.25ms, min 4.08ms, max 4.96ms)
attn-flax           8192 11.19ms     98.3 TFLOPS (std 0.31ms, min 10.85ms, max 12.08ms)
flash-attn         16384  7.86ms    279.8 TFLOPS (std 0.35ms, min 7.63ms, max 8.81ms)
attn-flax          16384 26.56ms     82.8 TFLOPS (std 0.48ms, min 25.96ms, max 27.62ms)
bwd flash-attn       512  3.01ms     79.9 TFLOPS (std 0.44ms, min 2.74ms, max 4.42ms)
bwd attn-flax        512  4.26ms     56.4 TFLOPS (std 0.43ms, min 3.88ms, max 5.50ms)
bwd flash-attn      1024  3.90ms    123.3 TFLOPS (std 0.53ms, min 3.30ms, max 4.92ms)
bwd attn-flax       1024  5.43ms     88.6 TFLOPS (std 0.53ms, min 5.05ms, max 6.70ms)
bwd flash-attn      2048  5.22ms    184.4 TFLOPS (std 0.61ms, min 4.52ms, max 6.51ms)
bwd attn-flax       2048  8.69ms    110.6 TFLOPS (std 0.62ms, min 8.22ms, max 10.66ms)
bwd flash-attn      4096  7.58ms    253.9 TFLOPS (std 0.30ms, min 7.35ms, max 8.47ms)
bwd attn-flax       4096 15.08ms    127.6 TFLOPS (std 0.55ms, min 14.55ms, max 16.43ms)
bwd flash-attn      8192 14.22ms    270.7 TFLOPS (std 0.76ms, min 13.56ms, max 16.65ms)
bwd attn-flax       8192 28.03ms    137.3 TFLOPS (std 0.58ms, min 27.51ms, max 29.30ms)
bwd flash-attn     16384 26.42ms    291.4 TFLOPS (std 0.45ms, min 26.03ms, max 27.50ms)
bwd attn-flax      16384 57.84ms    133.1 TFLOPS (std 0.61ms, min 57.28ms, max 59.24ms)
```
