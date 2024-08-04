python_executable=$(shell python -c 'import sys; print(sys.executable)')
pybind_include_path=$(shell python -c "import pybind11; print(pybind11.get_include())")

FLASH_FWD_CU =  \
	build/flash_fwd_hdim32_fp16_sm80.cu.o \
	build/flash_fwd_hdim64_fp16_sm80.cu.o \
	build/flash_fwd_hdim96_fp16_sm80.cu.o \
	build/flash_fwd_hdim128_fp16_sm80.cu.o \
	build/flash_fwd_hdim160_fp16_sm80.cu.o \
	build/flash_fwd_hdim192_fp16_sm80.cu.o \
	build/flash_fwd_hdim224_fp16_sm80.cu.o \
	build/flash_fwd_hdim256_fp16_sm80.cu.o \
	build/flash_fwd_hdim32_bf16_sm80.cu.o \
	build/flash_fwd_hdim64_bf16_sm80.cu.o \
	build/flash_fwd_hdim96_bf16_sm80.cu.o \
	build/flash_fwd_hdim128_bf16_sm80.cu.o \
	build/flash_fwd_hdim160_bf16_sm80.cu.o \
	build/flash_fwd_hdim192_bf16_sm80.cu.o \
	build/flash_fwd_hdim224_bf16_sm80.cu.o \
	build/flash_fwd_hdim256_bf16_sm80.cu.o \
	build/flash_fwd_hdim32_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim64_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim96_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim128_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim160_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim192_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim224_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim256_fp16_causal_sm80.cu.o \
	build/flash_fwd_hdim32_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim64_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim96_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim128_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim160_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim192_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim224_bf16_causal_sm80.cu.o \
	build/flash_fwd_hdim256_bf16_causal_sm80.cu.o
FLASH_BWD_CU =  \
	build/flash_bwd_hdim32_fp16_sm80.cu.o \
	build/flash_bwd_hdim64_fp16_sm80.cu.o \
	build/flash_bwd_hdim96_fp16_sm80.cu.o \
	build/flash_bwd_hdim128_fp16_sm80.cu.o \
	build/flash_bwd_hdim160_fp16_sm80.cu.o \
	build/flash_bwd_hdim192_fp16_sm80.cu.o \
	build/flash_bwd_hdim224_fp16_sm80.cu.o \
	build/flash_bwd_hdim256_fp16_sm80.cu.o \
	build/flash_bwd_hdim32_bf16_sm80.cu.o \
	build/flash_bwd_hdim64_bf16_sm80.cu.o \
	build/flash_bwd_hdim96_bf16_sm80.cu.o \
	build/flash_bwd_hdim128_bf16_sm80.cu.o \
	build/flash_bwd_hdim160_bf16_sm80.cu.o \
	build/flash_bwd_hdim192_bf16_sm80.cu.o \
	build/flash_bwd_hdim224_bf16_sm80.cu.o \
	build/flash_bwd_hdim256_bf16_sm80.cu.o

all: build/flash_attn.so

clean:
	rm -Rf build/*
build/%.cu.o : src/%.cu
	nvcc --threads 4 -Xcompiler -Wall -ldl --expt-relaxed-constexpr -O3 -DNDEBUG -Xcompiler -O3 \
		-Icutlass/include -std=c++17 \
		--generate-code=arch=compute_90,code=[compute_90,sm_90] \
		-Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden -x cu -c $< -o $@

build/flash_attn_ops.cpp.o: src/flash_attn_ops.cpp src/pybind11_kernel_helpers.h src/kernels.h
	c++ -I/usr/local/cuda/include -I/usr/include/python3.10 -std=c++17 \
		-I$(pybind_include_path) $(${python_executable}-config --cflags) \
		-O3 -DNDEBUG -O3 -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects \
		-o build/flash_attn_ops.cpp.o -c src/flash_attn_ops.cpp

build/flash_attn.so: $(FLASH_FWD_CU) $(FLASH_BWD_CU) build/flash_attn_ops.cpp.o build/flash_api.cu.o
	c++ -fPIC -O3 -DNDEBUG -O3 -flto -shared -o $@ -std=c++17 \
		build/flash_attn_ops.cpp.o $(FLASH_FWD_CU) $(FLASH_BWD_CU) build/flash_api.cu.o -L/usr/local/cuda/lib64 \
		-lcudadevrt -lcudart_static -lrt -lpthread -ldl
