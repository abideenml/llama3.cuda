/*
Kernels for swiglu forward pass.
NOTE: The results shown are performed on L4-GPU

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt swiglu_forward.cu -o swiglu_forward

- version 1 is naive port from CPU code to kernel: parallelizes over B,T,C
./swiglu_forward 1

RESULTS:
block_size   32 | time 0.2018 ms | bandwidth 187.05 GB/s
block_size   64 | time 0.1769 ms | bandwidth 213.37 GB/s
block_size  128 | time 0.1762 ms | bandwidth 214.22 GB/s
block_size  256 | time 0.1762 ms | bandwidth 214.21 GB/s
block_size  512 | time 0.1750 ms | bandwidth 215.73 GB/s
block_size 1024 | time 0.1939 ms | bandwidth 194.65 GB/s

- version 2 - uses 2 bfloat16 with the Packed128 data structure which helps in faster load/store operations, parallelizes over B,T,C
./swiglu_forward 2

RESULTS:
block_size   32 | time 0.1711 ms | bandwidth 220.62 GB/s
block_size   64 | time 0.1694 ms | bandwidth 222.87 GB/s
block_size  128 | time 0.1686 ms | bandwidth 223.90 GB/s
block_size  256 | time 0.1697 ms | bandwidth 222.44 GB/s
block_size  512 | time 0.1689 ms | bandwidth 223.46 GB/s
block_size 1024 | time 0.1701 ms | bandwidth 221.87 GB/s

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void swiglu_forward_cpu(float *out, const float *inp, const float *gate, int N)
{
    /**
     *
     * SwiGLU(x) = Swish(x) * Gate(x)
     * SwiGLU(x) = SiLU(x*W) * (x*V)
     * SiLU is the Swish activation function.
     * inp = x*W
     * gate = x*V
     *
     */

    for (int i = 0; i < N; i++)
    {
        float xW = inp[i];
        float xV = gate[i];
        out[i] = (floatX)((xW / (1.0f + expf(-xW))) * xV);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void swiglu_forward_kernel1(floatX *out, const floatX *inp, const floatX *gate, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float xiW = inp[i];
        float xiV = gate[i];
        out[i] = (floatX)((xiW / (1.0f + expf(-xiW))) * xiV);
    }
}

/**
 *  `x128` is a custom-datatype that represents a vector of data, typically used to pack multiple scalar values (like floats) into a single register for more efficient parallel processing.
 *  It holds 128 bits of data, which is commonly 4 float values (1 float -> 32 bits => 4*32 = 128)
 *
 *  By loading, processing, and storing 4 floats at a time (or however many the data type holds), you reduce the number of memory accesses and arithmetic operations, and thus resulting in better throughput and efficiency
 */
__global__ void swiglu_forward_kernel2(floatX *out, const floatX *inp, const floatX *gate, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N)
    {
        x128 packed_out;
        x128 packed_inp = load128cs(inp + i);   // load input (W) without cache streaming
        x128 packed_gate = load128cs(gate + i); // load gate (V) without cache streaming

        for (int k = 0; k < packed_inp.size; ++k)
        {
            float xiW = (float)packed_inp[k];                            // Extract element from packed input
            float xiV = (float)packed_gate[k];                           // Extract element from packed gate
            packed_out[k] = (floatX)((xiW / (1.0f + expf(-xiW))) * xiV); // SwiGLU operation
        }

        // Store the result back in memory (cached)
        store128(out + i, packed_out);
    }
}
// ----------------------------------------------------------------------------
// kernel launcher

void swiglu_forward1(floatX *out, const floatX *inp, const floatX *gate, int N, const int block_size)
{
    const int grid_size = ceil_div(N, block_size);
    swiglu_forward_kernel1<<<grid_size, block_size>>>(out, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

void swiglu_forward2(floatX *out, const floatX *inp, const floatX *gate, int N, const int block_size)
{
    const int grid_size = ceil_div(N, block_size * x128::size);
    swiglu_forward_kernel2<<<grid_size, block_size>>>(out, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void swiglu_forward(int kernel_num,
                    floatX *out, const floatX *inp, const floatX *gate,
                    int B, int T, int C, int block_size)
{
    switch (kernel_num)
    {
    case 1:
        swiglu_forward1(out, inp, gate, B * T * C, block_size);
        break;
    case 2:
        swiglu_forward2(out, inp, gate, B * T * C, block_size);
        break;

    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;

    // create host memory of random numbers
    float *out = (float *)malloc(B * T * C * sizeof(float));
    float *inp = make_random_float(B * T * C);  // precomputed x*W
    float *gate = make_random_float(B * T * C); // precomputed x*V

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    swiglu_forward_cpu(out, inp, gate, B * T * C);

    // move to GPU
    floatX *d_out;
    floatX *d_inp;
    floatX *d_gate; // Allocate device memory for gate
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_gate, B * T * C * sizeof(floatX))); // Allocate device memory for gate
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_gate, gate, B * T * C)); // Copy gate data to device

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        swiglu_forward(kernel_num, d_out, d_inp, d_gate, B, T, C, block_size); // Pass gate to the kernel
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out, out, "out", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, swiglu_forward,
                                              kernel_num, d_out, d_inp, d_gate,
                                              B, T, C, block_size); // Pass gate to the benchmark

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 3 * (int)sizeof(floatX); // Include gate in memory operations
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    free(inp);
    free(gate); // Free host memory for gate

    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_gate)); // Free device memory for gate
    return 0;
}
