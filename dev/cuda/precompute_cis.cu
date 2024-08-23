/*
Precompute-cis, A helper kernel, used to calculate the Polar vector for the Rotational Positional Embedding (RoPE).
It is a mathematical computatinal kernel, which is used just to help in advance calculations in order to add psitional information in Query(Q) and Key(K) Matrices in Attention.

Kernels for precompute_cis.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt precompute_cis.cu -o precompute_cis

version 1 is naive kernel from CPU Port, Parellelizes over `dim` only
./precompute_cis 1

RESULTS:
block_size   32 | time 0.0410 ms | bandwidth 12.80 GB/s
block_size   64 | time 0.0408 ms | bandwidth 12.84 GB/s
block_size  128 | time 0.0409 ms | bandwidth 12.81 GB/s
block_size  256 | time 0.0408 ms | bandwidth 12.84 GB/s
block_size  512 | time 0.0414 ms | bandwidth 12.67 GB/s
block_size 1024 | time 0.0415 ms | bandwidth 12.64 GB/s

version 2 utilizes coalesced Global Memory access, paralleizes over `dim`,`t` both
~ 4x faster than naive CPU implementation
./precompute_cis 2

RESULTS:
block_size   32 | time 0.0098 ms | bandwidth 53.62 GB/s
block_size   64 | time 0.0095 ms | bandwidth 55.32 GB/s
block_size  128 | time 0.0098 ms | bandwidth 53.50 GB/s
block_size  256 | time 0.0098 ms | bandwidth 53.54 GB/s
block_size  512 | time 0.0112 ms | bandwidth 46.71 GB/s
block_size 1024 | time 0.0167 ms | bandwidth 31.37 GB/s
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void precompute_freqs_cis_cpu(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    for (int tid = 0; tid < dim / 2; tid++)
    {
        // Compute the frequency for each tid
        float freq = 1.0f / powf(theta, (float)(tid * 2.0f) / dim); // float powf(float base, float exponent);

        // Compute the cosine and sine for all values of 't'
        for (int t = 0; t < end; t++)
        {
            freqs_cos[t * (dim / 2) + tid] = cosf(t * freq);
            freqs_sin[t * (dim / 2) + tid] = sinf(t * freq);
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

/**
 * A naive implementation from CPU Code.
 */
__global__ void precompute_freqs_cis_kernel1(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dim / 2)
    {
        float freq = 1.0f / powf(theta, (float)tid * 2.0f / dim);

        for (int t = 0; t < end; t++)
        {
            freqs_cos[t * (dim / 2) + tid] = cosf(t * freq);
            freqs_sin[t * (dim / 2) + tid] = sinf(t * freq);
        }
    }
}

/**
 * Coalesced Gloabal Memory Access, because threads within a warp write consecutive values to global memory
 * - Launches one block per t value - a nice startegy
 */
__global__ void precompute_freqs_cis_kernel2(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load global index for the t-loop in parallel to maximize throughput
    int tid_t = blockIdx.y;

    if (tid < dim / 2 && tid_t < end)
    {
        float freq = 1.0f / powf(theta, (float)(tid * 2.0f) / dim);

        // Global memory coalesced access: every warp accesses a consecutive memory address
        freqs_cos[tid_t * (dim / 2) + tid] = cosf(tid_t * freq);
        freqs_sin[tid_t * (dim / 2) + tid] = sinf(tid_t * freq);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void precompute_freqs_cis1(float *freqs_cos, float *freqs_sin, int dim, int end, float theta, int block_size)
{
    int grid_size = (dim / 2 + block_size - 1) / block_size;
    precompute_freqs_cis_kernel1<<<grid_size, block_size>>>(freqs_cos, freqs_sin, dim, end, theta);
    cudaDeviceSynchronize();
}

void precompute_freqs_cis2(float *freqs_cos, float *freqs_sin, int dim, int end, float theta, int block_size)
{
    int blocks_x = (dim / 2 + block_size - 1) / block_size;
    int blocks_y = end; // Launch one block per `t`

    dim3 blocks(blocks_x, blocks_y); // 2D grid for (dim, end)

    precompute_freqs_cis_kernel2<<<blocks, block_size>>>(freqs_cos, freqs_sin, dim, end, theta);
    cudaDeviceSynchronize();
}

// kernel version dispatch
void precompute_freqs_cis(int kernel_num,
                          float *freqs_cos, float *freq_sin,
                          int dim, int end, float theta,
                          int block_size)
{
    switch (kernel_num)
    {
    case 1:
        precompute_freqs_cis1(freqs_cos, freq_sin, dim, end, theta, block_size);
        break;
    case 2:
        precompute_freqs_cis2(freqs_cos, freq_sin, dim, end, theta, block_size);
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
    int NH = 12;
    float theta = 10000.0f;

    // create host memory of random numbers
    float *freq_cos = (float *)malloc(T * (C / (NH * 2)) * sizeof(float));
    float *freq_sin = (float *)malloc(T * (C / (NH * 2)) * sizeof(float));

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    precompute_freqs_cis_cpu(freq_cos, freq_sin, (C / NH), T, theta);

    // move to GPU
    float *d_freq_cos;
    float *d_freq_sin;
    cudaCheck(cudaMalloc(&d_freq_cos, T * (C / (NH * 2)) * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freq_sin, T * (C / (NH * 2)) * sizeof(float)));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        precompute_freqs_cis(kernel_num, d_freq_cos, d_freq_sin, (C / NH), T, theta, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_freq_cos, freq_cos, "freq_cos", T * (C / (NH * 2)), tol);
        validate_result(d_freq_sin, freq_sin, "freq_sin", T * (C / (NH * 2)), tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, precompute_freqs_cis,
                                              kernel_num,
                                              d_freq_cos, d_freq_sin, (C / NH), 2 * T, theta,
                                              block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (T, C/(NH*2)) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = 2 * T * (C / (NH * 2)) * 2 * (int)sizeof(float); // Since I have two putp (freqs_cos and freqs_sin),
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(freq_cos);
    free(freq_sin);

    cudaCheck(cudaFree(d_freq_cos));
    cudaCheck(cudaFree(d_freq_sin));
    return 0;
}