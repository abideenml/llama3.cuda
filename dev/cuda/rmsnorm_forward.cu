/*
Kernels for layernorm forward pass.
The results shown are performed on L4-GPU

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt rmsnorm_forward.cu -o rmsnorm_forward

- version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./rmsnorm_forward 1

RESULTS:
block size   32 | time 0.5607 ms | bandwidth 89.76 GB/s
block_size   64 | time 0.6078 ms | bandwidth 82.81 GB/s
block_size  128 | time 0.6814 ms | bandwidth 73.86 GB/s
block_size  256 | time 0.6786 ms | bandwidth 74.17 GB/s
block_size  512 | time 0.9401 ms | bandwidth 53.54 GB/s
block_size 1024 | time 1.8104 ms | bandwidth 27.80 GB/s


- version 2 uses co-operative groups to work with warp-level reductions with a warp_size of 32 threads, parallelizes over B,T,C
./rmsnorm_forward 2

RESULTS:
All results match. Starting benchmarks.
block_size   32 | time 0.2380 ms | bandwidth 211.52 GB/s
block_size   64 | time 0.2438 ms | bandwidth 206.44 GB/s
block_size  128 | time 0.2445 ms | bandwidth 205.83 GB/s
block_size  256 | time 0.2434 ms | bandwidth 206.79 GB/s
block_size  512 | time 0.2421 ms | bandwidth 207.89 GB/s
block_size 1024 | time 0.2405 ms | bandwidth 209.31 GB/s

*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

// LLama RMSNorm forward pass
void rmsnorm_forward_cpu(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C)
{
    float eps = 1e-5f;
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // seek to the input position inp[b,t,:]
            const float *x = inp + b * T * C + t * C;
            // calculate the rms (root mean square)
            float rms = 0.0f;
            for (int i = 0; i < C; i++)
            {
                rms += x[i] * x[i];
            }
            rms = sqrtf(rms / C + eps);
            // seek to the output position in out[b,t,:]
            float *out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++)
            {
                float n = x[i] / rms;              // normalized output
                float o = n * weight[i] + bias[i]; // scale and shift it
                out_bt[i] = o;                     // write
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

/**
 * RMSNorm Kernel:
 *
 *   // https://pytorch.org/torchtune/stable/generated/torchtune.modules.RMSNorm.html
 *   // Source Code: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/rms_norm.html#RMSNorm.forward
 */
__global__ void rmsnorm_forward_kernel1(float *out, const float *inp, const float *weight, const float *bias, int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float eps = 1e-5f;

    if (idx < N)
    {
        // seek to the input position inp[idx,:]
        const float *x = inp + idx * C;
        // calculate the rms (root mean square)
        float rms = 0.0f;
        for (int i = 0; i < C; i++)
        {
            rms += x[i] * x[i];
        }
        rms = sqrtf(rms / C + eps);
        // seek to the output position in out[idx,:]
        float *out_idx = out + idx * C;
        for (int i = 0; i < C; i++)
        {
            float n = x[i] / rms;              // normalized output
            float o = n * weight[i] + bias[i]; // scale and shift it
            out_idx[i] = o;                    // write
        }
    }
}

// __restrict__ will hint the compiler to optimize the code better.
// Using __restrict__ will ensure no aliasing, and compiler can (freedom to) utilize maximum registers and stuff for optimization
__global__ void rmsnorm_forward_kernel2(float *__restrict__ out, const float *__restrict__ inp,
                                        const float *__restrict__ weight, const float *__restrict__ bias,
                                        int N, int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Calculate thread index within grid (each warp handles one row)
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
    {
        return;
    }

    // Pointer to input row
    const float *x = inp + idx * C;

    // RMS Calculation: First calculate sum of squares
    float sum_squares = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum_squares += x[i] * x[i];
    }

    // Reduce sum across threads within the warp
    sum_squares = cg::reduce(warp, sum_squares, cg::plus<float>{});

    // Calculate RMS
    float rms = sqrtf(sum_squares / C + 1e-5f);

    // Final normalization and scaling by weight/bias
    float *o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size())
    {
        float n = __ldcs(x + c) / rms;          // Normalized output
        __stcs(o + c, n * weight[c] + bias[c]); // Scale, shift and write output
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void rmsnorm_forward1(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, const int block_size)
{
    const int N = B * T;
    const int grid_size = (N + block_size - 1) / block_size; // equivalent to ceil(N / block_size)
    rmsnorm_forward_kernel1<<<grid_size, block_size>>>(out, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_forward2(float *out, const float *inp, const float *weight, const float *bias,
                      int B, int T, int C, const int block_size)
{
    assert(block_size % 32 == 0);
    const int N = B * T;
    const int grid_size = (N * 32 + block_size - 1) / block_size;
    rmsnorm_forward_kernel2<<<grid_size, block_size>>>(out, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void rmsnorm_forward(int kernel_num,
                     float *out,
                     const float *inp, const float *weight, const float *bias,
                     int B, int T, int C,
                     const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        rmsnorm_forward1(out, inp, weight, bias, B, T, C, block_size);
        break;
    case 2:
        rmsnorm_forward2(out, inp, weight, bias, B, T, C, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    srand(0);

    int B = 8;
    int T = 1024;
    int C = 768;

    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));

    // create host memory of random numbers
    float *out = (float *)malloc(B * T * C * sizeof(float));
    // float *mean = (float *)malloc(B * T * sizeof(float));
    // float *rstd = (float *)malloc(B * T * sizeof(float));
    float *inp = make_random_float(B * T * C);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);

    // move to GPU
    float *d_out;
    // float *d_mean;
    // float *d_rstd;
    float *d_inp;
    float *d_weight;
    float *d_bias;
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    // cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
    // cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};

    rmsnorm_forward_cpu(out, inp, weight, bias, B, T, C);

    // check the correctness of the kernel at all block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);

        rmsnorm_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, block_size);

        validate_result(d_out, out, "out", B * T * C, 1e-5f);
    }

    printf("All results match. Starting benchmarks.\n\n");

    // time the kernel at different block sizes
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 2000;
        float elapsed_time = benchmark_kernel(repeat_times, rmsnorm_forward,
                                              kernel_num, d_out, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(out);
    // free(mean);
    // free(rstd);
    free(inp);
    free(weight);
    free(bias);
    cudaCheck(cudaFree(d_out));
    // cudaCheck(cudaFree(d_mean));
    // cudaCheck(cudaFree(d_rstd));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}