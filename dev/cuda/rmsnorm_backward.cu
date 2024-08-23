/*
Kernels for layernorm backward pass.

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt rmsnorm_backward.cu -o rmsnorm_backward

* version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
./rmsnorm_backward 1

RESULTS: (on L4-GPU)
block_size   32 | time 3.4939 ms | bandwidth 14.41 GB/s
block_size   64 | time 3.2542 ms | bandwidth 15.47 GB/s
block_size  128 | time 3.3129 ms | bandwidth 15.19 GB/s
block_size  256 | time 3.3933 ms | bandwidth 14.83 GB/s
block_size  512 | time 5.0531 ms | bandwidth 9.96 GB/s
block_size 1024 | time 6.9545 ms | bandwidth 7.24 GB/

* version 2 uses co-operative groups to work with warp-level reductions with a warp_size of 32 threads, parallelizes over B,T,C
~9x faster than kernel 1.
./rmsnorm_backward 2

RESULTS: (on L4-GPU)
block_size   32 | time 0.3957 ms | bandwidth 127.21 GB/s
block_size   64 | time 0.4064 ms | bandwidth 123.83 GB/s
block_size  128 | time 0.4066 ms | bandwidth 123.79 GB/s
block_size  256 | time 0.4075 ms | bandwidth 123.52 GB/s
block_size  512 | time 0.4061 ms | bandwidth 123.95 GB/s
block_size 1024 | time 0.4092 ms | bandwidth 122.99 GB/s


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

// RMSNorm Forward CPU Reference Code
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

void rmsnorm_backward_cpu(float *dinp, float *dweight, float *dbias,
                          const float *dout, const float *inp, const float *weight, const float *bias,
                          int B, int T, int C)
{
    float eps = 1e-5f;
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            const float *dout_bt = dout + b * T * C + t * C;
            const float *inp_bt = inp + b * T * C + t * C;
            float *dinp_bt = dinp + b * T * C + t * C;

            // Calculate the rms
            float rms = 0.0f;
            for (int i = 0; i < C; i++)
            {
                rms += inp_bt[i] * inp_bt[i];
            }
            rms = sqrtf(rms / C + eps);

            // First, calculate the gradients for the weights and biases
            // using `+=` for gradients accumuation.
            for (int i = 0; i < C; i++)
            {
                float norm = inp_bt[i] / rms;
                dbias[i] += dout_bt[i];
                dweight[i] += norm * dout_bt[i];
            }

            // Now, calculate the gradients for the inputs
            float drms = 0.0f;
            for (int i = 0; i < C; i++)
            {
                drms += inp_bt[i] * dout_bt[i] * weight[i];
            }
            drms = drms * (-1.0f / (rms * rms * rms * C));

            for (int i = 0; i < C; i++)
            {
                float norm = inp_bt[i] / rms;
                dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void rmsnorm_backward_kernel1(float *dinp, float *dweight, float *dbias,
                                         const float *dout, const float *inp, const float *weight, const float *bias,
                                         int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    float eps = 1e-5f;
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;

    // Calculate the rms
    float rms = 0.0f;
    for (int i = 0; i < C; i++)
    {
        rms += inp_bt[i] * inp_bt[i];
    }
    rms = sqrtf(rms / C + eps);

    // First, calculate the gradients for the weights and biases
    for (int i = 0; i < C; i++)
    {
        float norm = inp_bt[i] / rms;
        atomicAdd(&dbias[i], dout_bt[i]);
        atomicAdd(&dweight[i], norm * dout_bt[i]);
    }

    // Calculate drms
    float drms = 0.0f;
    for (int i = 0; i < C; i++)
    {
        drms += inp_bt[i] * dout_bt[i] * weight[i];
    }
    drms = drms * (-1.0f / (rms * rms * rms * C));

    // Now, calculate the gradients for the inputs
    for (int i = 0; i < C; i++)
    {
        dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
    }
}

// __restrict__ will hint the compiler to optimize the code better.
// Using __restrict__ will ensure no aliasing, and compiler can (freedom to) utilize maximum registers and stuff for optimization
__global__ void rmsnorm_backward_kernel2(float *__restrict__ dinp, float *__restrict__ dweight, float *__restrict__ dbias,
                                         const float *__restrict__ dout, const float *__restrict__ inp, const float *__restrict__ weight, const float *__restrict__ bias,
                                         int N, int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Calculate thread index within grid (each warp handles one row)
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
        return;

    const float eps = 1e-5f;
    const float *dout_bt = dout + idx * C;
    const float *inp_bt = inp + idx * C;
    float *dinp_bt = dinp + idx * C;

    // Compute the RMS using cooperative group reduction
    float sum_squares = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum_squares += inp_bt[i] * inp_bt[i];
    }
    sum_squares = cg::reduce(warp, sum_squares, cg::plus<float>());
    float rms = sqrtf(sum_squares / C + eps);

    // Calculate the gradients for the weights and biases (accumulated across threads)
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float norm = inp_bt[i] / rms;
        // Accumulate gradient for bias and weight using atomicAdd with warp-level synchronization
        atomicAdd(&dbias[i], dout_bt[i]);
        atomicAdd(&dweight[i], norm * dout_bt[i]);
    }

    // Compute drms (gradient with respect to rms)
    float drms = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        drms += inp_bt[i] * dout_bt[i] * weight[i];
    }
    drms = cg::reduce(warp, drms, cg::plus<float>());
    drms = drms * (-1.0f / (rms * rms * rms * C));

    // Step 4: Compute gradients for inputs
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void rmsnorm_backward1(float *dinp, float *dweight, float *dbias,
                       const float *dout, const float *inp, const float *weight, const float *bias,
                       int B, int T, int C, const int block_size)
{
    const int N = B * T;
    const int grid_size = ceil_div((N + block_size - 1), block_size); // equivalent to ceil(N / block_size)
    rmsnorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_backward2(float *dinp, float *dweight, float *dbias,
                       const float *dout, const float *inp, const float *weight, const float *bias,
                       int B, int T, int C, const int block_size)
{
    assert(block_size % 32 == 0); // Ensure block size is a multiple of warp size
    const int N = B * T;
    const int grid_size = ceil_div((N * 32 + block_size - 1), block_size);
    rmsnorm_backward_kernel2<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void rmsnorm_backward(int kernel_num,
                      float *dinp, float *dweight, float *dbias,
                      const float *dout, const float *inp, const float *weight, const float *bias,
                      int B, int T, int C,
                      const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        rmsnorm_backward1(dinp, dweight, dbias, dout, inp, weight, bias, B, T, C, block_size);
        break;
    case 2:
        rmsnorm_backward2(dinp, dweight, dbias, dout, inp, weight, bias, B, T, C, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768; // embed_dim

    // first do the forward pass in CPU
    float *out = (float *)malloc(B * T * C * sizeof(float));

    float *inp = make_random_float(B * T * C);
    float *weight = make_random_float(C);
    float *bias = make_random_float(C);

    rmsnorm_forward_cpu(out, inp, weight, bias, B, T, C);

    // now do the backward pass, again on CPU
    float *dout = make_random_float(B * T * C);
    float *dinp = make_zeros_float(B * T * C);
    float *dweight = make_zeros_float(C);
    float *dbias = make_zeros_float(C);

    rmsnorm_backward_cpu(dinp, dweight, dbias, dout, inp, weight, bias, B, T, C);

    // the above calculations act as the reference
    // now let's do the same on the GPU

    // read kernel_num from command line
    int kernel_num = 2;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move all the variables we need for backward pass onto the GPU
    float *d_dinp;
    float *d_dweight;
    float *d_dbias;
    float *d_dout;
    float *d_inp;
    float *d_weight;
    float *d_bias;

    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dweight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dbias, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));

    // copy over the "inputs" to the backward call
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_weight, weight, C));
    cudaCheck(memcpy_convert(d_bias, bias, C));

    // launch the kernel
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        // init the "outputs" of the backward call to zeros
        cudaCheck(cudaMemset(d_dinp, 0, B * T * C * sizeof(float)));
        cudaCheck(cudaMemset(d_dweight, 0, C * sizeof(float)));
        cudaCheck(cudaMemset(d_dbias, 0, C * sizeof(float)));

        rmsnorm_backward(kernel_num, d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_bias,
                         B, T, C, block_size);

        // check the correctness of the kernel
        float error_threshold_dinp = sizeof(float) == 4 ? 1e-3f : 1e-1f;    // allow larger errors for BF16/FP16
        float error_threshold_dparams = sizeof(float) == 4 ? 1e-3f : 5e-1f; // much, much larger...
        printf("Checking correctness...\n");
        printf("dinp:\n");
        validate_result(d_dinp, dinp, "dinp", B * T * C, error_threshold_dinp);
        printf("dweight:\n");
        validate_result(d_dweight, dweight, "dweight", C, error_threshold_dparams);
        printf("dbias:\n");
        validate_result(d_dbias, dbias, "dbias", C, error_threshold_dparams);

        printf("All results match for block_size=%d.\n\n", block_size);
    }

    // now time the kernel
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, rmsnorm_backward, kernel_num,
                                              d_dinp, d_dweight, d_dbias, d_dout, d_inp, d_weight, d_bias,
                                              B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = (2 * B * T * C) * 4; // *4 for float
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // cleanups
    free(out);

    free(inp);
    free(weight);
    free(bias);
    free(dout);
    free(dinp);
    free(dweight);
    free(dbias);
    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dweight));
    cudaCheck(cudaFree(d_dbias));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));
    return 0;
}