/*
Kernels for Kernels for repeat_kv (repeat_interleavce over dim=2) forward pass.
NOTE: The results shown are performed on L4-GPU

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt repeat_kv_backward.cu -o repeat_kv_backward

- version 1 is naive CPU port: Parallelize over B, T, and head_dim
./repeat_kv_backward 1

RESULTS:
block_size   32 | time 0.3331 ms | bandwidth 188.89 GB/s
block_size   64 | time 0.3196 ms | bandwidth 196.85 GB/s
block_size  128 | time 0.3024 ms | bandwidth 208.02 GB/s
block_size  256 | time 0.3011 ms | bandwidth 208.97 GB/s
block_size  512 | time 0.3037 ms | bandwidth 207.14 GB/s

- version 2 - derived from version-1, parallelizes over B,T, num_kv_heads, and head_dim
./repeat_kv_backward 2

NOTE: This kernel utilizes a hard-coded block_size (=head_dim), so NO NEED to try it for different block_sizes

RESULTS:
time 0.3046 ms | bandwidth 206.58 GB/s

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code referenc

void repeat_kv_backward_cpu(
    float *dk, float *dv,                     // Gradients for original k and v
    const float *dk_rep, const float *dv_rep, // Gradients from the next layer (repeated k and v)
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    // Iterate over the B, T, and num_kv_heads
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int kv_head = 0; kv_head < num_kv_heads; kv_head++)
            {
                for (int rep = 0; rep < num_queries_per_kv; rep++)
                {
                    int out_head = kv_head * num_queries_per_kv + rep;

                    for (int d = 0; d < head_dim; d++)
                    {
                        int dk_rep_index = ((b * T + t) * (num_kv_heads * num_queries_per_kv) + out_head) * head_dim + d;
                        int dk_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;

                        // Sum the gradients from the repeated values
                        dk[dk_index] += dk_rep[dk_rep_index];
                        dv[dk_index] += dv_rep[dk_rep_index];
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

/**
 * Basic implementation, adapted from CPU implementation
 * Parallelize over B, T, and head_dim (split across z blocks)
 *
 * Sums the gradients `dk_rep` and `dv_rep` across the repeated heads (num_queries_per_kv), and
 * accumulates gradients using `atomicAdd` to ensure no data races when multiple threads write to the same location in dk and dv.
 */
__global__ void repeat_kv_backward_kernel1(
    float *dk, float *dv, const float *dk_rep, const float *dv_rep,
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x + blockIdx.z * blockDim.x; // Parallelize over head_dim split across z blocks

    if (d < head_dim) // Guard for out-of-bounds threads
    {
        // Loop over kv_heads and accumulate gradients from repeated heads
        for (int kv_head = 0; kv_head < num_kv_heads; kv_head++)
        {
            // Initialize accumulation variables for dk and dv
            float dk_accum = 0.0f;
            float dv_accum = 0.0f;

            // Accumulate gradients across the repeated heads
            for (int rep = 0; rep < num_queries_per_kv; rep++)
            {
                int out_head = kv_head * num_queries_per_kv + rep;
                int dk_rep_index = ((b * T + t) * (num_kv_heads * num_queries_per_kv) + out_head) * head_dim + d;

                // Sum gradients from the repeated heads
                dk_accum += dk_rep[dk_rep_index];
                dv_accum += dv_rep[dk_rep_index];
            }

            // Write the accumulated gradients to the output dk and dv
            int dk_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
            dk[dk_index] = dk_accum;
            dv[dk_index] = dv_accum;
        }
    }
}

/**
 * Extension of Kernel-1, adds more parallelization over num_kv_heads as well
 * Launches with a fixed block-size
 */
__global__ void repeat_kv_backward_kernel2(
    float *dk, float *dv, const float *dk_rep, const float *dv_rep,
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z; // Each block processes a specific kv_head
    int d = threadIdx.x;      // handles individual elements in the head_dim

    if (d < head_dim) // Guard against over-indexing
    {
        // accumulation variables for dk and dv
        float dk_accum = 0.0f;
        float dv_accum = 0.0f;

        //  gradients from repeated queries
        for (int rep = 0; rep < num_queries_per_kv; rep++)
        {
            int out_head = kv_head * num_queries_per_kv + rep;
            int in_index = ((b * T + t) * num_kv_heads * num_queries_per_kv + out_head) * head_dim + d;

            // Sum gradients from the repeated heads
            dk_accum += dk_rep[in_index];
            dv_accum += dv_rep[in_index];
        }

        // Writing accumulated gradients to dk and dv
        int out_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
        dk[out_index] = dk_accum;
        dv[out_index] = dv_accum;
    }
}
// ----------------------------------------------------------------------------
// kernel launcher

void repeat_kv_backward1(float *dk, float *dv, const float *dk_rep, const float *dv_rep,
                         int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim, const int block_size)
{
    int grid_nh = ceil_div((head_dim + block_size - 1), block_size); // num of blocks to parallelize over head_dim
    dim3 blocks(B, T, grid_nh);

    repeat_kv_backward_kernel1<<<blocks, block_size>>>(dk, dv, dk_rep, dv_rep, B, T, num_kv_heads, num_queries_per_kv, head_dim);
    cudaDeviceSynchronize();
}

void repeat_kv_backward2(float *dk, float *dv, const float *dk_rep, const float *dv_rep,
                         int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    dim3 blocks(B, T, num_kv_heads);
    int block_size = head_dim; // Fixed block size

    repeat_kv_backward_kernel2<<<blocks, block_size>>>(dk, dv, dk_rep, dv_rep, B, T, num_kv_heads, num_queries_per_kv, head_dim);
    cudaDeviceSynchronize();
}

void repeat_kv_backward(int kernel_num,
                        float *dk, float *dv, const float *dk_rep, const float *dv_rep,
                        int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim, const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        repeat_kv_backward1(dk, dv, dk_rep, dv_rep, B, T, num_kv_heads, num_queries_per_kv, head_dim, block_size);
        break;
    case 2:
        repeat_kv_backward2(dk, dv, dk_rep, dv_rep, B, T, num_kv_heads, num_queries_per_kv, head_dim);
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
    int num_heads = 8;
    int num_kv_heads = 2;
    int num_queries_per_kv = num_heads / num_kv_heads;
    int head_dim = C / num_heads;

    // create host memory of random numbers for gradients
    float *dk_rep = make_random_float(B * T * num_heads * head_dim);              // dk_rep shape [B, T, num_heads, head_dim]
    float *dv_rep = make_random_float(B * T * num_heads * head_dim);              // dv_rep shape [B, T, num_heads, head_dim]
    float *dk = (float *)malloc(B * T * num_kv_heads * head_dim * sizeof(float)); // dk shape: [B, T, num_kv_heads, head_dim]
    float *dv = (float *)malloc(B * T * num_kv_heads * head_dim * sizeof(float)); // dv shape: [B, T, num_kv_heads, head_dim]

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move to GPU
    float *d_dk_rep, *d_dv_rep, *d_dk, *d_dv;
    cudaCheck(cudaMalloc(&d_dk, B * T * num_kv_heads * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dv, B * T * num_kv_heads * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dk_rep, B * T * num_heads * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dv_rep, B * T * num_heads * head_dim * sizeof(float)));
    cudaCheck(cudaMemcpy(d_dk_rep, dk_rep, B * T * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dv_rep, dv_rep, B * T * num_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));

    // CPU-Version, to check the correctness
    repeat_kv_backward_cpu(dk, dv, dk_rep, dv_rep, B, T, num_kv_heads, num_queries_per_kv, head_dim);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        repeat_kv_backward(kernel_num, d_dk, d_dv, d_dk_rep, d_dv_rep, B, T, num_kv_heads, num_queries_per_kv, head_dim, block_size);

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dk, dk, "dk", B * T * num_kv_heads * head_dim, tol);
        validate_result(d_dv, dv, "dv", B * T * num_kv_heads * head_dim, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, repeat_kv_backward,
                                              kernel_num, d_dk, d_dv, d_dk_rep, d_dv_rep,
                                              B, T, num_kv_heads, num_queries_per_kv, head_dim,
                                              block_size);

        // napkin math: estimate the memory bandwidth achieved
        // Estimate memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        // e.g. NVIDIA L4 24GB PCIe is advertised at 300GB/s

        // for each input element, we do 2 reads (dk_rep, dv_rep) and 2 writes (dk, dv), 4 bytes each (float32)
        long memory_ops = (B * T * num_heads * head_dim + B * T * num_kv_heads * head_dim) * 2 * sizeof(float); // 2 reads, 2 writes
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // Free host memory
    free(dk_rep);
    free(dv_rep);
    free(dk);
    free(dv);

    // Free device memory
    cudaCheck(cudaFree(d_dk_rep));
    cudaCheck(cudaFree(d_dv_rep));
    cudaCheck(cudaFree(d_dk));
    cudaCheck(cudaFree(d_dv));

    return 0;
}