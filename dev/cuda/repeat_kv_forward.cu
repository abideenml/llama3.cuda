/*
Kernels for Kernels for repeat_kv (repeat_interleavce over dim=2) forward pass.
NOTE: The results shown are performed on L4-GPU

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt repeat_kv_forward.cu -o repeat_kv_forward

- version 1 is naive CPU port: Parallelize over B, T, and head_dim
./repeat_kv_forward 1

RESULTS:
block_size   32 | time 0.3275 ms | bandwidth 192.10 GB/s
block_size   64 | time 0.3261 ms | bandwidth 192.93 GB/s
block_size  128 | time 0.3241 ms | bandwidth 194.14 GB/s
block_size  256 | time 0.3234 ms | bandwidth 194.52 GB/s
block_size  512 | time 0.3195 ms | bandwidth 196.93 GB/s

- version 2 - derived from version-1, parallelizes over B,T, num_kv_heads, and head_dim
./repeat_kv_forward 2

NOTE: This kernel utilizes a hard-coded block_size (=head_dim), so NO NEED to try it for different block_sizes

RESULTS:
time 0.3202 ms | bandwidth 196.50 GB/s

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void repeat_kv_forward_cpu(
    float *k_out, float *v_out, const float *k, const float *v,
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    // Calculate output shape
    int num_heads = num_kv_heads * num_queries_per_kv;

    // Repeat `k` along the num_kv_heads dimension
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
                        int in_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
                        int out_index = ((b * T + t) * num_heads + out_head) * head_dim + d;

                        k_out[out_index] = k[in_index];
                        v_out[out_index] = v[in_index];
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
 */
__global__ void repeat_kv_forward_kernel1(float *k_out, float *v_out, const float *k, const float *v,
                                          int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x + blockIdx.z * blockDim.x;

    if (d < head_dim) // Guard
    {
        // Iterate over kv_heads and repeat num_queries_per_kv times
        for (int kv_head = 0; kv_head < num_kv_heads; kv_head++)
        {
            for (int rep = 0; rep < num_queries_per_kv; rep++)
            {
                int out_head = kv_head * num_queries_per_kv + rep;

                int in_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
                int out_index = ((b * T + t) * (num_kv_heads * num_queries_per_kv) + out_head) * head_dim + d;

                k_out[out_index] = k[in_index];
                v_out[out_index] = v[in_index];
            }
        }
    }
}

/**
 * Extension of Kernel-1, adds more parallelization over num_kv_heads as well
 * Launches with a fixed block-size
 */
__global__ void repeat_kv_forward_kernel2(
    float *k_out, float *v_out, const float *k, const float *v,
    int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{

    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int d = threadIdx.x;

    if (d < head_dim)
    {
        // Each thread will now handle one specific kv_head and repeat it
        for (int rep = 0; rep < num_queries_per_kv; rep++)
        {
            int out_head = kv_head * num_queries_per_kv + rep;

            // Calculate input and output indices
            int in_index = ((b * T + t) * num_kv_heads + kv_head) * head_dim + d;
            int out_index = ((b * T + t) * (num_kv_heads * num_queries_per_kv) + out_head) * head_dim + d;

            // Copy values for both k and v
            k_out[out_index] = k[in_index];
            v_out[out_index] = v[in_index];
        }
    }
}
// ----------------------------------------------------------------------------
// kernel launcher

void repeat_kv_forward1(float *k_out, float *v_out, const float *k, const float *v,
                        int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim, const int block_size)
{
    int grid_nh = ceil_div((head_dim + block_size - 1), block_size); // num of blocks to parallelize over head_dim
    dim3 blocks(B, T, grid_nh);

    repeat_kv_forward_kernel1<<<blocks, block_size>>>(k_out, v_out, k, v, B, T, num_kv_heads, num_queries_per_kv, head_dim);
    cudaDeviceSynchronize();
}

void repeat_kv_forward2(float *k_out, float *v_out, const float *k, const float *v,
                        int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    dim3 blocks(B, T, num_kv_heads);
    int block_size = head_dim;

    repeat_kv_forward_kernel2<<<blocks, block_size>>>(k_out, v_out, k, v, B, T, num_kv_heads, num_queries_per_kv, head_dim);
    cudaDeviceSynchronize();
}

// kernel version dispatch
void repeat_kv_forward(int kernel_num,
                       float *k_out, float *v_out, const float *k, const float *v,
                       int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim, const int block_size)
{
    switch (kernel_num)
    {
    case 1:
        repeat_kv_forward1(k_out, v_out, k, v, B, T, num_kv_heads, num_queries_per_kv, head_dim, block_size);
        break;
    case 2:
        repeat_kv_forward2(k_out, v_out, k, v, B, T, num_kv_heads, num_queries_per_kv, head_dim);
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

    // create host memory of random numbers
    float *k = make_random_float(B * T * num_kv_heads * head_dim);                                        // k shape [B, T, num_kv_heads, head_dim]
    float *v = make_random_float(B * T * num_kv_heads * head_dim);                                        // v shape [B, T, num_kv_heads, head_dim]
    float *k_out = (float *)malloc(B * T * num_kv_heads * num_queries_per_kv * head_dim * sizeof(float)); // Output shape: [B, T, num_heads, head_dim]
    float *v_out = (float *)malloc(B * T * num_kv_heads * num_queries_per_kv * head_dim * sizeof(float)); // Output shape: [B, T, num_heads, head_dim]

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // move to GPU
    float *d_k, *d_v, *d_k_out, *d_v_out;
    cudaCheck(cudaMalloc(&d_k_out, B * T * num_kv_heads * num_queries_per_kv * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v_out, B * T * num_kv_heads * num_queries_per_kv * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_k, B * T * num_kv_heads * head_dim * sizeof(float)));
    cudaCheck(cudaMalloc(&d_v, B * T * num_kv_heads * head_dim * sizeof(float)));
    cudaCheck(cudaMemcpy(d_k, k, B * T * num_kv_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v, v, B * T * num_kv_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));

    // CPU-Version, to check the correctness
    repeat_kv_forward_cpu(k_out, v_out, k, v, B, T, num_kv_heads, num_queries_per_kv, head_dim);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        repeat_kv_forward(kernel_num, d_k_out, d_v_out, d_k, d_v, B, T, num_kv_heads, num_queries_per_kv, head_dim, block_size);

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_k_out, k_out, "k_out", B * T * num_kv_heads * num_queries_per_kv * head_dim, tol);
        validate_result(d_v_out, v_out, "v_out", B * T * num_kv_heads * num_queries_per_kv * head_dim, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, repeat_kv_forward,
                                              kernel_num, d_k_out, d_v_out, d_k, d_v,
                                              B, T, num_kv_heads, num_queries_per_kv, head_dim,
                                              block_size);

        // napkin math: estimate the memory bandwidth achieved
        // Estimate memory bandwidth achieved
        // e.g. A100 40GB PCIe is advertised at 1,555GB/s
        // e.g. NVIDIA L4 24GB PCIe is advertised at 300GB/s

        // for each output element, we do 2 reads (k, v) and 2 writes (k_out, v_out), 4 bytes each (float32)
        long memory_ops = (B * T * num_kv_heads * num_queries_per_kv * head_dim + B * T * num_kv_heads * head_dim) * 2 * sizeof(float); // 2 reads, 2 writes
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // Free host memory
    free(k);
    free(v);
    free(k_out);
    free(v_out);

    // Free device memory
    cudaCheck(cudaFree(d_k));
    cudaCheck(cudaFree(d_v));
    cudaCheck(cudaFree(d_k_out));
    cudaCheck(cudaFree(d_v_out));

    return 0;
}
