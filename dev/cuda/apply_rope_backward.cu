/*
Kernels for apply_rope forward pass.
The results shown are performed on L4-GPU.
The GPU Bandwidth memory

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt apply_rope_backward.cu -o apply_rope_backward

- version 1 is naive CPU port, parallelizes over B, T, NH, utilizes Coalesed Memory accesses
./apply_rope_backward 1

RESULTS:
- time 0.3319 ms | bandwidth 189.55 GB/s

- version 2 is same as version 1, but utilizes shared memory access for `freqs_cos` and `freqs_sin`
./apply_rope_backward 2
RESULTS:
- time 0.3296 ms | bandwidth 190.90 GB/s
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU-Version

// apply_rope backward pass.
// utlizes in-place gradient updates (dout info available in dq and dk already)
// The gradients for q and k are computed based solely on the cosine and sine values (freqs_cos, freqs_sin) and the gradients from the next layer (dq and dk).

void apply_rope_backward_cpu(
    float *dq, float *dk,                           // Gradients to be computed for q and k
    const float *freqs_cos, const float *freqs_sin, // Precomputed cosine and sine values
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    int half_hs = C_per_NH / 2;

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // Backprop for q (shape: B, T, num_kv_heads, C/NH)
            for (int kv_head = 0; kv_head < num_kv_heads; kv_head++)
            {
                for (int hs = 0; hs < half_hs; hs++)
                {
                    int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs;
                    int freq_index = t * half_hs + hs;

                    float cos_val = freqs_cos[freq_index];
                    float sin_val = freqs_sin[freq_index];

                    // Gradients from the next layer
                    float dq_r = dq[q_index];
                    float dq_i = dq[q_index + half_hs];

                    // Backpropagation using chain rule
                    dq[q_index] = dq_r * cos_val + dq_i * sin_val;           // (df/dq_r)
                    dq[q_index + half_hs] = dq_i * cos_val - dq_r * sin_val; // (df/dq_i)
                }
            }

            // Backprop for k (shape: B, T, NH, C/NH)
            for (int nh = 0; nh < NH; nh++)
            {
                for (int hs = 0; hs < half_hs; hs++)
                {
                    int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;
                    int freq_index = t * half_hs + hs;

                    float cos_val = freqs_cos[freq_index];
                    float sin_val = freqs_sin[freq_index];

                    // Gradients from the next layer
                    float dk_r = dk[k_index];
                    float dk_i = dk[k_index + half_hs];

                    // Backpropagation using chain rule
                    dk[k_index] = dk_r * cos_val + dk_i * sin_val;           // (df/dk_r)
                    dk[k_index + half_hs] = dk_i * cos_val - dk_r * sin_val; // (df/dk_i)
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPU Kernels

__global__ void apply_rope_backward_kernel1(
    float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    if (hs < half_hs) // Guard to handle only half_hs elements for real and imaginary pairs
    {
        int freq_index = t * half_hs + hs;

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        // Backprop for q (shape: B, T, num_kv_heads, C/NH)
        if (nh < num_kv_heads) // only the q heads are processed
        {
            int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + nh * C_per_NH + hs;

            // Gradients from the next layer (dout_q)
            float dq_r = dq[q_index];
            float dq_i = dq[q_index + half_hs];

            // Backpropagation using chain rule
            dq[q_index] = dq_r * cos_val + dq_i * sin_val;           // (df/dq_r)
            dq[q_index + half_hs] = dq_i * cos_val - dq_r * sin_val; // (df/dq_i)
        }

        // Backprop for k (shape: B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;

        // Gradients from the next layer
        float dk_r = dk[k_index];
        float dk_i = dk[k_index + half_hs];

        // Backpropagation using chain rule (dout_k)
        dk[k_index] = dk_r * cos_val + dk_i * sin_val;           // (df/dk_r)
        dk[k_index + half_hs] = dk_i * cos_val - dk_r * sin_val; // (df/dk_i)
    }
}

/**
 * Similar to Kernel-1 but uses Shared Memory for `freqs_cos` and `freqs_sin`
 * It may help us address our limiting perf factor (Memory Bandwidth), since we will be utilizing SRAM (less latency, faster memory)
 */
__global__ void apply_rope_backward_kernel2(
    float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    extern __shared__ float shared_mem[]; // Shared memory for freqs_cos and freqs_sin
    float *shared_freqs_cos = shared_mem;
    float *shared_freqs_sin = shared_mem + blockDim.x;

    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    // Each thread loads the necessary cos and sin values into shared memory
    if (hs < half_hs)
    {
        int freq_index = t * half_hs + hs;
        shared_freqs_cos[hs] = freqs_cos[freq_index];
        shared_freqs_sin[hs] = freqs_sin[freq_index];
    }

    __syncthreads(); // wait till all threads have loaded the shared memory

    if (hs < half_hs)
    {
        float cos_val = shared_freqs_cos[hs];
        float sin_val = shared_freqs_sin[hs];

        // Backprop for q (shape: B, T, num_kv_heads, C/NH)
        if (nh < num_kv_heads)
        {
            int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + nh * C_per_NH + hs;

            // Gradients from the next layer
            float dq_r = dq[q_index];
            float dq_i = dq[q_index + half_hs];

            // Backpropagation using chain rule (dout_q)
            dq[q_index] = dq_r * cos_val + dq_i * sin_val;           // (df/dq_r)
            dq[q_index + half_hs] = dq_i * cos_val - dq_r * sin_val; // (df/dq_i)
        }

        // Backprop for k (shape: B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;

        // Gradients from the next layer
        float dk_r = dk[k_index];
        float dk_i = dk[k_index + half_hs];

        // Backpropagation using chain rule (dout_k)
        dk[k_index] = dk_r * cos_val + dk_i * sin_val;           // (df/dk_r)
        dk[k_index + half_hs] = dk_i * cos_val - dk_r * sin_val; // (df/dk_i)
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void apply_rope_backward1(float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
                          int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    dim3 blocks(B, T, NH); // Parallelizing over B, T, NH
    int threads = C_per_NH / 2;

    apply_rope_backward_kernel1<<<blocks, threads>>>(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
    cudaDeviceSynchronize();
}

void apply_rope_backward2(float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
                          int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    dim3 blocks(B, T, NH); // Parallelizes over B, T, NH
    int threads = C_per_NH / 2;
    int shared_mem_size = 2 * (C_per_NH / 2) * sizeof(float); // Shared memory size for freqs_cos and freqs_sin

    apply_rope_backward_kernel2<<<blocks, threads, shared_mem_size>>>(
        dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
    cudaDeviceSynchronize();
}

// kernel version dispatch
void apply_rope_backward(int kernel_num,
                         float *dq, float *dk, const float *freqs_cos, const float *freqs_sin,
                         int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    switch (kernel_num)
    {
    case 1:
        apply_rope_backward1(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    case 2:
        apply_rope_backward2(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

int main(int argc, const char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 8;
    int num_kv_heads = 2;
    int C_per_NH = C / NH;

    // Randomize q, k, freqs_cos, freqs_sin
    float *dq = make_random_float(B * T * num_kv_heads * C_per_NH); // dq [B, T, num_kv_heads, C/NH]
    float *dk = make_random_float(B * T * NH * C_per_NH);           // dk [B, T, NH, C/NH]
    float *freqs_cos = make_random_float(T * (C_per_NH / 2));       // freqs_cos [T, C/2NH]
    float *freqs_sin = make_random_float(T * (C_per_NH / 2));       // freqs_sin shape [T, C/2NH]

    // Read kernel number from the command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // Move data to GPU
    float *d_dq;
    float *d_dk;
    float *d_freqs_cos;
    float *d_freqs_sin;
    cudaCheck(cudaMalloc(&d_dq, B * T * num_kv_heads * C_per_NH * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dk, B * T * NH * C_per_NH * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_cos, T * (C_per_NH / 2) * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_sin, T * (C_per_NH / 2) * sizeof(float)));

    cudaCheck(cudaMemcpy(d_dq, dq, B * T * num_kv_heads * C_per_NH * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dk, dk, B * T * NH * C_per_NH * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freqs_cos, freqs_cos, T * (C_per_NH / 2) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freqs_sin, freqs_sin, T * (C_per_NH / 2) * sizeof(float), cudaMemcpyHostToDevice));

    // Executing our CPU-Version (Backward-Pass)
    apply_rope_backward_cpu(dq, dk, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);

    // Since, we are utilizing coalesced Memory accesses where Block sizes are calculated using dims of `q` and `k`, so there is no need to check perf on different block_sizes
    // Validate kernel correctness
    apply_rope_backward(kernel_num, d_dq, d_dk, d_freqs_cos, d_freqs_sin, B, T, num_kv_heads, NH, C_per_NH);

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
    float tol = 1e-5;
#else
    float tol = 1e-2f;
#endif
    // Validate both q and k results
    validate_result(d_dq, dq, "dq", B * T * num_kv_heads * C_per_NH, tol);
    validate_result(d_dk, dk, "dk", B * T * NH * C_per_NH, tol);

    printf("All results match. Starting benchmarks.\n\n");

    // Benchmark the kernel for each block size
    int repeat_times = 1000; // Number of times to repeat for benchmarking

    float elapsed_time = benchmark_kernel(repeat_times, apply_rope_backward,
                                          kernel_num, d_dq, d_dk, d_freqs_cos, d_freqs_sin,
                                          B, T, num_kv_heads, NH, C_per_NH);

    // Estimate memory bandwidth achieved
    // e.g. A100 40GB PCIe is advertised at 1,555GB/s
    // e.g. NVIDIA L4 24GB PCIe is advertised at 300GB/s
    long memory_ops = (B * T * num_kv_heads * C_per_NH + B * T * NH * C_per_NH) * 2 * (int)sizeof(float);
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;

    printf("time %.4f ms | bandwidth %.2f GB/s\n", elapsed_time, memory_bandwidth);

    // Free allocated memory
    free(dq);
    free(dk);
    free(freqs_cos);
    free(freqs_sin);
    cudaCheck(cudaFree(d_dq));
    cudaCheck(cudaFree(d_dk));
    cudaCheck(cudaFree(d_freqs_cos));
    cudaCheck(cudaFree(d_freqs_sin));

    return 0;
}