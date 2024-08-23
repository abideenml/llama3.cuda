/*
Kernels for apply_rope forward pass.
The results shown are performed on L4-GPU.
The GPU Bandwidth memory

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt apply_rope_forward.cu -o apply_rope_forward

- version 1 is naive CPU port, parallelizes over B, T, NH, utilizes Coalesed Memory accesses
./apply_rope_forward 1

RESULTS:
time 0.3262 ms | bandwidth 192.89 GB/s


- version 2 is same as version 1, but with different kernel launches for `q`, and `k`, thus reducing Warp-Divergence
./apply_rope_forward 2
RESULTS:
- time 0.3267 ms | bandwidth 192.58 GB/s

- version 3 is same as version 2, but utilizes shared memory for faster access for `freq_cos` and `freq_sin`
./apply_rope_forward 3
RESULTS:
- time 0.3285 ms | bandwidth 191.52 GB/s
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU-Version
// A naive cpu implementation, sequentially traverses every query(Q) and key(K) values, and applies RoPE to it, and stores it back in their respective Matrices
void apply_rope_forward_cpu(
    float *q, float *k, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    // Loop over the batch, sequence, and heads
    int half_hs = C_per_NH / 2;

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            // Apply RoPE to q (shape: B, T, num_kv_heads, C/NH)
            for (int kv_head = 0; kv_head < num_kv_heads; kv_head++)
            {
                for (int hs = 0; hs < half_hs; hs++)
                {
                    // Indexing for q
                    int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs;
                    int freq_index = t * half_hs + hs;

                    float cos_val = freqs_cos[freq_index];
                    float sin_val = freqs_sin[freq_index];

                    // Get the real and imaginary parts of q
                    float q_r = q[q_index];
                    float q_i = q[q_index + half_hs];

                    // Apply rotation to q
                    q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
                    q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc)
                }
            }

            // Apply RoPE to k (shape: B, T, NH, C/NH)
            for (int nh = 0; nh < NH; nh++)
            {
                for (int hs = 0; hs < half_hs; hs++)
                {
                    // Indexing for k
                    int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;
                    int freq_index = t * half_hs + hs;

                    float cos_val = freqs_cos[freq_index];
                    float sin_val = freqs_sin[freq_index];

                    // Get the real and imaginary parts of k
                    float k_r = k[k_index];
                    float k_i = k[k_index + half_hs];

                    // Apply rotation to k
                    k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
                    k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc)
                }
            }
        }
    }
}

/**
 * Derived from the CPU PORT of the apply_rope kernel. Utilized Coalesced Memory access for `q`, `k`,
 * - Applies RoPE to `q` and `k` separately.
 * - Each thread handles a real/imaginary pair
 * - Can be optimized more, since we can warp-divergence (because of the if condition), making some threads become idle
 */
__global__ void apply_rope_forward_kernel1(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    // Separate indexing for q and k based on their respective shapes
    if (hs < half_hs)
    {
        // Query (q) index for num_kv_heads shape (B, T, num_kv_heads, C/NH)
        if (kv_head < num_kv_heads)
        {
            // coalesced memory accesses for `q`
            int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs;

            // Frequency index (T, C/2NH)
            int freq_index = t * half_hs + hs;

            float cos_val = freqs_cos[freq_index];
            float sin_val = freqs_sin[freq_index];

            // Apply RoPE to q (query)
            float q_r = q[q_index];
            float q_i = q[q_index + half_hs];

            q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
            q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i
        }

        // Key (k) index for NH shape (B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + kv_head * C_per_NH + hs;

        // Apply RoPE to k (key)
        int freq_index = t * half_hs + hs;
        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        float k_r = k[k_index];
        float k_i = k[k_index + half_hs];

        k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
}

// ----------------------------------------------------------------------------

/**
 * Below, in order to remove the warp-divergence, we are separating the kernels for `q` and `k`
 * - Utilizes coalesced memory access for `q`, `k`, `freq_cos`, and `freq_sin`
 * Each thread handles a real/imaginary pair for `q`, and `k`(in their respective kernels)
 */
__global__ void apply_rope_forward_q1(
    float *q, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int C_per_NH)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    if (hs < half_hs)
    {
        int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs; // Query (q) index for num_kv_heads shape (B, T, num_kv_heads, C/NH)
        int freq_index = t * half_hs + hs;                                                                     // Frequency index (T, C/2NH)

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];
        float q_r = q[q_index];
        float q_i = q[q_index + half_hs];

        // Apply RoPE to q (query)
        q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
        q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i
    }
}

__global__ void apply_rope_forward_k1(
    float *k, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int NH, int C_per_NH)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    if (hs < half_hs)
    {

        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs; // Key (k) index for NH shape (B, T, NH, C/NH)
        int freq_index = t * half_hs + hs;                                            // Frequency index (T, C/2NH)

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];
        float k_r = k[k_index];
        float k_i = k[k_index + half_hs];

        // Apply RoPE to k (key)
        k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
}
// ----------------------------------------------------------------------------

/**
 * Verion-1 and version 2, are of same perf (no as such significant performance increase due to warp-divergence), since the kernels are memory bandwidth bound
 *  These kernels use shared memory to store `freqs_cos` and `freqs_sin` values (frequently accessed in the computation).
 *
 * Each thread loads one cos and one sin value, so the total size of shared memory is 2 * blockDim.x * sizeof(float).
 */

__global__ void apply_rope_forward_q2(
    float *q, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int num_kv_heads, int C_per_NH)
{
    extern __shared__ float shared_mem[];              // Shared memory for freqs_cos and freqs_sin
    float *shared_freqs_cos = shared_mem;              // First part of shared memory for freqs_cos
    float *shared_freqs_sin = shared_mem + blockDim.x; // Second part for freqs_sin

    int b = blockIdx.x;
    int t = blockIdx.y;
    int kv_head = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    // Load freqs_cos and freqs_sin into shared memory for reuse
    if (hs < half_hs)
    {
        int freq_index = t * half_hs + hs;
        shared_freqs_cos[hs] = freqs_cos[freq_index];
        shared_freqs_sin[hs] = freqs_sin[freq_index];
    }

    __syncthreads(); // Ensure all threads have loaded shared memory before proceeding

    if (hs < half_hs)
    {
        // Query (q) index for num_kv_heads shape (B, T, num_kv_heads, C/NH)
        int q_index = b * T * num_kv_heads * C_per_NH + t * num_kv_heads * C_per_NH + kv_head * C_per_NH + hs;

        // Apply RoPE to q (query)
        float q_r = q[q_index];
        float q_i = q[q_index + half_hs];

        // Use shared memory for cos and sin values
        float cos_val = shared_freqs_cos[hs];
        float sin_val = shared_freqs_sin[hs];

        q[q_index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
        q[q_index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i
    }
}

__global__ void apply_rope_forward_k2(
    float *k, const float *freqs_cos, const float *freqs_sin,
    int B, int T, int NH, int C_per_NH)
{
    extern __shared__ float shared_mem[];              // Shared memory for freqs_cos and freqs_sin
    float *shared_freqs_cos = shared_mem;              // First part of shared memory for freqs_cos
    float *shared_freqs_sin = shared_mem + blockDim.x; // Second part for freqs_sin

    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    // Half of the head size (real and imaginary components)
    int half_hs = C_per_NH / 2;

    // Load freqs_cos and freqs_sin into shared memory for reuse
    if (hs < half_hs)
    {
        int freq_index = t * half_hs + hs;
        shared_freqs_cos[hs] = freqs_cos[freq_index];
        shared_freqs_sin[hs] = freqs_sin[freq_index];
    }

    __syncthreads(); // Ensure all threads have loaded shared memory before proceeding

    if (hs < half_hs)
    {
        // Key (k) index for NH shape (B, T, NH, C/NH)
        int k_index = b * T * NH * C_per_NH + t * NH * C_per_NH + nh * C_per_NH + hs;

        // Apply RoPE to k (key)
        float k_r = k[k_index];
        float k_i = k[k_index + half_hs];

        // Use shared memory for cos and sin values
        float cos_val = shared_freqs_cos[hs];
        float sin_val = shared_freqs_sin[hs];

        k[k_index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[k_index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void apply_rope_forward1(float *q, float *k, float *freqs_cos, float *freqs_sin, int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    dim3 blocks(B, T, NH);
    int threads = C_per_NH / 2;

    apply_rope_forward_kernel1<<<blocks, threads>>>(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
    cudaDeviceSynchronize();
}

void apply_rope_forward2(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    // Separate kernel launches for `q` and `k` to avoid warp-divergence

    dim3 blocks_q(B, T, num_kv_heads); // For q (shape: B, T, num_kv_heads, C/NH)
    dim3 blocks_k(B, T, NH);           // For k (shape: B, T, NH, C/NH)

    int block_size = C_per_NH / 2;

    apply_rope_forward_q1<<<blocks_q, block_size>>>(q, freqs_cos, freqs_sin, B, T, num_kv_heads, C_per_NH);
    apply_rope_forward_k1<<<blocks_k, block_size>>>(k, freqs_cos, freqs_sin, B, T, NH, C_per_NH);
    cudaDeviceSynchronize();
}

void apply_rope_forward3(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    // Separate kernel launches for `q` and `k` with shared memory for `freqs_cos` and `freqs_sin`

    dim3 blocks_q(B, T, num_kv_heads); // For q (shape: B, T, num_kv_heads, C/NH)
    dim3 blocks_k(B, T, NH);           // For k (shape: B, T, NH, C/NH)

    int block_size = C_per_NH / 2;

    size_t shared_mem_size = 2 * block_size * sizeof(float); // Shared memory for cos and sin values

    apply_rope_forward_q2<<<blocks_q, block_size, shared_mem_size>>>(q, freqs_cos, freqs_sin, B, T, num_kv_heads, C_per_NH);
    apply_rope_forward_k2<<<blocks_k, block_size, shared_mem_size>>>(k, freqs_cos, freqs_sin, B, T, NH, C_per_NH);
    cudaDeviceSynchronize();
}

// kernel version dispatch
void apply_rope_forward(int kernel_num,
                        float *q, float *k, float *freqs_cos, float *freqs_sin,
                        int B, int T, int num_kv_heads, int NH, int C_per_NH)
{
    switch (kernel_num)
    {
    case 1:
        apply_rope_forward1(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    case 2:
        apply_rope_forward2(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
        break;
    case 3:
        apply_rope_forward3(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);
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
    float *q = make_random_float(B * T * num_kv_heads * C_per_NH); // q [B, T, num_kv_heads, C/NH]
    float *k = make_random_float(B * T * NH * C_per_NH);           // k [B, T, NH, C/NH]
    float *freqs_cos = make_random_float(T * (C_per_NH / 2));      // freqs_cos [T, C/2NH]
    float *freqs_sin = make_random_float(T * (C_per_NH / 2));      // freqs_sin shape [T, C/2NH]

    // Read kernel number from the command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // Move data to GPU
    float *d_q;
    float *d_k;
    float *d_freqs_cos;
    float *d_freqs_sin;
    cudaCheck(cudaMalloc(&d_q, B * T * num_kv_heads * C_per_NH * sizeof(float)));
    cudaCheck(cudaMalloc(&d_k, B * T * NH * C_per_NH * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_cos, T * (C_per_NH / 2) * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_sin, T * (C_per_NH / 2) * sizeof(float)));

    cudaCheck(cudaMemcpy(d_q, q, B * T * num_kv_heads * C_per_NH * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_k, k, B * T * NH * C_per_NH * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freqs_cos, freqs_cos, T * (C_per_NH / 2) * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freqs_sin, freqs_sin, T * (C_per_NH / 2) * sizeof(float), cudaMemcpyHostToDevice));

    // Executing our CPU-Version
    apply_rope_forward_cpu(q, k, freqs_cos, freqs_sin, B, T, num_kv_heads, NH, C_per_NH);

    // Since, we are utilizing coalesced Memory accesses where Block sizes are calculated using dims of `q` and `k`, so there is no need to check perf on different block_sizes
    // Validate kernel correctness
    apply_rope_forward(kernel_num, d_q, d_k, d_freqs_cos, d_freqs_sin, B, T, num_kv_heads, NH, C_per_NH);

#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
    float tol = 1e-5;
#else
    float tol = 1e-2f;
#endif
    // Validate both q and k results
    validate_result(d_q, q, "q", B * T * num_kv_heads * C_per_NH, tol);
    validate_result(d_k, k, "k", B * T * NH * C_per_NH, tol);

    printf("All results match. Starting benchmarks.\n\n");

    // Benchmark the kernel for each block size
    int repeat_times = 1000; // Number of times to repeat for benchmarking

    float elapsed_time = benchmark_kernel(repeat_times, apply_rope_forward,
                                          kernel_num, d_q, d_k, d_freqs_cos, d_freqs_sin,
                                          B, T, num_kv_heads, NH, C_per_NH);

    // Estimate memory bandwidth achieved
    // e.g. A100 40GB PCIe is advertised at 1,555GB/s
    // e.g. NVIDIA L4 24GB PCIe is advertised at 300GB/s
    long memory_ops = (B * T * num_kv_heads * C_per_NH + B * T * NH * C_per_NH) * 2 * (int)sizeof(float);
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;

    printf("time %.4f ms | bandwidth %.2f GB/s\n", elapsed_time, memory_bandwidth);

    // Free allocated memory
    free(q);
    free(k);
    free(freqs_cos);
    free(freqs_sin);
    cudaCheck(cudaFree(d_q));
    cudaCheck(cudaFree(d_k));
    cudaCheck(cudaFree(d_freqs_cos));
    cudaCheck(cudaFree(d_freqs_sin));

    return 0;
}