/*
Kernels for attention_gqa (repeat_interleavce over dim=2) forward pass.
NOTE: The results shown are performed on L4-GPU

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt attention_forward_gqa.cu -o attention_forward_gqa

- version 2 - derived from version-1, parallelizes over B,T, num_kv_heads, and head_dim
./attention_forward_gqa 2

RESULTS:
block_size   32 | time 0.1711 ms | bandwidth 220.62 GB/s


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

// ----------------------------------------------------------------------------
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

void repeat_kv_forward2(float *k_out, float *v_out, const float *k, const float *v,
                        int B, int T, int num_kv_heads, int num_queries_per_kv, int head_dim)
{
    dim3 blocks(B, T, num_kv_heads);
    int block_size = head_dim;

    repeat_kv_forward_kernel2<<<blocks, block_size>>>(k_out, v_out, k, v, B, T, num_kv_heads, num_queries_per_kv, head_dim);
    cudaDeviceSynchronize();
}

__global__ void permute_kernel(float *q, float *k, float *v, const float *inp,
                               int B, int T, int NH, int num_kv_heads, int HS)
{
    // Compute the total number of elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * (NH + 2 * num_kv_heads) * T * HS; // Total threads across B, NH, num_kv_heads, T, HS

    // Check bounds
    if (idx >= total_threads)
        return;

    // For Q: (B, NH, T, HS)
    if (idx < B * NH * T * HS)
    {
        int b = idx / (NH * T * HS);
        int nh = (idx / (T * HS)) % NH;
        int t = (idx / HS) % T;
        int hs = idx % HS;

        int q_index = (b * NH * T + nh * T + t) * HS + hs;
        int inp_index = (b * (NH + 2 * num_kv_heads) * T + nh * T + t) * HS + hs;

        q[q_index] = inp[inp_index];
    }
    // For K: (B, num_kv_heads, T, HS)
    else if (idx < B * (NH + num_kv_heads) * T * HS)
    {
        int k_idx = idx - B * NH * T * HS;
        int b = k_idx / (num_kv_heads * T * HS);
        int kv = (k_idx / (T * HS)) % num_kv_heads;
        int t = (k_idx / HS) % T;
        int hs = k_idx % HS;

        int k_index = (b * num_kv_heads * T + kv * T + t) * HS + hs;
        int inp_index = (b * (NH + 2 * num_kv_heads) * T + (NH + kv) * T + t) * HS + hs;

        k[k_index] = inp[inp_index];
    }
    // For V: (B, num_kv_heads, T, HS)
    else
    {
        int v_idx = idx - B * (NH + num_kv_heads) * T * HS;
        int b = v_idx / (num_kv_heads * T * HS);
        int kv = (v_idx / (T * HS)) % num_kv_heads;
        int t = (v_idx / HS) % T;
        int hs = v_idx % HS;

        int v_index = (b * num_kv_heads * T + kv * T + t) * HS + hs;
        int inp_index = (b * (NH + 2 * num_kv_heads) * T + (NH + num_kv_heads + kv) * T + t) * HS + hs;

        v[v_index] = inp[inp_index];
    }
}

// ----------------------------------------------------------------------------
__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH, int d)
{
    // out has shape (B, nh, N, d) but we need to unpermute it to (B, N, nh, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // out[b][n][nh_][d_] <- inp[b][nh_][n][d_]
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        out[other_idx] = __ldcs(&inp[idx]);
    }
}

// ----------------------------------------------------------------------------
__device__ float &vec_at(float4 &vec, int index)
{
    return reinterpret_cast<float *>(&vec)[index];
}

__device__ float vec_at(const float4 &vec, int index)
{
    return reinterpret_cast<const float *>(&vec)[index];
}
// ----------------------------------------------------------------------------

__global__ void softmax_forward_kernel5(float *out, float inv_temperature, const float *inp, int N, int T)
{
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
    namespace cg = cooperative_groups;
    assert(T % 4 == 0);
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // micro-optimization: we iterate backwards so that
    // after the softmax backward operation completes, the cache retains the
    // part of the matrix close to the upper left corner, which benefits the
    // matmul operation that immediately follows.
    // int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank(); // forward order
    int idx = (gridDim.x - blockIdx.x - 1) * warp.meta_group_size() + warp.meta_group_rank(); // backward order
    if (idx >= N * T)
    {
        return;
    }
    int own_pos = idx % T;
    int pos_by_4 = own_pos / 4;

    // one row of inp, i.e. inp[idx, :] of shape (T,)
    const float *x = inp + idx * T;

    // not INF, so we don't get NaNs accidentally when subtracting two values.
    float maxval = -FLT_MAX;
    float sumval = 0.0f;

    const float4 *x_vec = reinterpret_cast<const float4 *>(x);
    for (int i = warp.thread_rank(); i < pos_by_4; i += warp.size())
    {
        float4 v = x_vec[i];
        float old_maxval = maxval;
        for (int k = 0; k < 4; ++k)
        {
            maxval = fmaxf(maxval, vec_at(v, k));
        }
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        for (int k = 0; k < 4; ++k)
        {
            sumval += expf(inv_temperature * (vec_at(v, k) - maxval));
        }
    }

    if (4 * pos_by_4 + warp.thread_rank() <= own_pos)
    {
        float old_maxval = maxval;
        maxval = fmaxf(maxval, x[4 * pos_by_4 + warp.thread_rank()]);
        sumval *= expf(inv_temperature * (old_maxval - maxval));
        sumval += expf(inv_temperature * (x[4 * pos_by_4 + warp.thread_rank()] - maxval));
    }

    float global_maxval = cg::reduce(warp, maxval, cg::greater<float>{});
    sumval *= expf(inv_temperature * (maxval - global_maxval));

    float sum = cg::reduce(warp, sumval, cg::plus<float>{});
    float norm = 1.f / sum;

    // divide the whole row by the sum
    for (int i = warp.thread_rank(); i <= own_pos; i += warp.size())
    {
        // recalculation is faster than doing the round-trip through memory.
        float ev = expf(inv_temperature * (__ldcs(x + i) - global_maxval));
        __stcs(out + idx * T + i, ev * norm);
    }
}
// ----------------------------------------------------------------------------

void attention_forward_gqa(float *out, float *qkvr, float *preatt, float *att, float *inp,
                           float *freqs_cos, float *freqs_sin,
                           int B, int T, int C, int NH, int num_kv_heads, int HS, int num_queries_per_kv)
{
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, (NH + 2*num_kv_heads) * HS) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)

    // permute and separate inp from (B, T, NH+2*num_kv_heads, HS) to 1X (B, NH, T, HS) and 2X (B, num_kv, T, HS)
    float *q, *k, *v;
    q = qkvr;                                         // q will have shape (B, T, NH * HS)
    k = qkvr + 1 * B * T * NH * HS;                   // k will have shape (B, T, num_kv_heads * HS)
    v = qkvr + B * T * (NH * HS + num_kv_heads * HS); // V will have shape (B, T, num_kv_heads * HS)

    int total_threads = B * T * (NH + 2 * num_kv_heads) * HS;
    int num_blocks = ceil_div(total_threads, block_size);

    // okay so now, this kernel wants Q,K,V to be of their respective shapes: 1X (B, NH, T, HS) and 2X (B, num_kv, T, HS)
    // but instead, we have a single tensor QKV (inp) of shape (B, T, NH+2*num_kv_heads, HS)
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, num_kv_heads, HS);
    cudaCheck(cudaGetLastError());

    // Applyiing rope to q, and k matrices
    dim3 blocks_q(B, T, num_kv_heads); // For q (shape: B, T, num_kv_heads, C/NH)
    dim3 blocks_k(B, T, NH);           // For k (shape: B, T, NH, C/NH)

    int block_size_apply_rope = HS / 2; // Half of head_dim

    apply_rope_forward_q1<<<blocks_q, block_size_apply_rope>>>(q, freqs_cos, freqs_sin, B, T, num_kv_heads, HS);
    apply_rope_forward_k1<<<blocks_k, block_size_apply_rope>>>(k, freqs_cos, freqs_sin, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // Repeat interleave for GQA

    float *k_rep = make_random_float(B * NH * T * HS * sizeof(float));
    float *v_rep = make_random_float(B * NH * T * HS * sizeof(float));

    dim3 blocks_repeat_kv(B, T, num_kv_heads);
    int block_size_repeat_kv = HS;

    repeat_kv_forward_kernel2<<<blocks_repeat_kv, block_size_repeat_kv>>>(k_rep, v_rep, k, v, B, num_kv_heads, T, HS, num_queries_per_kv);
    cudaCheck(cudaGetLastError());

    // Batched matrix multiply with cuBLAS for QK^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k_rep, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    // Multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = ceil_div(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // New approach: first cuBLAS another batched matmul
    float *vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v_rep, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // Now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = ceil_div(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaFree(&k_rep));
    cudaCheck(cudaFree(&v_rep));
}

// ----------------------------------------------------------------------------

int main(int argc, const char **argv)
{
    setup_main();

    int B = 8;
    int T = 1024;
    int C = 768;
    int NH = 8;
    int num_kv_heads = 2;
    int HS = C / NH;
    int num_queries_per_kv = NH / num_kv_heads;

    // Allocating host memory for input
    float *inp = make_random_float(B * T * (NH + 2 * num_kv_heads) * HS);                // QKV buffer [B, T, (NH + 2*num_kv_heads)*HS]
    float *qkvr = (float *)malloc(B * T * (NH + 2 * num_kv_heads) * HS * sizeof(float)); // qkvr to store QKV for backward-pass
    float *att = (float *)malloc(B * NH * T * T * sizeof(float));                        // Attention matrix (B, NH, T, T)
    float *preatt = (float *)malloc(B * NH * T * T * sizeof(float));                     // QK^T Scores (B, NH, T, T)
    float *out = (float *)malloc(B * T * C * sizeof(float));                             // Output buffer [B, T, C]
    float *freqs_cos = make_random_float(T * HS / 2);                                    // RoPE cosine values [T, HS/2]
    float *freqs_sin = make_random_float(T * HS / 2);                                    // RoPE sine values [T, HS/2]

    // read kernel_num from command line (this would switch between different kernels if necessary)
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // Move input data to GPU
    float *d_inp, *d_qkvr, *d_att, *d_preatt, *d_out, *d_freqs_cos, *d_freqs_sin;
    cudaCheck(cudaMalloc(&d_inp, B * T * (NH + 2 * num_kv_heads) * HS * sizeof(float)));
    cudaCheck(cudaMalloc(&d_qkvr, B * T * (NH + 2 * num_kv_heads) * HS * sizeof(float)));
    cudaCheck(cudaMalloc(&d_att, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_preatt, B * NH * T * T * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_cos, T * HS / 2 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_freqs_sin, T * HS / 2 * sizeof(float)));

    cudaCheck(cudaMemcpy(d_inp, inp, B * T * (NH + 2 * num_kv_heads) * HS * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freqs_cos, freqs_cos, T * HS / 2 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_freqs_sin, freqs_sin, T * HS / 2 * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize cuBLAS handle
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));
    /**
     * TODO:
     * - Adding a CPU Implementation
     */

    attention_forward_gqa(d_out, d_qkvr, d_preatt, d_att, d_inp, d_freqs_cos, d_freqs_sin, B, T, C, NH, num_kv_heads, HS, num_queries_per_kv);
    cudaCheck(cudaGetLastError());

    /**
     * TODO: Implement CPU Implementation of GQA-Attn
     * and Validate our GPU implementation
     */
    // Validate the results
    // float tol = 1e-5;
    // validate_result(d_out, out, "output", B * T * C, tol);
    // printf("All results match. Starting benchmarks.\n\n");

    // Benchmark the kernel performance
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, attention_forward_gqa,
                                          d_out, d_qkvr, d_preatt, d_att, d_inp, d_freqs_cos, d_freqs_sin, B, T, C, NH, num_kv_heads, HS, num_queries_per_kv);

    // Memory bandwidth calculation: 2 reads (QKV and Attention), 1 write (output), each of 4 bytes
    long memory_ops = B * T * C * sizeof(float) * 3;
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;

    printf("time %.4f ms | bandwidth %.2f GB/s\n", elapsed_time, memory_bandwidth);

    // Clean up resources
    cudaCheck(cudaFree(d_qkvr));
    cudaCheck(cudaFree(d_att));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_freqs_cos));
    cudaCheck(cudaFree(d_freqs_sin));
    cudaCheck(cudaFree(d_inp));

    free(qkvr);
    free(att);
    free(out);
    free(inp);
    free(freqs_cos);
    free(freqs_sin);
    free(out_cpu);

    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}