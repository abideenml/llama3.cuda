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
// CUDA & cuDNN setup
static bool first_run_validation = true; // always run e.g. permute on 1st run

#ifdef ENABLE_CUDNN
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;
#if CUBLAS_LOWP == CUDA_R_16BF
#define CUDNN_16BIT fe::DataType_t::BFLOAT16
#else
#define CUDNN_16BIT fe::DataType_t::HALF
#endif

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void *cudnn_workspace = NULL;

#define checkCudaErr(err) assert((int)err == 0);
#define checkCudnnErr(err) assert((int)err == 0);
#endif // ENABLE_CUDNN
// ----------------------------------------------------------------------------

void attention_backward_gqa(float *dinp, float *dqkvr, float *dpreatt, float *datt,
                            float *scratch,
                            const float *dout,
                            const float *freq_cos, const float *freq_sin,
                            const float *qkvr, const float *att,
                            int B, int T, int C, int NH, int num_kv_heads)
{
    const int block_size = 256;
    int HS = C / NH; // head size
    int queries_per_kv = NH / num_kv_heads;
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;

    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));

    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));

    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());

    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));

    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));

    // Repeat interleave for GQA if num_kv_heads != NH
    if (num_kv_heads != NH)
    {
        // Allocate intermediate tensors for backward repeat interleave
        float *dsrc_k, *dsrc_v;
        cudaMalloc((void **)&dsrc_k, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));
        cudaMalloc((void **)&dsrc_v, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));
        cudaMemset(dsrc_k, 0, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));
        cudaMemset(dsrc_v, 0, B * num_kv_heads * queries_per_kv * T * HS * sizeof(float));

        // backward through repeat interleave operation for dk and dv
        int repeat_interleave_threads = B * NH * T * HS;
        num_blocks = CEIL_DIV(repeat_interleave_threads, block_size);
        repeat_interleave_backward_kernel<<<num_blocks, block_size>>>(dsrc_k, dk, B, num_kv_heads, T, HS, queries_per_kv);
        repeat_interleave_backward_kernel<<<num_blocks, block_size>>>(dsrc_v, dv, B, num_kv_heads, T, HS, queries_per_kv);
        cudaCheck(cudaGetLastError());

        // Apply RoPE backward
        apply_rope_backward(dq, dsrc_k, q, k, freq_cos, freq_sin, B, T, NH, (C / NH)); // (C /NH) = (C /NH) is the head_dim (hs)

        // backward into inp
        num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
        permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dsrc_k, dsrc_v, B, T, NH, HS);
        cudaCheck(cudaGetLastError());

        // Cleanup
        cudaFree(dsrc_k);
        cudaFree(dsrc_v);
    }
    else
    {
        // backward into inp
        // backward into inp without repeat interleave
        num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
        permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
        cudaCheck(cudaGetLastError());
    }
}