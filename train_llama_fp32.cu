/*
GPT-2 Transformer Neural Net trained in raw CUDA
Non-trivial notes to be aware of:

We are being clever in the backward pass to conserve memory.
In particular, all parameters use a += in the backward pass, so we
can later do gradient accumulation. But all activations have = instead of +=
because these are faster (just read, no write). This is okay for all activations
except for those in the residual stream, where the gradients have to add. We make
sure that those parts work out ok and that we do a += as necessary. E.g.,
the layernorms are connected to the residuals so we += in layernorm backward.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// CUDA utils

// // convenience macro for calculating grid/block dimensions for kernels
// #define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// CUDA error checking
void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};
#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// cuBLAS error checking
void cublasCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status)                        \
    {                                              \
        cublasCheck((status), __FILE__, __LINE__); \
    }

static cublasComputeType_t cublas_compute_type;
cublasHandle_t cublas_handle;

namespace cg = cooperative_groups;

// ----------------------------------------------------------------------------
// all the kernels

__device__ inline float4 add_float4(const float4 &a, const float4 &b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// use of float4 leads to using 128-bit LDG / STG instructions in SASS,
// very helpful in memory-bound kernels like encoder_forward
__global__ void encoder_forward_kernel3(float4 *out,
                                        const int *inp, const float4 *wte,
                                        int B, int T, int C)
{
    int C4 = C / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C4;
    if (idx < N)
    {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];
        out[b * T * C4 + t * C4 + c4] = wte[ix * C4 + c4]; // Removed wpe
    }
}

// uses float4 wte and dout
__global__ void encoder_backward_kernel(float4 *dwte, const float4 *dout, const int *inp, int B, int T, int C)
{
    int C4 = C / 4;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C4;
    if (idx < N)
    {
        int bt = idx / C4;
        int b = bt / T;
        int t = bt % T;
        int c4 = idx % C4;
        int ix = inp[b * T + t];

        // Using atomicAdd to avoid race conditions while updating gradients
        atomicAdd(&(dwte[ix * C4 + c4].x), dout[b * T * C4 + t * C4 + c4].x);
        atomicAdd(&(dwte[ix * C4 + c4].y), dout[b * T * C4 + t * C4 + c4].y);
        atomicAdd(&(dwte[ix * C4 + c4].z), dout[b * T * C4 + t * C4 + c4].z);
        atomicAdd(&(dwte[ix * C4 + c4].w), dout[b * T * C4 + t * C4 + c4].w);
    }
}

/**
 *
 * A helper kernel using for RoPE
 *
 **/
__global__ void precompute_freqs_cis_kernel(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < dim / 2)
    {
        float freq = 1.0f / powf(theta, (float)tid * 2.0f / dim); // float powf(float base, float exponent);

        for (int t = 0; t < end; t++)
        {
            freqs_cos[t * (dim / 2) + tid] = cosf(t * freq);
            freqs_sin[t * (dim / 2) + tid] = sinf(t * freq);
        }
    }
}

__global__ void apply_rope_forward_kernel(
    float *q, float *k, float *freqs_cos, float *freqs_sin,
    int B, int T, int NH, int HS)
{

    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    int half_hs = HS / 2;

    if (hs < half_hs)
    {
        int index = b * T * NH * HS + t * NH * HS + nh * HS + hs;
        int freq_index = t * half_hs + hs;

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        float q_r = q[index];
        float q_i = q[index + half_hs];
        float k_r = k[index];
        float k_i = k[index + half_hs];

        q[index] = q_r * cos_val - q_i * sin_val;           // (ac-bd)
        q[index + half_hs] = q_r * sin_val + q_i * cos_val; // (ad+bc) * i

        k[index] = k_r * cos_val - k_i * sin_val;           // (ac-bd)
        k[index + half_hs] = k_r * sin_val + k_i * cos_val; // (ad+bc) * i
    }
}

__global__ void apply_rope_backward_kernel(
    float *dq, float *dk, const float *q, const float *k,
    const float *freqs_cos, const float *freqs_sin,
    int B, int T, int NH, int HS)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int nh = blockIdx.z;
    int hs = threadIdx.x;

    int half_hs = HS / 2;

    if (hs < half_hs)
    {
        int index = b * T * NH * HS + t * NH * HS + nh * HS + hs;
        int freq_index = t * half_hs + hs;

        float cos_val = freqs_cos[freq_index];
        float sin_val = freqs_sin[freq_index];

        float q_r = q[index];
        float q_i = q[index + half_hs];
        float k_r = k[index];
        float k_i = k[index + half_hs];

        // Gradients with respect to q and k (already computed)
        float dq_r = dq[index];
        float dq_i = dq[index + half_hs];
        float dk_r = dk[index];
        float dk_i = dk[index + half_hs];

        // Gradients with respect to q and k
        dq[index] = dq_r * cos_val + dq_i * sin_val;
        dq[index + half_hs] = dq_i * cos_val - dq_r * sin_val;
        dk[index] = dk_r * cos_val + dk_i * sin_val;
        dk[index + half_hs] = dk_i * cos_val - dk_r * sin_val;

        /**
         * WE DON'T NEED TO ACCUMULATE THE GRADIENTs IN d_freq_cos and d_freq_sin.
         */
        // // Gradients with respect to freqs_cos and freqs_sin
        // float d_freq_cos_q = q_r * dq_r + q_i * dq_i;
        // float d_freq_sin_q = -q_i * dq_r + q_r * dq_i;
        // float d_freq_cos_k = k_r * dk_r + k_i * dk_i;
        // float d_freq_sin_k = -k_i * dk_r + k_r * dk_i;

        // atomicAdd(&d_freq_cos[freq_index], d_freq_cos_q + d_freq_cos_k);
        // atomicAdd(&d_freq_sin[freq_index], d_freq_sin_q + d_freq_sin_k);
    }
}

/**
 * RMS-Norm Kernel:
 *
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

__global__ void rmsnorm_backward_kernel1(float *dinp, float *dweight, float *dbias,
                                         const float *dout, const float *inp, const float *weight,
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
        float norm = inp_bt[i] / rms;
        dinp_bt[i] = dout_bt[i] * weight[i] / rms + drms * inp_bt[i];
    }
}
__global__ void layernorm_forward_kernel3(float *__restrict__ out, float *__restrict__ mean, float *__restrict__ rstd,
                                          const float *__restrict__ inp, const float *__restrict__ weight,
                                          const float *__restrict__ bias, int N, int C)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
    {
        return;
    }

    // the row of input that this group of threads is responsible for
    const float *x = inp + idx * C;

    // mean
    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum += x[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float m = sum / C;
    if (warp.thread_rank() == 0 && mean != nullptr)
    {
        __stcs(mean + idx, m);
    }

    // rstd
    sum = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    sum = cg::reduce(warp, sum, cg::plus<float>{});
    float s = rsqrtf(sum / C + 1e-5f);
    if (warp.thread_rank() == 0 && rstd != nullptr)
    {
        __stcs(rstd + idx, s);
    }

    // final normalization and scaling by weight/bias
    float *o = out + idx * C;
    for (int c = warp.thread_rank(); c < C; c += warp.size())
    {
        // load and store using the .cs "streaming" hint to the compiler,
        // indicating that this data will not be reused soon, and can be streamed through the caches
        // this allows the threads to get more cache-hits for the (shared) weight and bias parameters
        float n = s * (__ldcs(x + c) - m);
        __stcs(o + c, n * weight[c] + bias[c]);
    }
}

__global__ void permute_kernel(float *q, float *k, float *v,
                               const float *inp,
                               int B, int N, int NH, int d)
{
    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Q[b][nh_][n][d_] = inp[b][n][0][nh_][d_]
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        q[idx] = __ldcs(&inp[inp_idx]);
        k[idx] = __ldcs(&inp[inp_idx + NH * d]);
        v[idx] = __ldcs(&inp[inp_idx + 2 * (NH * d)]);
    }
}

__global__ void permute_kernel_backward(float *dinp,
                                        const float *dq, const float *dk, const float *dv,
                                        int B, int N, int NH, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;

        int inp_idx = (b * N * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_ * d) + d_;
        dinp[inp_idx] = dq[idx];
        dinp[inp_idx + NH * d] = dk[idx];
        dinp[inp_idx + 2 * (NH * d)] = dv[idx];
    }
}

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

__global__ void unpermute_kernel_backward(float *dinp, const float *dout, int B, int N, int NH, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * NH * N * d)
    {
        int b = idx / (NH * N * d);
        int rest = idx % (NH * N * d);
        int nh_ = rest / (N * d);
        rest = rest % (N * d);
        int n = rest / d;
        int d_ = rest % d;
        int other_idx = (b * NH * N * d) + (n * NH * d) + (nh_ * d) + d_;
        dinp[idx] = dout[other_idx];
    }
}

__device__ float &vec_at(float4 &vec, int index)
{
    return reinterpret_cast<float *>(&vec)[index];
}

__device__ float vec_at(const float4 &vec, int index)
{
    return reinterpret_cast<const float *>(&vec)[index];
}

__global__ void softmax_forward_kernel5(float *out, float inv_temperature, const float *inp, int N, int T)
{
    // inp, out shape: (N, T, T), where N = B * NH
    // fuses the multiplication by scale inside attention
    // directly autoregressive, so we only compute the lower triangular part
    // uses the online softmax algorithm
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

__global__ void residual_forward_kernel(float *out, float *inp1, float *inp2, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward_kernel(float *out, const float *inp, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube)));
    }
}

__global__ void gelu_backward_kernel(float *dinp, const float *inp, const float *dout, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] = local_grad * dout[i];
    }
}

// this kernel performs a column-wise reduction over dout, in PyTorch equivalent to:
// dbias = dout.sum((0,1))
// the idea is to employ one block to reduce along several columns,
// where each block has a width of 32 columns to ensure coalesced access.
// at the end we accumulate the reductions performed by the warps in each block via shared memory
__global__ void matmul_backward_bias_kernel4(float *dbias, const float *dout, int B, int T, int OC)
{
    // this kernel is launched with 1D grid_dim of OC/32
    // for example let's say block_size is 128
    extern __shared__ float smem[];             // of size block_size (128)
    const int warp_id = threadIdx.x / warpSize; // warp index in the block, 0,1,2,3
    const int lane_id = threadIdx.x % warpSize; // thread index in the warp, 0,1,2,...,31
    const int tl = blockIdx.x * warpSize;       // pointer to the start column for this block
    const int vstep = blockDim.x / warpSize;    // number of warps in a block, e.g. 4

    // pointer to the start of the column for one lane of threads
    // so e.g. 4 threads (of the same lane_id) will reduce this one column
    const float *dout_col = dout + tl + lane_id;

    // column reductions by looping through the rows
    // each of the 4 threads offsets by its warp_id and then skips by vstep
    // together these 4 threads cover all B*T rows of this (lane_id) column
    // importantly, consecutive threads (in threadId) are processing adjacent columns,
    // leading to a coalesced memory access pattern
    float dout_sum = 0.0f;
    for (int row = warp_id; row < B * T; row += vstep)
    {
        dout_sum += dout_col[row * OC];
    }
    smem[lane_id + warp_id * warpSize] = dout_sum;
    __syncthreads();

    // warp_id 0 reduces the shared memory column-wise, linearly
    dout_sum = 0.0f;
    if (warp_id == 0)
    {
        for (int j = 0; j < vstep; j++)
        {
            dout_sum += smem[lane_id + j * warpSize];
        }
        dbias[tl + lane_id] += dout_sum;
    }
}

// uses shared memory instead for the reduces
__global__ void layernorm_backward_kernel2(float *dinp, float *dweight, float *dbias,
                                           const float *dout, const float *inp, const float *weight, const float *mean, const float *rstd,
                                           int B, int T, int C)
{
    extern __shared__ float shared[]; // size = 2 * C

    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    int N = B * T;
    if (idx >= N)
    {
        return;
    } // thread guards

    int b = idx / T;
    int t = idx % T;

    const float *dout_bt = dout + b * T * C + t * C;
    const float *inp_bt = inp + b * T * C + t * C;
    float *dinp_bt = dinp + b * T * C + t * C;
    const float mean_bt = mean[b * T + t];
    const float rstd_bt = rstd[b * T + t];

    // the first half of shared memory is bias, second is weight
    float *dbias_shared = shared;
    float *dweight_shared = shared + C;

// init shared memory to zero
#pragma unroll
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        dbias_shared[i] = 0.0f;
        dweight_shared[i] = 0.0f;
    }
    __syncthreads();

    // first: two reduce operations
    float dnorm_mean = 0.0f;
    float dnorm_norm_mean = 0.0f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        dnorm_mean += dnorm_i;
        dnorm_norm_mean += dnorm_i * norm_bti;
    }
    dnorm_mean = cg::reduce(warp, dnorm_mean, cg::plus<float>{});
    dnorm_norm_mean = cg::reduce(warp, dnorm_norm_mean, cg::plus<float>{});
    dnorm_mean = dnorm_mean / C;
    dnorm_norm_mean = dnorm_norm_mean / C;

    // now iterate again and accumulate all the gradients
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
        float dnorm_i = weight[i] * dout_bt[i];
        // gradient contribution to bias
        atomicAdd(&dbias_shared[i], dout_bt[i]);
        // gradient contribution to weight
        atomicAdd(&dweight_shared[i], norm_bti * dout_bt[i]);
        // gradient contribution to input
        float dval = 0.0f;
        dval += dnorm_i;                    // term 1
        dval -= dnorm_mean;                 // term 2
        dval -= norm_bti * dnorm_norm_mean; // term 3
        dval *= rstd_bt;                    // final scale
        dinp_bt[i] += dval;
    }
    __syncthreads();

    // write to global memory
    for (int i = threadIdx.x; i < C; i += blockDim.x)
    {
        atomicAdd(&dbias[i], dbias_shared[i]);
        atomicAdd(&dweight[i], dweight_shared[i]);
    }
}

__global__ void softmax_autoregressive_backward_kernel(float *dpreatt, const float *datt, const float *att,
                                                       int B, int T, int C, float scale)
{
    constexpr const int BlockSize = 256;
    constexpr int T_per_block = 4;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float block_acc[32];

    int idx = blockIdx.y;
    // go through blocks in reverse order, so the slowest block starts first
    int t0 = T - 1 - T_per_block * blockIdx.x;

    att += idx * T * T;
    datt += idx * T * T;
    dpreatt += idx * T * T;

    if (warp.meta_group_rank() == 0)
    {
        block_acc[warp.thread_rank()] = 0;
    }

    for (int to = 0; to < T_per_block; ++to)
    {
        int t = t0 - to;
        if (t < 0)
            return;
        const float *att_bth = att + t * T;
        const float *datt_bth = datt + t * T;
        float *dpreatt_bth = dpreatt + t * T;

        float local_sum = 0;
        for (int t2 = block.thread_rank(); t2 <= t; t2 += BlockSize)
        {
            local_sum += att_bth[t2] * datt_bth[t2];
        }

        block_acc[warp.meta_group_rank()] = cg::reduce(warp, local_sum, cg::plus<float>{});
        block.sync();
        local_sum = cg::reduce(warp, block_acc[warp.thread_rank()], cg::plus<float>{});

        for (int t3 = block.thread_rank(); t3 <= t; t3 += BlockSize)
        {
            // don't touch the cache. Some parts will still be here from the previous loop, and
            // we want to exploit those.
            float acc = __ldcs(att_bth + t3) * (__ldcs(datt_bth + t3) - local_sum);
            __stcs(dpreatt_bth + t3, scale * acc);
        }
    }
}

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ inline float lerp(float start, float end, float weight)
{
    return fma(weight, end, fma(-weight, start, start));
}

__global__ void adamw_kernel2(float *params_memory, float *grads_memory, float *m_memory, float *v_memory, long num_parameters,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_parameters)
        return; // guard
    float grad = grads_memory[i];
    float m = m_memory[i];
    float v = v_memory[i];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[i] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[i] = v;
    m /= beta1_correction; // m_hat
    v /= beta2_correction; // v_hat
    params_memory[i] -= learning_rate * (m / (sqrtf(v) + eps) + weight_decay * params_memory[i]);
}

struct SoftmaxParams
{
    float Scale;
    float Offset;
};

__device__ SoftmaxParams prepare_softmax_blockwide_nofloat4(cg::thread_block_tile<32> &warp,
                                                            int idx, const float *inp, int V, int P)
{
    // same but not float4
    // one row of inp, i.e. inp[idx, :] of shape (V,)

    const float *x = inp + idx * P;
    float thread_maxval = -INFINITY;
    float thread_sumval = 0.0f;
    // do the loop in reverse to maximise probability of L2 cache hits
    // so even small L2s get some hits on the 2nd read of the same thread
    for (int i = V + threadIdx.x - blockDim.x; i >= 0; i -= blockDim.x)
    {
        float v = x[i];
        float old_maxval = thread_maxval;
        thread_maxval = fmaxf(thread_maxval, v);
        thread_sumval *= expf((old_maxval - thread_maxval));
        thread_sumval += expf(v - thread_maxval);
    }

    // two reductions of up to 1024 threads:
    // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp (shuffle)
    // this results in much cleaner assembly than a multi-warp cg::reduce
    __shared__ float shared_maxval[32];
    __shared__ float shared_sumval[32];
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // reduce maxval within each warp
    float warp_maxval = cg::reduce(warp, thread_maxval, cg::greater<float>{});
    // thread 0 in each warp writes to shared memory
    if (lane_id == 0)
    {
        shared_maxval[warp_id] = warp_maxval;
    }
    __syncthreads();
    // each thread now loads the maxval across previous warps
    // if the thread is "out of range" of data, use -FLT_MAX as the maxval
    warp_maxval = (lane_id < num_warps) ? shared_maxval[lane_id] : -FLT_MAX;
    // now reduce the maxval among the warp threads
    float block_maxval = cg::reduce(warp, warp_maxval, cg::greater<float>{});
    // each thread uses maxval to scale sumval to avoid numerical instability / overflow
    thread_sumval *= expf(thread_maxval - block_maxval);
    // (warp-level) reduce sumval, thread 0 in each warp saves result in shared memory
    float warp_sumval = cg::reduce(warp, thread_sumval, cg::plus<float>{});
    if (lane_id == 0)
    {
        shared_sumval[warp_id] = warp_sumval;
    }
    __syncthreads();
    // same strategy, now reduce sumval across warps
    warp_sumval = (lane_id < num_warps) ? shared_sumval[lane_id] : 0.0f;
    float block_sumval = cg::reduce(warp, warp_sumval, cg::plus<float>{});
    // return the softmax parameters
    return SoftmaxParams{1.f / block_sumval, block_maxval};
}

// same as 2 but not using float4 (see dev/cuda/classifier_fused.cu)
// will _update_ logits to logit gradients
__global__ void fused_classifier_kernel3(float *logits, float *losses, float *probs,
                                         const float *dlosses, const int *targets,
                                         int B, int T, int V, int P)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x;
    int ix = targets[idx];

    // softmax (reading B * T * V, same logits read again below, hopefully still in cache)
    SoftmaxParams sp = prepare_softmax_blockwide_nofloat4(warp, idx, logits, V, P);

    // calculate the probability needed for the loss and update (single-threaded)
    if (threadIdx.x == 0)
    {
        float prob = expf(logits[idx * P + ix] - sp.Offset) * sp.Scale;
        losses[idx] = -logf(prob);
    }

    // very sensible default for dlosses is 1/(B*T), which is the uniform loss
    float dloss = dlosses != NULL ? dlosses[idx] : 1.0f / (B * T);
    // calculate the gradients directly, saves bandwidth from probs during training
    // but also supports writing probs for inference-only and debugging
    const float *logits_vec = logits + idx * P;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
    {
        // this is the 2nd read of logits after the one in prepare_softmax2
        // this data will never be needed again, so we reduce cache persistence
        float v = __ldcs(&logits_vec[i]);
        float prob = expf(v - sp.Offset) * sp.Scale;
        if (probs != NULL)
        {
            probs[idx * P + i] = prob;
        }
        float indicator = (i == ix) ? 1.0f : 0.0f;
        logits[idx * P + i] = (prob - indicator) * dloss;
    }
}

__device__ float4 ld_vec(const float *address)
{
    return *reinterpret_cast<const float4 *>(address);
}

__device__ void st_vec(float *address, float4 val)
{
    *reinterpret_cast<float4 *>(address) = val;
}

__global__ void __launch_bounds__(16 * 16, 2) matmul_forward_kernel4(float *out,
                                                                     const float *inp, const float *weight, const float *bias,
                                                                     int C, int OC)
{
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // each thread handles 8x8 elements; each block 128 by 128 elements.
    int oc = 8 * (blockIdx.y * blockDim.y + threadIdx.y);

    // buffers to cache chunks of the input matrices
    __shared__ float lhs_s[128][32];
    __shared__ float rhs_s[128][32];

    // adjust our pointers for the current block
    inp += 128 * blockIdx.x * C;
    weight += 128 * blockIdx.y * C;
    out += 128 * blockIdx.x * OC + 128 * blockIdx.y;

    float vals[8][8] = {};
    if (bias != NULL)
    {
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j += 4)
            {
                float4 b = ld_vec(bias + oc + j);
                vals[i][j + 0] = b.x;
                vals[i][j + 1] = b.y;
                vals[i][j + 2] = b.z;
                vals[i][j + 3] = b.w;
            }
        }
    }

    int si_start = 4 * (16 * threadIdx.y + threadIdx.x);
    for (int so = 0; so < C; so += 32)
    {
        __syncthreads();
        int xmod8 = threadIdx.x % 8;
        int xby8 = threadIdx.x / 8;
        int xo = 4 * xmod8;
        for (int y = 2 * threadIdx.y + xby8; y < 128; y += 32)
        {
            st_vec(&lhs_s[y][xo], ld_vec(inp + y * C + so + xo));
            st_vec(&rhs_s[y][xo], ld_vec(weight + y * C + so + xo));
        }
        __syncthreads();

        for (int si = si_start; si < si_start + 32; si += 4)
        {
            float4 rhs[8];
            for (int u = 0; u < 8; ++u)
            {
                rhs[u] = ld_vec(&rhs_s[u + 8 * threadIdx.y][si % 32]);
            }

            for (int ii = 0; ii < 8; ++ii)
            {
                float4 lhs = ld_vec(&lhs_s[ii + 8 * threadIdx.x][si % 32]);
                for (int ji = 0; ji < 8; ++ji)
                {
                    vals[ii][ji] += lhs.x * rhs[ji].x;
                    vals[ii][ji] += lhs.y * rhs[ji].y;
                    vals[ii][ji] += lhs.z * rhs[ji].z;
                    vals[ii][ji] += lhs.w * rhs[ji].w;
                }
            }
        }
    }

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; j += 4)
        {
            float4 result;
            result.x = vals[i][j + 0];
            result.y = vals[i][j + 1];
            result.z = vals[i][j + 2];
            result.w = vals[i][j + 3];
            st_vec(out + (8 * threadIdx.x + i) * OC + 8 * threadIdx.y + j, result);
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launchers

void encoder_forward(float *out,
                     const int *inp, const float *wte,
                     int B, int T, int C)
{
    assert(C % 4 == 0);
    const int block_size = 512;
    const int N = B * T * C;
    const int grid_size = CEIL_DIV((N / 4), block_size);
    encoder_forward_kernel3<<<grid_size, block_size>>>((float4 *)out, inp, (float4 *)wte, B, T, C);
    cudaCheck(cudaGetLastError());
}

void encoder_backward(float *dwte,
                      const float *dout, const int *inp,
                      int B, int T, int C)
{
    assert(C % 4 == 0);
    const int N = B * T * C;
    const int block_size = 512;
    const int grid_size = CEIL_DIV((N / 4), block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>((float4 *)dwte, (float4 *)dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}

// A helper function to calculate `cis` components freqs_cos & freqs_sin

void precompute_freqs_cis(float *freqs_cos, float *freqs_sin, int dim, int end, float theta)
{
    int threads = 64;
    int blocks = (dim / 2 + threads - 1) / threads;
    precompute_freqs_cis_kernel<<<blocks, threads>>>(freqs_cos, freqs_sin, dim, end, theta);
    cudaDeviceSynchronize();
}

void apply_rope_forward(float *q, float *k, float *freqs_cos, float *freqs_sin, int B, int T, int NH, int HS)
{
    dim3 blocks(B, T, NH);
    int threads = HS / 2;
    apply_rope_forward_kernel<<<blocks, threads>>>(q, k, freqs_cos, freqs_sin, B, T, NH, HS);
    cudaDeviceSynchronize();
}

void apply_rope_backward(
    float *dq, float *dk, const float *q, const float *k,
    const float *freqs_cos, const float *freqs_sin,
    int B, int T, int NH, int HS)
{
    dim3 blocks(B, T, NH);
    dim3 threads(HS / 2);
    apply_rope_backward_kernel<<<blocks, threads>>>(
        dq, dk, q, k, freqs_cos, freqs_sin, B, T, NH, HS);
    cudaDeviceSynchronize();
}

void rmsnorm_forward(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C)
{
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N + block_size - 1, block_size); // equivalent to ceil(N / block_size)
    rmsnorm_forward_kernel1<<<grid_size, block_size>>>(out, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

void rmsnorm_backward(float *dinp, float *dweight, float *dbias,
                      const float *dout, const float *inp, const float *weight,
                      int B, int T, int C)
{
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N + block_size - 1, block_size); // equivalent to ceil(N / block_size)
    rmsnorm_backward_kernel1<<<grid_size, block_size>>>(dinp, dweight, dbias, dout, inp, weight, N, C);
    cudaCheck(cudaGetLastError());
}

void layernorm_forward(float *out, float *mean, float *rstd,
                       float *inp, float *weight, float *bias,
                       int B, int T, int C)
{
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// kernel 1 is the most naive matmul kernel
void matmul_forward(float *out,
                    const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC)
{
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(B * T, 8 * sqrt_block_size), CEIL_DIV(OC, 8 * sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}

void attention_forward(float *out, float *qkvr, float *att,
                       float *inp,
                       int B, int T, int C, int NH)
{
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    v = qkvr + 2 * B * T * C;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // new approach: first cuBLAS another batched matmul
    float *vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

__global__ void repeat_interleave_forward_kernel(float *dst, const float *src, int B, int num_kv_heads, int T, int HS, int queries_per_kv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * num_kv_heads * queries_per_kv * T * HS;

    if (idx < total_threads)
    {
        int b = idx / (num_kv_heads * queries_per_kv * T * HS);
        int rest = idx % (num_kv_heads * queries_per_kv * T * HS);
        int nh = rest / (T * HS);
        rest = rest % (T * HS);
        int t = rest / HS;
        int hs = rest % HS;

        // Map destination head index to source head index
        int src_nh = nh % num_kv_heads;
        int src_idx = (b * num_kv_heads * T * HS) + (src_nh * T * HS) + (t * HS) + hs;
        int dst_idx = idx;
        dst[dst_idx] = src[src_idx];
    }
}

__global__ void repeat_interleave_backward_kernel(float *dsrc, const float *ddst, int B, int num_kv_heads, int T, int HS, int queries_per_kv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * num_kv_heads * queries_per_kv * T * HS;

    if (idx < total_threads)
    {
        int b = idx / (num_kv_heads * queries_per_kv * T * HS);
        int rest = idx % (num_kv_heads * queries_per_kv * T * HS);
        int nh = rest / (T * HS);
        rest = rest % (T * HS);
        int t = rest / HS;
        int hs = rest % HS;

        int src_nh = nh % num_kv_heads;
        int src_idx = (b * num_kv_heads * T * HS) + (src_nh * T * HS) + (t * HS) + hs;
        atomicAdd(&dsrc[src_idx], ddst[idx]);
    }
}

void attention_forward_gqa(float *out, float *qkvr, float *att, float *inp,
                           float *freq_cos, float *freq_sin,
                           int B, int T, int C, int NH, int num_kv_heads)
{
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size
    int queries_per_kv = NH / num_kv_heads;

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    // size_t ksize = sizeof(k) / sizeof(k[0]);
    // size_t vsize = sizeof(v) / sizeof(v[0]);
    // printf("ksize-Vsize: %ld, %ld\n%d, %d, %d\n", ksize, vsize, HS, kv_HS, queries_per_kv);
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);

    // okay so now, this kernel wants Q,K,V to all be of shape (B, NH, N, d)
    // but instead, we have a single tensor QKV (inp) of shape (B, N, 3, NH, d)
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    apply_rope_forward(q, k, freq_cos, freq_sin, B, T, NH, (C / NH));

    // Repeat interleave for GQA
    if (num_kv_heads != NH)
    {
        float *new_k, *new_v;
        cudaMalloc((void **)&new_k, B * NH * T * HS * sizeof(float));
        cudaMalloc((void **)&new_v, B * NH * T * HS * sizeof(float));

        int repeat_interleave_threads = B * num_kv_heads * queries_per_kv * T * HS;
        repeat_interleave_forward_kernel<<<num_blocks, block_size>>>(new_k, k, B, num_kv_heads, T, HS, queries_per_kv);
        repeat_interleave_forward_kernel<<<num_blocks, block_size>>>(new_v, v, B, num_kv_heads, T, HS, queries_per_kv);
        cudaCheck(cudaGetLastError());

        // Copy the contents of new_k and new_v back to k and v
        cudaMemcpy(k, new_k, B * NH * T * HS * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(v, new_v, B * NH * T * HS * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(new_k);
        cudaFree(new_v);
    }

    // size_t k1size = sizeof(k) / sizeof(k[0]);
    // size_t v1size = sizeof(v) / sizeof(v[0]);
    // printf("%ld, %ld", k1size, v1size);

    // Batched matrix multiply with cuBLAS for QK^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));
    // size_t sizepatt = sizeof(preatt) / sizeof(preatt[0]);
    // printf("Preatt: %ld", sizepatt);

    // Multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);

    cudaCheck(cudaGetLastError());

    // New approach: first cuBLAS another batched matmul
    float *vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // Now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

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

void residual_forward(float *out, float *inp1, float *inp2, int N)
{
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

__global__ void swiglu_forward_kernel(float *out, const float *inp, const float *gate, int N)
{
    /**
     * SwiGLU(x) = Swish(x) * Gate(x)
     * SwiGLU(x) = SiLU(x*W) * (x*V)
     * SiLU is the Swish activation function.
     * inp = x*W
     * gate = x*V
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float xiW = inp[i];
        float xiV = gate[i];
        out[i] = (xiW / (1.0f + expf(-xiW))) * xiV;
    }
}

void swiglu_forward(float *out, const float *inp, const float *gate, int N)
{
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    swiglu_forward_kernel<<<grid_size, block_size>>>(out, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

__global__ void swiglu_backward_kernel(float *dinp, const float *inp, const float *gate, const float *dout, const float *W, const float *V, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float xW = (float)inp[i];
        float xV = (float)gate[i];
        float y = xW / (1.0f + expf(-xW)) * xV;                     // SwiGLU(x)
        float sig_xW = 1.0f / (1.0f + expf(-xW));                   // Sigmoid(xW)
        float silu_prime = sig_xW + xW * sig_xW * (1.0f - sig_xW);  // SiLU'(xW)
        float grad_xW = (silu_prime * xV * W[i]) * dout[i];         // Gradient w.r.t. xW
        float grad_xV = (xW / (1.0f + expf(-xW))) * V[i] * dout[i]; // Gradient w.r.t. xV
        dinp[i] = grad_xW + grad_xV;                                // Sum of gradients
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void swiglu_backward(float *dinp, const float *inp, const float *gate, const float *dout, const float *W, const float *V, int N)
{
    /**
     * y=SiLU(xW)*(xV)
     * z=xW
     * g=xV
     *
     * Using Chain-Rule
     * ∂y/∂x = (σ(z) + z*σ(z)*(1−σ(z)))*g*W + SiLU(z)*V
     */
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    swiglu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, gate, dout, W, V, N);
    cudaCheck(cudaGetLastError());
}

void gelu_forward(float *out, const float *inp, int N)
{
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

void gelu_backward(float *dinp, const float *inp, const float *dout, const int N)
{
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}

void matmul_backward(float *dinp, float *dweight, float *dbias,
                     float *dout, float *inp, float *weight,
                     int B, int T, int C, int OC)
{
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B * T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B * T, &one, inp, C, dout, OC, &one, dweight, C));
    // backward to bias, if given, does a +=
    if (dbias != NULL)
    {
        const int block_size = 1024;
        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
        matmul_backward_bias_kernel4<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}

void layernorm_backward(float *dinp, float *dweight, float *dbias,
                        const float *dout, const float *inp, const float *weight, const float *mean, const float *rstd,
                        int B, int T, int C)
{
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(32 * N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}

// the sequence of transformations in this compound op is:
// inp (B,T,3C) -> qkvr (B,T,3C) -> preatt (B,NH,T,T) -> att (B,NH,T,T) -> vaccum (B,T,C) -> out (B,T,C)
void attention_backward(float *dinp, float *dqkvr, float *dpreatt, float *datt, float *scratch,
                        const float *dout,
                        const float *qkvr, const float *att,
                        int B, int T, int C, int NH)
{
    const int block_size = 256;
    int HS = C / NH; // head size
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
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}

// replaces logits with logit gradients
void fused_classifier3(float *logits, float *losses,
                       const float *dlosses, const int *targets,
                       int B, int T, int V, int P)
{
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct
{
    int max_seq_len = 1024;        // max sequence length, e.g. 1024
    int vocab_size = 50257;        // vocab size, e.g. 50257
    int num_layers = 12;           // number of layers, e.g. 12
    int num_heads = 12;            // number of heads in attention, e.g. 12
    int channels = 768;            // number of channels, e.g. 768
    int padded_vocab_size = 50304; // padded to e.g. %128==0, 50304
    int num_kv_heads = 6;          // attention_gqa
    float rope_theta = 10000.0;
} LLamaConfig;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 17
typedef struct
{
    float *wte; // (V, C)
    // float *wpe;      // (maxT, C)  No need of Positional Information parameter here. Since we are using RoPE
    float *ln1w;     // (L, C)
    float *ln1b;     // (L, C)
    float *qkvw;     // (L, 3*C, C)
    float *qkvb;     // (L, 3*C)
    float *attprojw; // (L, C, C)
    float *attprojb; // (L, C)
    float *ln2w;     // (L, C)
    float *ln2b;     // (L, C)
    float *fcw;      // (L, 4*C, C)
    float *fcb;      // (L, 4*C)
    float *fcw_g;    // (L, 4*C, C) Added for gate Mechanism of SwiGLU Activation
    float *fcb_g;    // (L, 4*C)
    float *fcprojw;  // (L, C, 4*C)
    float *fcprojb;  // (L, C)
    float *lnfw;     // (C)
    float *lnfb;     // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t *param_sizes, LLamaConfig config)
{
    int Vp = config.padded_vocab_size;
    int C = config.channels;
    int maxT = config.max_seq_len;
    int L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    // param_sizes[1] = maxT * C;         // wpe   No need. Using RoPE
    param_sizes[1] = L * C;            // ln1w
    param_sizes[2] = L * C;            // ln1b
    param_sizes[3] = L * (3 * C) * C;  // qkvw
    param_sizes[4] = L * (3 * C);      // qkvb
    param_sizes[5] = L * C * C;        // attprojw
    param_sizes[6] = L * C;            // attprojb
    param_sizes[7] = L * C;            // ln2w
    param_sizes[8] = L * C;            // ln2b
    param_sizes[9] = L * (4 * C) * C;  // fcw
    param_sizes[10] = L * (4 * C);     // fcb
    param_sizes[11] = L * (4 * C) * C; // fcw_g
    param_sizes[12] = L * (4 * C);     // fcb_g
    param_sizes[13] = L * C * (4 * C); // fcprojw
    param_sizes[14] = L * C;           // fcprojb
    param_sizes[15] = C;               // lnfw
    param_sizes[16] = C;               // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes, int on_device)
{
    // on_device: 0 = CPU, 1 = GPU
    // calculate the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once on the device
    float *params_memory;
    if (on_device)
    {
        cudaCheck(cudaMalloc((void **)&params_memory, num_parameters * sizeof(float)));
    }
    else
    {
        params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
    }
    // assign all the tensors their place in the array
    // Added 2 new params for SwiGLU
    // Removed wpe because of RoPE
    float **ptrs[] = {
        &params->wte, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb, &params->fcw_g, &params->fcb_g,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb};
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 18
typedef struct
{
    float *encoded;  // (B, T, C)
    float *freq_cos; // (T, (C/NH)/2)      # T * head_dim/2
    float *freq_sin; // (T, (C/NH)/2)      # T * head_dim/2

    float *ln1; // (L, B, T, C)
    // float *ln1_mean;   // (L, B, T)
    // float *ln1_rstd;   // (L, B, T)
    float *atty;      // (L, B, T, C)
    float *att;       // (L, B, NH, T, T)
    float *attproj;   // (L, B, T, C)
    float *residual2; // (L, B, T, C)
    float *ln2;       // (L, B, T, C)
    // float *ln2_mean;   // (L, B, T)
    // float *ln2_rstd;   // (L, B, T)
    float *fch;        // (L, B, T, 4*C)
    float *fch_glu;    // (L, B, T, 4*C)
    float *fch_swiglu; // (L, B, T, 4*C)
    float *fcproj;     // (L, B, T, C)
    float *residual3;  // (L, B, T, C)
    float *lnf;        // (B, T, C)
    // float *lnf_mean;   // (B, T)
    // float *lnf_rstd;   // (B, T)

    float *losses; // (B, T)
    // adding these two compared to the CPU .c code, needed for attention kernel as buffers
    float *qkvr; // (L, B, T, 3*C)
    // in inference mode, this buffer will store the logits
    // in training mode, this buffer will contain the *gradients* of the logits.
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    float *output;
} ActivationTensors;

void fill_in_activation_sizes(size_t *act_sizes, int B, int T, LLamaConfig config)
{
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * C;          // encoded
    act_sizes[1] = T * (C / (2 * NH)); // freq_cos  (seq_len * head_dim)
    act_sizes[2] = T * (C / (2 * NH)); // freq_sin
    act_sizes[3] = L * B * T * C;      // ln1

    // act_sizes[2] = L * B * T;                            // ln1_mean
    // act_sizes[3] = L * B * T;                            // ln1_rstd
    act_sizes[4] = L * B * T * C;      // atty
    act_sizes[5] = L * B * NH * T * T; // att
    act_sizes[6] = L * B * T * C;      // attproj
    act_sizes[7] = L * B * T * C;      // residual2
    act_sizes[8] = L * B * T * C;      // ln2
    // act_sizes[9] = L * B * T;                            // ln2_mean
    // act_sizes[10] = L * B * T;                           // ln2_rstd
    act_sizes[9] = L * B * T * 4 * C;  // fch
    act_sizes[10] = L * B * T * 4 * C; // fch_glu
    act_sizes[11] = L * B * T * 4 * C; // fch_swiglu
    act_sizes[12] = L * B * T * C;     // fcproj
    act_sizes[13] = L * B * T * C;     // residual3
    act_sizes[14] = B * T * C;         // lnf
    // act_sizes[17] = B * T;                               // lnf_mean
    // act_sizes[18] = B * T;                               // lnf_rstd
    act_sizes[15] = B * T;                               // losses
    act_sizes[16] = L * B * T * 3 * C;                   // qkvr
    act_sizes[17] = B * T * max(3 * C, max(NH * T, Vp)); // output / scratch
}

// Backward pass is conceptually quite different from forward, because we can discard
// the activations of a layer as soon as we're done with it. This lets us aggressively
// reuse memory, so that we need far fewer tensors for backward state.
#define NUM_BACKWARD_TENSORS 3
typedef struct
{
    float *bt4c;      // (B, T, 4*C)
    float *preatt;    // (B, NH, T, T)
    float *residual3; // (B, T, C)
    /**
     * NO NEED of accumulating these gradients.
     *
     *  // float *d_freq_cos; // (T, (C/NH)/2) T * (C / (2 * NH))
        // float *d_freq_sin; // (T, (C/NH)/2)

     *

    act_sizes[3] = T * (C / (2 * NH)); // d_freq_cos grads accumulation (Needed because we have to traverse from attention to precompute kernel)
    act_sizes[4] = T * (C / (2 * NH)); // d_freq_sin
     *
    */

} GradActTensors;

void fill_in_grad_act_sizes(size_t *act_sizes, int B, int T, LLamaConfig config)
{
    size_t NH = config.num_heads;
    size_t C = config.channels;
    act_sizes[0] = B * T * 4 * C;  // bt4c
    act_sizes[1] = B * NH * T * T; // preatt
    act_sizes[2] = B * T * C;      // residual3
}

float *malloc_and_point(float **targets[], const size_t *act_sizes, int n)
{
    size_t num_activations = 0;
    for (size_t i = 0; i < n; i++)
    {
        num_activations += act_sizes[i];
    }
    float *acts_memory;
    cudaCheck(cudaMalloc((void **)&acts_memory, num_activations * sizeof(float)));
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < n; i++)
    {
        *(targets[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

float *malloc_and_point_activations(ActivationTensors *acts, const size_t *act_sizes)
{
    float **ptrs[] = {
        &acts->encoded, &acts->freq_cos, &acts->freq_sin, &acts->ln1, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2,
        &acts->fch, &acts->fch_glu, &acts->fch_swiglu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->losses, &acts->qkvr, &acts->output};
    return malloc_and_point(ptrs, act_sizes, NUM_ACTIVATION_TENSORS);
}

float *malloc_and_point_backward(GradActTensors *acts, const size_t *act_sizes)
{
    float **ptrs[] = {
        &acts->bt4c,
        &acts->preatt,
        &acts->residual3,
    };
    return malloc_and_point(ptrs, act_sizes, NUM_BACKWARD_TENSORS);
}

typedef struct
{
    LLamaConfig config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float *grads_memory;
    // buffers for the AdamW optimizer
    float *m_memory;
    float *v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;
    // gradients of the activations
    GradActTensors grads_acts;
    size_t num_grad_acts;
    float *grads_acts_memory;
    // other run state configuration
    int batch_size;    // the batch size (B) of current forward pass
    int seq_len;       // the sequence length (T) of current forward pass
    int *inputs;       // the input tokens for the current forward pass
    int *targets;      // the target tokens for the current forward pass
    float mean_loss;   // after a forward pass with targets, will be populated with the mean loss
    float *cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
} LLaMA3;

void gpt2_build_from_checkpoint(LLaMA3 *model, const char *checkpoint_path)
{

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326)
    {
        fprintf(stderr, "Bad magic model file\n");
        exit(EXIT_FAILURE);
    }
    if (model_header[1] != 3)
    {
        // was bumped from 1 -> 3 to incorporate the padded vocab size
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float *params_memory_cpu = (float *)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

// ---------------------------------------------------------------

// Xavier Initialization
void xavier_initialization(float *param_values, size_t size, size_t fan_in, size_t fan_out)
{
    float scale = sqrt(2.0 / (fan_in + fan_out));
    for (size_t i = 0; i < size; i++)
    {
        param_values[i] = scale * ((float)rand() / RAND_MAX - 0.5);
    }
}

void load_model_params(LLaMA3 *model)
{
    // Initialize model configuration and parameters from given param_values
    model->config.max_seq_len = 1024;
    model->config.vocab_size = 50257;
    model->config.num_layers = 12;
    model->config.num_heads = 12;
    model->config.channels = 768;
    model->config.padded_vocab_size = 50304;
    model->config.rope_theta = 10000.0;
    model->config.num_kv_heads = 6;

    // allocate space for all the parameters and point them to the right places
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // // Debug: Print each parameter size
    // printf("Parameter sizes:\n");
    // for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    // {
    //     printf("param_sizes[%zu] = %zu\n", i, model->param_sizes[i]);
    // }

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // // Debug: Print the total number of parameters
    // printf("Total number of parameters: %zu\n", num_parameters);

    // Allocate CPU memory for parameter values
    float *params_memory_cpu = (float *)mallocCheck(num_parameters * sizeof(float));

    // Initialize parameter values on the CPU
    size_t offset = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        size_t size = model->param_sizes[i];
        size_t fan_in = (i == 0) ? model->config.padded_vocab_size : model->config.channels;
        size_t fan_out = (i == 0) ? model->config.channels : model->config.channels;
        xavier_initialization(params_memory_cpu + offset, size, fan_in, fan_out);
        offset += size;
    }

    // Allocate GPU memory for parameters and copy from CPU
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void llama3_forward(LLaMA3 *model, int *inputs, int *targets, int B, int T)
{
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL)
    {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    int num_kv_heads = model->config.num_kv_heads;
    float rope_theta = model->config.rope_theta;

    // validate inputs, all indices must be in the range [0, V)
    for (int i = 0; i < B * T; i++)
    {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL)
        {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if (model->acts_memory == NULL)
    {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
        // also create memory for caching inputs and targets
        cudaCheck(cudaMalloc((void **)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMalloc((void **)&model->targets, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void **)&model->cpu_losses, B * T * sizeof(float)));
    }
    else
    {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len)
        {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // copy inputs/targets to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));
    if (targets != NULL)
    {
        cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float *residual;
    // The freq_cos and freq_sin (cis-values) are same for every layer in the model. So, loading them for one time only
    // TODO: Extract them from activations (if needed)
    float *freq_cos = acts.freq_cos;
    float *freq_sin = acts.freq_sin;
    encoder_forward(acts.encoded, model->inputs, params.wte, B, T, C);       // encoding goes into residual[0]
    precompute_freqs_cis(freq_cos, freq_sin, ((C / NH) / 2), T, rope_theta); // rope_theta = 10000.0

    for (int l = 0; l < L; l++)
    {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // get the pointers of the weights for this layer
        float *l_ln1w = params.ln1w + l * C;
        float *l_ln1b = params.ln1b + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_qkvb = params.qkvb + l * 3 * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_attprojb = params.attprojb + l * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_ln2b = params.ln2b + l * C;
        float *l_fcw = params.fcw + l * 4 * C * C;
        float *l_fcb = params.fcb + l * 4 * C;
        float *l_fcw_g = params.fcw_g + l * 4 * C * C; // (L, 4*C, C) Added for gate Mechanism of SwiGLU Activation
        float *l_fcb_g = params.fcb_g + l * 4 * C;     // (L, 4*C)
        float *l_fcprojw = params.fcprojw + l * C * 4 * C;
        float *l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float *l_ln1 = acts.ln1 + l * B * T * C;
        // float *l_ln1_mean = acts.ln1_mean + l * B * T;
        // float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkvr = acts.qkvr + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_attproj = acts.attproj + l * B * T * C;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        // float *l_ln2_mean = acts.ln2_mean + l * B * T;
        // float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * 4 * C;
        float *l_fch_glu = acts.fch_glu + l * B * T * 4 * C;
        float *l_fch_swiglu = acts.fch_swiglu + l * B * T * 4 * C;
        float *l_fcproj = acts.fcproj + l * B * T * C;
        float *l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float *scratch = acts.output;

        // now do the forward pass
        rmsnorm_forward(l_ln1, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
        attention_forward_gqa(l_atty, l_qkvr, l_att, scratch, freq_cos, freq_cos, B, T, C, NH, num_kv_heads); // Added  acts.freq_cos, acts.freq_sin for q and k - RoPE
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B * T * C);
        rmsnorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);         // xW
        matmul_forward(l_fch_glu, l_ln2, l_fcw_g, l_fcb_g, B, T, C, 4 * C); // xV
        swiglu_forward(l_fch_swiglu, l_fch, l_fch_glu, B * T * 4 * C);
        matmul_forward(l_fcproj, l_fch_swiglu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
    }

    residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
    rmsnorm_forward(acts.lnf, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL)
    {
        // fused classifier: does the forward pass and first part of the backward pass
        // we're passing dlosses = NULL, which will default them to 1.0f/(B*T), i.e. uniform loss
        fused_classifier3(acts.output, acts.losses, NULL, model->targets, B, T, V, Vp);
        // for convenience also evaluate the mean loss (TODO re-think this compute+sync point)
        // move the (B,T) losses to CPU
        cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++)
        {
            mean_loss += model->cpu_losses[i];
        }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    }
    else
    {
        // if we don't have targets, we don't have loss
        model->mean_loss = -1.0f;
    }
}

void llama3_zero_grad(LLaMA3 *model)
{
    if (model->grads_acts_memory != NULL)
    {
        cudaCheck(cudaMemset(model->grads_acts_memory, 0, model->num_grad_acts * sizeof(float)));
    }
    if (model->grads_memory != NULL)
    {
        cudaCheck(cudaMemset(model->grads_memory, 0, model->num_parameters * sizeof(float)));
    }
}

void llama3_backward(LLaMA3 *model)
{

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f)
    {
        printf("Error: must forward with targets before backward\n");
        exit(EXIT_FAILURE);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL)
    {
        // allocate buffers for weight gradients
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes, 1);
        printf("allocated %zu MiB for parameter gradients\n", (model->num_parameters * sizeof(float)) >> 20);
        // we're going to be clever for the activations backward pass. we don't need to exactly
        // mirror the forward pass acrtivations and we will save memory.
        size_t bw_act_sizes[NUM_ACTIVATION_TENSORS];
        LLamaConfig cfg = model->config;
        cfg.num_layers = 1; // copy the configuration but override number of layers to 1
        fill_in_grad_act_sizes(bw_act_sizes, model->batch_size, model->seq_len, cfg);
        // count up and allocate the space
        model->grads_acts_memory = malloc_and_point_backward(&model->grads_acts, bw_act_sizes);
        model->num_grad_acts = 0;
        for (int i = 0; i < NUM_BACKWARD_TENSORS; i++)
        {
            model->num_grad_acts += bw_act_sizes[i];
        }
        printf("allocated %zu MiB for activation gradients\n", (model->num_grad_acts * sizeof(float)) >> 20);
        // init gradients of parameters and activations to zero
        llama3_zero_grad(model);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->seq_len;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    int num_kv_heads = model->config.num_kv_heads;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    GradActTensors grads_acts = model->grads_acts;

    // The freq_cos and freq_sin activations are same for every layer in the mode. So, loading them for one time only
    // Same goes for their grads accumulation (Needed because we have to traverse from attention to precompute kernel)
    float *freq_cos = acts.freq_cos;
    float *freq_sin = acts.freq_sin;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // this was done in the fused classifier kernel as last step of forward pass
    // technically that is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    // next: backward the classifier matmul
    matmul_backward(grads_acts.bt4c, grads.wte, NULL, acts.output, acts.lnf, params.wte, B, T, C, Vp);
    // backward the final layernorm
    float *residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
    float *dresidual = grads_acts.residual3;                // the main buffer holding the gradient in the backward pass
    rmsnorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.bt4c, residual, params.lnfw, B, T, C);

    // now backward all the layers
    for (int l = L - 1; l >= 0; l--)
    {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // get the pointers of the weights for this layer
        float *l_ln1w = params.ln1w + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_fcw = params.fcw + l * 4 * C * C;
        float *l_fcw_g = params.fcw_g + l * 4 * C * C;
        float *l_fcprojw = params.fcprojw + l * C * 4 * C;
        // get the pointers of the gradients of the weights for this layer
        float *dl_ln1w = grads.ln1w + l * C;
        float *dl_ln1b = grads.ln1b + l * C;
        float *dl_qkvw = grads.qkvw + l * 3 * C * C;
        float *dl_qkvb = grads.qkvb + l * 3 * C;
        float *dl_attprojw = grads.attprojw + l * C * C;
        float *dl_attprojb = grads.attprojb + l * C;
        float *dl_ln2w = grads.ln2w + l * C;
        float *dl_ln2b = grads.ln2b + l * C;
        float *dl_fcw = grads.fcw + l * 4 * C * C;
        float *dl_fcb = grads.fcb + l * 4 * C;
        float *dl_fcw_g = grads.fcw_g + l * 4 * C * C; // (L, 4*C, C) Added for gate Mechanism of SwiGLU Activation
        float *dl_fcb_g = grads.fcb_g + l * 4 * C;     // (L, 4*C)
        float *dl_fcprojw = grads.fcprojw + l * C * 4 * C;
        float *dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float *l_ln1 = acts.ln1 + l * B * T * C;
        // float *l_ln1_mean = acts.ln1_mean + l * B * T;
        // float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkvr = acts.qkvr + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        // float *l_ln2_mean = acts.ln2_mean + l * B * T;
        // float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * 4 * C;
        float *l_fch_glu = acts.fch_glu + l * B * T * 4 * C;
        float *l_fch_swiglu = acts.fch_swiglu + l * B * T * 4 * C;
        // get the pointers of the gradients of the activations for this layer
        // notice that there is no l *, because we just have a single copy, and keep
        // re-using this memory in every Transformer block as we calculate backward pass

        // we need a B x T x C buffer; thankfully, the forward activation for lnf isn't needed anymore,
        // so we can co-opt it here.
        float *dl_btc = acts.lnf;
        float *dl_bt4c = grads_acts.bt4c;
        float *dl_preatt = grads_acts.preatt;

        // re-use scratch buffer of the forward pass
        float *scratch = acts.output;

        // backprop this layer
        matmul_backward(dl_bt4c, dl_fcprojw, dl_fcprojb, dresidual, l_fch_swiglu, l_fcprojw, B, T, 4 * C, C);
        swiglu_backward(dl_bt4c, l_fch, l_fch_glu, dl_bt4c, l_fcw, l_fcw_g, B * T * 4 * C);
        matmul_backward(dl_btc, dl_fcw_g, dl_fcb_g, dl_bt4c, l_ln2, l_fcw_g, B, T, C, 4 * C);
        matmul_backward(dl_btc, dl_fcw, dl_fcb, dl_bt4c, l_ln2, l_fcw, B, T, C, 4 * C);
        // layernorm backward does += to the dresidual, so it correctly accumulates grad from the MLP block above
        rmsnorm_backward(dresidual, dl_ln2w, dl_ln2b, dl_btc, l_residual2, l_ln2w, B, T, C);
        matmul_backward(dl_btc, dl_attprojw, dl_attprojb, dresidual, l_atty, l_attprojw, B, T, C, C);
        // we more B x T x (4)C buffers. l_atty and l_fch aren't needed anymore at this point, so reuse their memory
        float *buffer_a = l_atty;
        float *buffer_b = l_fch; // this is B x T x 4C, so even larger than what we need

        attention_backward_gqa(dl_bt4c, buffer_b, dl_preatt, scratch, buffer_a, dl_btc, freq_cos, freq_sin, l_qkvr, l_att, B, T, C, NH, num_kv_heads);
        matmul_backward(dl_btc, dl_qkvw, dl_qkvb, dl_bt4c, l_ln1, l_qkvw, B, T, C, 3 * C);
        // layernorm backward does += to dresidual, so it correctly accumulates gradient for the Attention block above
        rmsnorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_btc, residual, l_ln1w, B, T, C);
    }

    encoder_backward(grads.wte, dresidual, model->inputs, B, T, C);
}

void llama3_update(LLaMA3 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t)
{
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL)
    {
        cudaCheck(cudaMalloc((void **)&model->m_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMalloc((void **)&model->v_memory, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->m_memory, 0, model->num_parameters * sizeof(float)));
        cudaCheck(cudaMemset(model->v_memory, 0, model->num_parameters * sizeof(float)));
        printf("allocated %zu MiB for AdamW optimizer state m\n", (model->num_parameters * sizeof(float)) >> 20);
        printf("allocated %zu MiB for AdamW optimizer state v\n", (model->num_parameters * sizeof(float)) >> 20);
    }

    int block_size = 512;
    int num_blocks = CEIL_DIV(model->num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel2<<<num_blocks, block_size>>>(model->params_memory, model->grads_memory, model->m_memory, model->v_memory,
                                              model->num_parameters,
                                              learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay);
    cudaCheck(cudaGetLastError());
}

void llama3_free(LLaMA3 *model)
{
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->grads_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    cudaFreeHost(model->cpu_losses);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.cu), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler: takes probabilities and samples integers from them

#define GPT2_EOT 50256

unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state)
{ // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float *logits, int n, float coin)
{
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++)
    {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += expf(logits[i]);
        if (coin < cdf)
        {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Logger lite, will probably grow/change some over time

typedef struct
{
    FILE *logfile;
    int flush_every; // every how many steps to flush the log
} Logger;

void logger_init(Logger *logger, const char *filename)
{
    logger->flush_every = 20;
    logger->logfile = NULL;
    if (filename != NULL)
    {
        logger->logfile = fopenCheck(filename, "w");
    }
}

void logger_log_val(Logger *logger, int step, float val_loss)
{
    if (logger->logfile != NULL)
    {
        fprintf(logger->logfile, "s:%d tel:%.4f\n", step, val_loss);
    }
}

void logger_log_train(Logger *logger, int step, float train_loss)
{
    if (logger->logfile != NULL)
    {
        fprintf(logger->logfile, "s:%d trl:%.4f\n", step, train_loss);
        if (step % 10 == 0)
        {
            fflush(logger->logfile);
        }
    }
}

void logger_free(Logger *logger)
{
    if (logger->logfile != NULL)
    {
        fclose(logger->logfile);
    }
}

// ----------------------------------------------------------------------------
// CLI, poor man's argparse

void error_usage()
{
    fprintf(stderr, "Usage:   ./train_gpt2fp32cu [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <string> train data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_train.bin)\n");
    fprintf(stderr, "  -j <string> val data filename pattern (default = dev/data/tinyshakespeare/tiny_shakespeare_val.bin)\n");
    fprintf(stderr, "  -o <string> output log file (default = NULL)\n");
    fprintf(stderr, "  -b <int>    batch size B (default = 4)\n");
    fprintf(stderr, "  -t <int>    sequence length T (default = 1024)\n");
    fprintf(stderr, "  -l <float>  learning rate (default = 3e-4f)\n");
    fprintf(stderr, "  -v <int>    val_loss_every, how often we evaluate val loss (default = 20)\n");
    fprintf(stderr, "  -m <int>    val_max_steps, up to how many val batches to estimate val loss? (default = 20)\n");
    fprintf(stderr, "  -s <int>    sample_every, how often we inference the model (default = 20)\n");
    fprintf(stderr, "  -g <int>    genT, how many steps of inference we do (default = 64)\n");
    exit(EXIT_FAILURE);
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[])
{

    // read in the (optional) command line arguments
    const char *train_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char *val_data_pattern = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char *output_log_file = NULL;
    int B = 4;    // batch size
    int T = 1024; // sequence length max
    float learning_rate = 3e-4f;
    int val_loss_every = 20; // every how many steps do we eval validation loss?
    int val_max_steps = 20;  // how many batches max do we eval for validation loss?
    int sample_every = 20;   // every how many steps to do inference?
    int genT = 64;           // number of steps of inference we will do
    for (int i = 1; i < argc; i += 2)
    {
        if (i + 1 >= argc)
        {
            error_usage();
        } // must have arg after flag
        if (argv[i][0] != '-')
        {
            error_usage();
        } // must start with dash
        if (strlen(argv[i]) != 2)
        {
            error_usage();
        } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 'i')
        {
            train_data_pattern = argv[i + 1];
        }
        else if (argv[i][1] == 'j')
        {
            val_data_pattern = argv[i + 1];
        }
        else if (argv[i][1] == 'o')
        {
            output_log_file = argv[i + 1];
        }
        else if (argv[i][1] == 'b')
        {
            B = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 't')
        {
            T = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'l')
        {
            learning_rate = atof(argv[i + 1]);
        }
        else if (argv[i][1] == 'v')
        {
            val_loss_every = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'm')
        {
            val_max_steps = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 's')
        {
            sample_every = atoi(argv[i + 1]);
        }
        else if (argv[i][1] == 'g')
        {
            genT = atoi(argv[i + 1]);
        }
        else
        {
            error_usage();
        }
    }
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| Parameter             | Value                                              |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| train data pattern    | %-50s |\n", train_data_pattern);
    printf("| val data pattern      | %-50s |\n", val_data_pattern);
    printf("| output log file       | %-50s |\n", output_log_file == NULL ? "NULL" : output_log_file);
    printf("| batch size B          | %-50d |\n", B);
    printf("| sequence length T     | %-50d |\n", T);
    printf("| learning rate         | %-50f |\n", learning_rate);
    printf("| val_loss_every        | %-50d |\n", val_loss_every);
    printf("| val_max_steps         | %-50d |\n", val_max_steps);
    printf("| sample_every          | %-50d |\n", sample_every);
    printf("| genT                  | %-50d |\n", genT);
    printf("+-----------------------+----------------------------------------------------+\n");

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
    printf("| device                | %-50s |\n", deviceProp.name);
    printf("| TF32                  | %-50s |\n", enable_tf32 ? "enabled" : "disabled");
    printf("+-----------------------+----------------------------------------------------+\n");

    // build the GPT-2 model from a checkpoint
    LLaMA3 model;
    // gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    /**
     * Not loading pre_trained model weights. Randomly Initializing model weights using Xavier Initialization
     */
    load_model_params(&model);
    printf("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf("| channels C            | %-50d |\n", model.config.channels);
    printf("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");

    // build DataLoaders for both train and val
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_data_pattern, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_data_pattern, B, T, 0, 1, 0);
    int train_num_batches = train_loader.num_tokens / (B * T); // let's do 1 epoch by default for now
    int val_num_batches = val_loader.num_tokens / (B * T);
    if (val_num_batches > val_max_steps)
    {
        val_num_batches = val_max_steps;
    }
    printf("| train_num_batches     | %-50d |\n", train_num_batches);
    printf("| val_num_batches       | %-50d |\n", val_num_batches);
    printf("+-----------------------+----------------------------------------------------+\n");

    // print model parameter allocations from gpt2_build_from_checkpoint down here to not mess up our table above
    printf("allocated %d MiB for model parameters\n", (int)round(model.num_parameters * sizeof(float) / (1024 * 1024)));

    // set up the Logger
    Logger logger;
    logger_init(&logger, output_log_file);

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int *gen_tokens = (int *)mallocCheck(B * T * sizeof(int));
    float *cpu_logits = (float *)mallocCheck(model.config.vocab_size * sizeof(float));

    // train
    struct timespec start, end;
    double total_sum_iteration_time_s = 0.0;
    for (int step = 0; step <= train_num_batches; step++)
    {
        int last_step = step == train_num_batches;

        // once in a while estimate the validation loss
        if (step % val_loss_every == 0 || last_step)
        {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++)
            {
                dataloader_next_batch(&val_loader);
                llama3_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
            logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % sample_every == 0 || last_step)
        {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for (int i = 0; i < B * T; ++i)
            {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++)
            {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                llama3_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                float *logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
                // move probs back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
                cudaCheck(cudaMemcpy(cpu_logits, logits, model.config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
                float coin = random_f32(&rng_state);
                int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok)
                {
                    const char *token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                }
                else
                {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // bit confusing: we want to make sure to eval and sample on 0th iteration
        // but also after the very last iteration. so we loop for step <= train_num_batches
        // instead of just < train_num_batches (one extra due to <=), only to do
        // the validation/sampling one last time, and then we break right here as we're done.
        if (last_step)
        {
            break;
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        llama3_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        llama3_zero_grad(&model);
        llama3_backward(&model);
        llama3_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
        cudaCheck(cudaDeviceSynchronize()); // finish all CUDA work to get correct precise timings
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        total_sum_iteration_time_s += time_elapsed_s;
        int tokens_per_second = (B * T) / time_elapsed_s;
        printf("step %4d/%d: train loss %f (%f ms, %d tok/s)\n", step + 1, train_num_batches, model.mean_loss, time_elapsed_s * 1000, tokens_per_second);
        logger_log_train(&logger, step, model.mean_loss);
    }
    // add a total average, for optimizations that are only mild improvements
    printf("total average iteration time: %f ms\n", total_sum_iteration_time_s / train_num_batches * 1000);

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    llama3_free(&model);
    free(cpu_logits);
    free(gen_tokens);
    cublasCheck(cublasDestroy(cublas_handle));
    logger_free(&logger);

    return 0;
}
#endif