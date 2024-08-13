# llama3.cuda

llama3.cuda is an implementation of Llama 3.1 in pure C/CUDA. I have implemented three major kernels:

### `swiglu_forward_kernel`

```__global__ void swiglu_forward_kernel(float *out, const float *inp, const float *gate, int N)
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
```

### `attention_forward_gqa`

```
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
```

### `apply_rope_forward`

```
// parallelized over b,t,c
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
```

These kernels will help us implement the necessary LLaMA architecture.

Respectively, their backward kernels are also implemented as:

- `swiglu_backward_kernel`
- `apply_rope_backward_kernel`
- `attention_backward_gqa`

Along, with these kernels, I have also implemented 2 helper kernels, named `precompute_freqs_cis_kernel` done of which will help us compute the`cis`-component of RoPE, and other `repeat_interleave_forward_kernel` (along with its backward-pass) which will help in implementing correctly the kv-heads.

## Setup

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_llama_fp32cu
./train_llamafp32cu``
```

## 📝 Implementation Details:

In `attention_forward_gqa`, I have leveraged the optimized code from `llm.c` that uses CuBLAS for matmul-computations, with addition of positional information in `q` and `k` matrices, with the rotational-component of `freq_cos` and `freq_sin` (computed from `precompute_freq_cis kernel`).

- `apply_rope_forward_kernel` and `apply_rope_backward_kernel`, both kernels are implemented using simple parallelizing techniques, parallelized over b,t,c.
- `swiglu_forward_kernel` leverages simple parallelizing technique over b,t,c. The `inp` and `gate` params to swiglu are computed using the `matmul_kernel` (very-optimized, utilized cuBLAS).
- `precompute_freq_cis` kernel also leverages simple parallelizing technique over c/2 (`embed_dim/2`) elements, since in one-invokation, we are computing 2-components `freq_cos` (real-part), and `freq_sin`(imaginary part), for each embed_column.



## 🤞 Todos

Finally there are a couple more todos which I'll hopefully add really soon:
* Currently, attention_forward_GQA only takes into account num_kv_heads as half of the NH (num_kv_heads = NH/2). An important To-do will be to add support any integer divisor of NH as `num_kv_heads` (num_kv_heads = integar \* NH => NH % num_kv_heads == 0) 
* Implementing these simply parallelized kernels by furthur optimizing and utilizing advanced CUDA-techniques, such as: 
 🌟 using `Shared-Memory` and `reductions` for faster access 
 🌟 Using `co-operative groups` to implement warp-level synchronization (and see if significant perf gains).
 🌟 Apply kernel fusions if possible.