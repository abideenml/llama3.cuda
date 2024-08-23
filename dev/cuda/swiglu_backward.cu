/*
Kernels for swiglu forward pass.
NOTE: The results shown are performed on L4-GPU

Compile example:
nvcc -O3 --use_fast_math -lcublas -lcublasLt swiglu_backward.cu -o swiglu_backward

- version 1 is naive port from CPU code to kernel: parallelizes over B,T,C
./swiglu_backward 1

RESULTS:
block_size   32 | time 0.3049 ms | bandwidth 165.08 GB/s
block_size   64 | time 0.2911 ms | bandwidth 172.93 GB/s
block_size  128 | time 0.2918 ms | bandwidth 172.51 GB/s
block_size  256 | time 0.2910 ms | bandwidth 172.98 GB/s
block_size  512 | time 0.2906 ms | bandwidth 173.21 GB/s
block_size 1024 | time 0.2899 ms | bandwidth 173.64 GB/s


- version 2 uses 2 bfloat16 with the Packed128 data structure which helps in faster load/store operations, parallelizes over B,T,C
./swiglu_backward 2

RESULTS:
block_size   32 | time 0.2900 ms | bandwidth 173.59 GB/s
block_size   64 | time 0.2968 ms | bandwidth 169.59 GB/s
block_size  128 | time 0.2954 ms | bandwidth 170.37 GB/s
block_size  256 | time 0.2967 ms | bandwidth 169.64 GB/s
block_size  512 | time 0.2923 ms | bandwidth 172.22 GB/s
block_size 1024 | time 0.2895 ms | bandwidth 173.83 GB/s

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ENABLE_BF16
#include "common.h"

// ----------------------------------------------------------------------------
// CPU code reference

void swiglu_backward_cpu(float *dinp, float *dgate, const float *dout, const float *inp, const float *gate, int N)
{
    /**
     *
     * Reference Implementation:
     * y=SiLU(xW)*(xV)
     * z=xW
     * g=xV
     *
     * Using Chain-Rule:
     *
     * ∂SILU(z)/∂z = (σ(z) + SiLU(z)*(1-σ(z))
     *
     * ∂L/∂z = ∂L/∂out * ∂SILU(z)/∂z * g
     * ∂L/∂g = ∂L/∂out * SILU(z)/∂z
     *
     */
    for (int i = 0; i < N; i++)
    {
        // Recalculating SiLU & calculating sigmoid
        float sigmoid = 1.0f / (1.0f + expf(-inp[i]));
        float siLU = inp[i] / (1.0f + expf(-inp[i]));

        // Gradient of SwiGLU w.r.t inp
        dinp[i] = (dout[i] * (sigmoid + siLU * (1.0f - sigmoid)) * gate[i]);

        // Gradient of SwiGLU w.r.t gate
        dgate[i] = (dout[i] * siLU);
    }
}

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void swiglu_backward_kernel1(floatX *dinp, floatX *dgate,
                                        const floatX *dout,
                                        const floatX *inp, const floatX *gate, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float inp_i = (float)inp[i];
        float gate_i = (float)gate[i];
        float dout_i = (float)dout[i];

        // Calculate sigmoid and SiLU (Swish) activations
        float sigmoid = 1.0f / (1.0f + expf(-inp_i));
        float siLU = inp_i * sigmoid;

        // Perform gradient computations
        dinp[i] = (floatX)(dout_i * (sigmoid + siLU * (1.0f - sigmoid)) * gate_i);
        dgate[i] = (floatX)(dout_i * siLU);
    }
}

/**
 * Uses `x128`, a custom-datatype responsible for faster processing (load and store) capabilities
 *
 *  By loading, processing, and storing 4 floats at a time (or however many the data type holds), you reduce the number of memory accesses and arithmetic operations, and thus resulting in better throughput and efficiency
 */
__global__ void swiglu_backward_kernel2(floatX *dinp, floatX *dgate,
                                        const floatX *dout,
                                        const floatX *inp, const floatX *gate, int N)
{
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (i < N)
    {
        // Packed variables for dinp, dgate, dout, inp, gate
        x128 packed_dinp, packed_dgate;
        x128 packed_dout = load128cs(dout + i);
        x128 packed_inp = load128cs(inp + i);
        x128 packed_gate = load128cs(gate + i);

        for (int k = 0; k < packed_inp.size; ++k)
        {
            // Compute the SiLU and sigmoid for each packed element
            float xiW = (float)packed_inp[k];  // input
            float xiV = (float)packed_gate[k]; // gate
            float dxi = (float)packed_dout[k]; // dout

            // Calculate sigmoid and SiLU (Swish) activations
            float sigmoid = 1.0f / (1.0f + expf(-xiW));
            float siLU = xiW * sigmoid;

            // Gradient of SwiGLU w.r.t inp and gate
            packed_dinp[k] = (floatX)(dxi * (sigmoid + siLU * (1.0f - sigmoid)) * xiV);
            packed_dgate[k] = (floatX)(dxi * siLU);
        }

        // Store the gradients back to global memory
        store128(dinp + i, packed_dinp);
        store128(dgate + i, packed_dgate);
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void swiglu_backward1(floatX *dinp, floatX *dgate,
                      const floatX *dout,
                      const floatX *inp, const floatX *gate, int N,
                      int block_size)
{

    const int grid_size = ceil_div(N, block_size);
    swiglu_backward_kernel1<<<grid_size, block_size>>>(dinp, dgate, dout, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

void swiglu_backward2(floatX *dinp, floatX *dgate,
                      const floatX *dout,
                      const floatX *inp, const floatX *gate, int N,
                      int block_size)
{
    const int grid_size = ceil_div(N, block_size * x128::size);
    swiglu_backward_kernel2<<<grid_size, block_size>>>(dinp, dgate, dout, inp, gate, N);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void swiglu_backward(int kernel_num, floatX *dinp, floatX *dgate,
                     const floatX *dout,
                     const floatX *inp, const floatX *gate,
                     int B, int T, int C,
                     int block_size)
{
    switch (kernel_num)
    {
    case 1:
        swiglu_backward1(dinp, dgate, dout, inp, gate, B * T * C, block_size);
        break;
    case 2:
        swiglu_backward2(dinp, dgate, dout, inp, gate, B * T * C, block_size);
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

    // create host memory of random numbers
    float *dinp = (float *)malloc(B * T * C * sizeof(float));
    float *dgate = (float *)malloc(B * T * C * sizeof(float));

    float *inp = (float *)make_random_float(B * T * C);  // precomputed x*W
    float *gate = (float *)make_random_float(B * T * C); // precomputed x*V
    float *dout = (float *)make_random_float(B * T * C);

    // read kernel_num from command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_num);

    // first check the correctness of the kernel
    swiglu_backward_cpu(dinp, dgate, dout, inp, gate, B * T * C);

    // move to GPU
    floatX *d_dinp;
    floatX *d_dgate;
    floatX *d_inp;
    floatX *d_gate;
    floatX *d_dout;
    cudaCheck(cudaMalloc(&d_dinp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dgate, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_gate, B * T * C * sizeof(floatX)));
    cudaCheck(cudaMalloc(&d_dout, B * T * C * sizeof(floatX)));
    cudaCheck(memcpy_convert(d_inp, inp, B * T * C));
    cudaCheck(memcpy_convert(d_gate, gate, B * T * C));
    cudaCheck(memcpy_convert(d_dout, dout, B * T * C));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        swiglu_backward(kernel_num, d_dinp, d_dgate, d_dout, d_inp, d_gate, B, T, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_dinp, dinp, "dinp", B * T * C, tol);
        validate_result(d_dgate, dgate, "dgate", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, swiglu_backward,
                                              kernel_num, d_dinp, d_dgate, d_dout,
                                              d_inp, d_gate, B, T, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 1 read and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * (int)sizeof(floatX); // Include dout in memory operations
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(dinp);
    free(dgate);
    free(inp);
    free(gate);
    free(dout);

    cudaCheck(cudaFree(d_dinp));
    cudaCheck(cudaFree(d_dgate));
    cudaCheck(cudaFree(d_inp));
    cudaCheck(cudaFree(d_gate));
    cudaCheck(cudaFree(d_dout));
    return 0;
}