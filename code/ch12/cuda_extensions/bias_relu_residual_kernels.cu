// bias_relu_residual_kernels.cu
// Baseline: two kernels (bias+ReLU, residual add)
// Fused: one kernel (bias+ReLU+residual in one pass)

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <cmath>

namespace {

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                            \
        TORCH_CHECK(status == cudaSuccess,                                     \
                    "CUDA error at ", __FILE__, ":", __LINE__, " - ",          \
                    cudaGetErrorString(status));                                \
    } while (0)

#define CHECK_TENSOR(t, name)                                                  \
    do {                                                                      \
        TORCH_CHECK((t).is_cuda(), (name), " must be CUDA tensor");             \
        TORCH_CHECK((t).is_contiguous(), (name), " must be contiguous");        \
        TORCH_CHECK((t).dtype() == torch::kFloat32, (name), " must be float32");\
        TORCH_CHECK((t).numel() > 0, (name), " must have at least one element");\
    } while (0)

} // namespace

__global__ void kernel_bias_relu(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ out,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] + bias[idx];
        out[idx] = fmaxf(val, 0.0f);
    }
}

__global__ void kernel_residual_add(const float* __restrict__ activated,
                                   const float* __restrict__ residual,
                                   float* __restrict__ out,
                                   int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = activated[idx] + residual[idx];
    }
}

__global__ void kernel_fused(const float* __restrict__ input,
                            const float* __restrict__ bias,
                            const float* __restrict__ residual,
                            float* __restrict__ out,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(input[idx] + bias[idx], 0.0f) + residual[idx];
    }
}

static void launch_config(int n, int& blocks, int& threads) {
    threads = 256;
    blocks = (n + threads - 1) / threads;
}

void separate_kernels(torch::Tensor input,
                     torch::Tensor bias,
                     torch::Tensor residual,
                     torch::Tensor tmp,
                     torch::Tensor output,
                     int64_t iterations) {
    CHECK_TENSOR(input, "input");
    CHECK_TENSOR(bias, "bias");
    CHECK_TENSOR(residual, "residual");
    CHECK_TENSOR(tmp, "tmp");
    CHECK_TENSOR(output, "output");

    auto n = input.numel();
    TORCH_CHECK(bias.numel() == n, "bias must match input numel");
    TORCH_CHECK(residual.numel() == n, "residual must match input numel");
    TORCH_CHECK(tmp.numel() == n, "tmp must match input numel");
    TORCH_CHECK(output.numel() == n, "output must match input numel");
    TORCH_CHECK(iterations >= 1, "iterations must be >= 1");

    int threads = 0;
    int blocks = 0;
    launch_config(static_cast<int>(n), blocks, threads);

    auto stream = at::cuda::getDefaultCUDAStream();
    cudaStream_t cuda_stream = stream.stream();

    for (int64_t i = 0; i < iterations; ++i) {
        kernel_bias_relu<<<blocks, threads, 0, cuda_stream>>>(
            input.data_ptr<float>(),
            bias.data_ptr<float>(),
            tmp.data_ptr<float>(),
            static_cast<int>(n)
        );
        kernel_residual_add<<<blocks, threads, 0, cuda_stream>>>(
            tmp.data_ptr<float>(),
            residual.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(n)
        );
    }

    CHECK_CUDA(cudaGetLastError());
}

void fused_kernel(torch::Tensor input,
                 torch::Tensor bias,
                 torch::Tensor residual,
                 torch::Tensor output,
                 int64_t iterations) {
    CHECK_TENSOR(input, "input");
    CHECK_TENSOR(bias, "bias");
    CHECK_TENSOR(residual, "residual");
    CHECK_TENSOR(output, "output");

    auto n = input.numel();
    TORCH_CHECK(bias.numel() == n, "bias must match input numel");
    TORCH_CHECK(residual.numel() == n, "residual must match input numel");
    TORCH_CHECK(output.numel() == n, "output must match input numel");
    TORCH_CHECK(iterations >= 1, "iterations must be >= 1");

    int threads = 0;
    int blocks = 0;
    launch_config(static_cast<int>(n), blocks, threads);

    auto stream = at::cuda::getDefaultCUDAStream();
    cudaStream_t cuda_stream = stream.stream();

    for (int64_t i = 0; i < iterations; ++i) {
        kernel_fused<<<blocks, threads, 0, cuda_stream>>>(
            input.data_ptr<float>(),
            bias.data_ptr<float>(),
            residual.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(n)
        );
    }

    CHECK_CUDA(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("separate_kernels", &separate_kernels, "Bias+ReLU + residual (separate kernels)");
    m.def("fused_kernel", &fused_kernel, "Bias+ReLU + residual (fused kernel)");
}
