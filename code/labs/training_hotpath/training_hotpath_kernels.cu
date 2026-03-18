// CUDA kernels for the training-hotpath lab.

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#define CHECK_CUDA(x)                                                        \
  do {                                                                       \
    cudaError_t status__ = (x);                                              \
    TORCH_CHECK(status__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(status__)); \
  } while (0)

namespace {

__global__ void segment_abs_mean_kernel(
    const float* flat,
    const int64_t* offsets,
    float* out,
    int64_t num_segments) {
  int segment = blockIdx.x;
  if (segment >= num_segments) {
    return;
  }

  int64_t start = offsets[segment];
  int64_t stop = offsets[segment + 1];
  float local_sum = 0.0f;
  for (int64_t idx = start + threadIdx.x; idx < stop; idx += blockDim.x) {
    local_sum += fabsf(flat[idx]);
  }

  __shared__ float shared[256];
  shared[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    int64_t count = stop > start ? (stop - start) : 1;
    out[segment] = shared[0] / static_cast<float>(count);
  }
}

__global__ void pack_rows_kernel(
    const float* input,
    const int64_t* row_indices,
    float* output,
    int64_t num_rows,
    int64_t num_cols) {
  int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_rows * num_cols;
  if (linear_idx >= total) {
    return;
  }
  int64_t out_row = linear_idx / num_cols;
  int64_t col = linear_idx % num_cols;
  int64_t in_row = row_indices[out_row];
  output[linear_idx] = input[in_row * num_cols + col];
}

__global__ void scatter_rows_kernel(
    const float* packed,
    const int64_t* row_indices,
    float* output,
    int64_t num_rows,
    int64_t num_cols) {
  int64_t linear_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = num_rows * num_cols;
  if (linear_idx >= total) {
    return;
  }
  int64_t packed_row = linear_idx / num_cols;
  int64_t col = linear_idx % num_cols;
  int64_t out_row = row_indices[packed_row];
  output[out_row * num_cols + col] = packed[linear_idx];
}

}  // namespace

torch::Tensor segment_abs_mean(torch::Tensor flat, torch::Tensor offsets) {
  TORCH_CHECK(flat.is_cuda(), "flat must be a CUDA tensor");
  TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
  TORCH_CHECK(flat.dtype() == torch::kFloat32, "flat must be float32");
  TORCH_CHECK(offsets.dtype() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(flat.dim() == 1, "flat must be 1D");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
  TORCH_CHECK(offsets.size(0) >= 2, "offsets must contain at least two elements");

  auto flat_contig = flat.contiguous();
  auto offsets_contig = offsets.contiguous();
  auto num_segments = offsets_contig.size(0) - 1;
  auto out = torch::zeros({num_segments}, flat.options());

  constexpr int threads = 256;
  segment_abs_mean_kernel<<<num_segments, threads>>>(
      flat_contig.data_ptr<float>(),
      offsets_contig.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      num_segments);
  CHECK_CUDA(cudaGetLastError());
  return out;
}

torch::Tensor pack_rows(torch::Tensor input, torch::Tensor row_indices) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
  TORCH_CHECK(row_indices.dtype() == torch::kInt64, "row_indices must be int64");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");

  auto input_contig = input.contiguous();
  auto rows_contig = row_indices.contiguous();
  auto num_rows = rows_contig.size(0);
  auto num_cols = input_contig.size(1);
  auto output = torch::empty({num_rows, num_cols}, input.options());

  constexpr int threads = 256;
  int64_t total = num_rows * num_cols;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  pack_rows_kernel<<<blocks, threads>>>(
      input_contig.data_ptr<float>(),
      rows_contig.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      num_rows,
      num_cols);
  CHECK_CUDA(cudaGetLastError());
  return output;
}

torch::Tensor scatter_rows(torch::Tensor packed, torch::Tensor row_indices, int64_t total_rows) {
  TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
  TORCH_CHECK(row_indices.is_cuda(), "row_indices must be a CUDA tensor");
  TORCH_CHECK(packed.dtype() == torch::kFloat32, "packed must be float32");
  TORCH_CHECK(row_indices.dtype() == torch::kInt64, "row_indices must be int64");
  TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
  TORCH_CHECK(row_indices.dim() == 1, "row_indices must be 1D");

  auto packed_contig = packed.contiguous();
  auto rows_contig = row_indices.contiguous();
  auto num_rows = rows_contig.size(0);
  auto num_cols = packed_contig.size(1);
  auto output = torch::zeros({total_rows, num_cols}, packed.options());

  constexpr int threads = 256;
  int64_t total = num_rows * num_cols;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  scatter_rows_kernel<<<blocks, threads>>>(
      packed_contig.data_ptr<float>(),
      rows_contig.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      num_rows,
      num_cols);
  CHECK_CUDA(cudaGetLastError());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("segment_abs_mean", &segment_abs_mean, "Segmented abs-mean reduction");
  m.def("pack_rows", &pack_rows, "Pack active rows into a dense tensor");
  m.def("scatter_rows", &scatter_rows, "Scatter packed rows back into padded layout");
}
