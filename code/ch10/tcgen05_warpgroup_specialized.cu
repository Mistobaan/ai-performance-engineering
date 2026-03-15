/**
 * ch10: Advanced CUTLASS-style tcgen05 GEMM (2SM warp-specialized array mainloop).
 *
 * Mirrors CUTLASS's sm100_mma_array_warpspecialized pipeline with a 2-SM cluster
 * and a larger MMA tile. CUTLASS's SM100 array warp-specialized kernels use a
 * single MMA warp; the "advanced" aspect here is the 2SM tile and CUTLASS
 * scheduler/mainloop plumbing.
 */

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/detail/collective/moe_stride_utils.hpp>

#include <cute/tensor.hpp>

#include <memory>

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "CUTLASS was not built with tcgen05 (SM100) support enabled."
#endif

namespace warpgroup_specialized_impl {

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;

using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using MmaTileShape_MNK = Shape<_256, _128, _64>;
using ClusterShape_MNK = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementAccumulator,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,
    OperatorClass,
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK,
    ClusterShape_MNK,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

static inline void check_status(cutlass::Status status, const char* what) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, what, ": ", cutlassGetStatusString(status));
}

torch::Tensor run_warpgroup_specialized_matmul(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
  TORCH_CHECK(a.size(1) == b.size(1));
  TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16);
  TORCH_CHECK(a.is_cuda() && b.is_cuda());

  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  int64_t m = a_contig.size(0);
  int64_t k = a_contig.size(1);
  int64_t n = b_contig.size(0);

  TORCH_CHECK(m % 256 == 0 && n % 128 == 0 && k % 64 == 0,
              "Size must be divisible by tcgen05 tile");

  // Keep the benchmark focused on the 2SM CUTLASS schedule by writing the
  // final output in FP16 directly instead of launching a trailing cast.
  auto options = a.options().dtype(torch::kFloat16);
  auto c_buffer = torch::zeros({m, n}, options);
  auto d_buffer = torch::empty_like(c_buffer);

  auto shape_A = cute::make_shape(static_cast<int>(m), static_cast<int>(k), 1);
  auto shape_B = cute::make_shape(static_cast<int>(n), static_cast<int>(k), 1);
  auto shape_C = cute::make_shape(static_cast<int>(m), static_cast<int>(n), 1);
  StrideA stride_A = cutlass::make_internal_packed_stride(StrideA{}, shape_A);
  StrideB stride_B = cutlass::make_internal_packed_stride(StrideB{}, shape_B);
  StrideC stride_C = cutlass::make_internal_packed_stride(StrideC{}, shape_C);
  StrideD stride_D = cutlass::make_internal_packed_stride(StrideD{}, shape_C);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), 1},
      {reinterpret_cast<ElementA const*>(a_contig.data_ptr<at::Half>()), stride_A,
       reinterpret_cast<ElementB const*>(b_contig.data_ptr<at::Half>()), stride_B},
      {{1.0f, 0.0f},
       reinterpret_cast<ElementC const*>(c_buffer.data_ptr<at::Half>()),
       stride_C,
       reinterpret_cast<ElementD*>(d_buffer.data_ptr<at::Half>()),
       stride_D}};

  args.scheduler.max_swizzle_size = 0;

  size_t workspace_size = Gemm::get_workspace_size(args);
  torch::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size) {
    workspace = torch::empty({static_cast<long long>(workspace_size)},
                             a.options().dtype(torch::kUInt8));
    workspace_ptr = workspace.data_ptr<uint8_t>();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  check_status(Gemm::can_implement(args), "gemm.can_implement failed");
  check_status(gemm.initialize(args, workspace_ptr, stream), "gemm.initialize failed");
  check_status(gemm.run(stream), "gemm.run failed");
  AT_CUDA_CHECK(cudaGetLastError());

  return d_buffer;
}

}  // namespace warpgroup_specialized_impl

torch::Tensor matmul_tcgen05_warpgroup_specialized(torch::Tensor a, torch::Tensor b) {
  return warpgroup_specialized_impl::run_warpgroup_specialized_matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_tcgen05_warpgroup_specialized", &matmul_tcgen05_warpgroup_specialized);
}
