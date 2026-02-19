#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

// This probe is compiled with NVCC, but we want to use CUTLASS/CuTe TMEM layout helpers
// on the host without triggering GCC's "always_inline function not inlinable" errors
// on mutually-recursive constexpr helpers (e.g., rotl/rotr). Override the CuTe
// force-inline macros to keep the probe build stable.
#include "cute/config.hpp"
#undef CUTE_HOST_DEVICE
#undef CUTE_DEVICE
#undef CUTE_HOST
#define CUTE_HOST_DEVICE __host__ __device__
#define CUTE_DEVICE __device__
#define CUTE_HOST __host__

#include "cute/atom/mma_traits_sm100.hpp"
#include "cute/tensor.hpp"
#include "cutlass/float8.h"

namespace {

// CUTLASS tmem_ptr uses rotr(offset, OffsetShift) for subword addressing.
inline uint32_t rotr32(uint32_t x, int s) {
  if (s == 0) {
    return x;
  }
  return (x >> s) | (x << (32 - s));
}

// -----------------------------------------------------------------------------
// Minimal tcgen05 + UMMA helpers for a device-side UTCCP+TMEM dump probe.
//
// Goal: empirically verify how `tcgen05.cp.cta_group::2.*` places the packed 128x16
// scale tiles into TMEM, then decode the mapping with the CUTLASS `tmem_sf_frg`
// layout on the host.
// -----------------------------------------------------------------------------

namespace cg = cooperative_groups;

namespace tcgen05 {

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ uint32_t block_rank_in_cluster() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank) :);
  return rank;
#else
  return 0u;
#endif
}

__device__ __forceinline__ void tmem_alloc_cta2(uint32_t* dst_ptr_smem, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr_smem);
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(dst_intptr), "r"(num_columns));
#else
  (void)dst_ptr_smem;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_dealloc_cta2(uint32_t tmem_ptr, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               "tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1; \n\t"
               "}\n"
               :
               : "r"(tmem_ptr), "r"(num_columns));
#else
  (void)tmem_ptr;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_relinquish_alloc_permit_cta2() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::);
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc));
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ uint32_t tmem_ld_32dp32b_x1(uint32_t src_addr) {
  uint32_t dst0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32"
               "{%0},"
               "[%1];\n"
               : "=r"(dst0)
               : "r"(src_addr));
#else
  dst0 = 0;
  (void)src_addr;
#endif
  return dst0;
}

__device__ __forceinline__ void tmem_ld_32dp32b_x4(uint32_t src_addr,
                                                   uint32_t& dst0,
                                                   uint32_t& dst1,
                                                   uint32_t& dst2,
                                                   uint32_t& dst3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32"
               "{%0, %1, %2, %3},"
               "[%4];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
               : "r"(src_addr));
#else
  dst0 = dst1 = dst2 = dst3 = 0u;
  (void)src_addr;
#endif
}

__device__ __forceinline__ void tmem_wait_st_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
#endif
}

__device__ __forceinline__ void tmem_wait_ld_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
#endif
}

// TMEM pointers are encoded as {col:16, dp:8, idx:8}. When forming derived addresses,
// preserve idx from the allocated base pointer.
__device__ __forceinline__ uint32_t tmem_addr_add(uint32_t base, uint32_t dp_add, uint32_t col_add) {
  const uint32_t base_col = base & 0x0000FFFFu;
  const uint32_t base_dp = (base >> 16) & 0x000000FFu;
  const uint32_t base_idx = base & 0xFF000000u;
  const uint32_t col = base_col + col_add;
  const uint32_t dp = (base_dp + dp_add) & 0x000000FFu;
  return base_idx | (dp << 16) | (col & 0x0000FFFFu);
}

}  // namespace tcgen05

namespace umma {

enum class LayoutType : uint8_t {
  SWIZZLE_NONE = 0,
};

union SmemDescriptor {
  uint64_t desc_ = 0;
  struct {
    uint16_t start_address_ : 14, : 2;
    uint16_t leading_byte_offset_ : 14, : 2;
    uint16_t stride_byte_offset_ : 14, version_ : 2;
    uint8_t : 1, base_offset_ : 3, lbo_mode_ : 1, : 3;
    uint8_t : 5, layout_type_ : 3;
  };
  __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

}  // namespace umma

// Probe dimensions:
// We want to understand how UTCCP places the packed 128x16 scale tile into TMEM for
// cta_group::2. Two common unknowns are:
// 1) Whether the second 32-row half of a 64-row UTCCP copy is encoded via DP (dp+32/64), and
// 2) Whether it is encoded via the per-seg column sub-index (col_sub within a seg's column span).
//
// So we probe:
// - dp windows: {0..31}, {32..63}, {64..95}, {96..127}
// - per-seg col_sub: {0,1} within the seg base pointer (seg stride = 2 cols in CUTLASS' mapping)
// reading word0 (bytes 0..3) per dp.
constexpr int kProbeDp = 32;
constexpr int kProbeDpWindows = 4;
constexpr int kProbeColSubs = 2;
// Probe more "virtual seg" indices than CUTLASS exposes (0..7) to see if UTCCP64 places the
// second 32-row half of each 64-row copy into additional column ranges beyond the CUTLASS
// (seg=0..3) base pointers.
constexpr int kProbeSegs = 8;

// pattern_kind:
//   0 -> byte value = row (0..127)
//   1 -> byte value = col (0..15)
__global__ void utccp_cta2_scale_dump_kernel(uint32_t* out_words,
                                             int pattern_kind,
                                             int use64,
                                             int schedule,
                                             int desc_step_rows,
                                             int skip_utccp,
                                             int skip_dump,
                                             int dst_col_add) {
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(tcgen05::block_rank_in_cluster());

  __shared__ uint32_t tmem_base;
  // NOTE: tcgen05.alloc/dealloc are warp-synchronous (must be issued by a fully-active warp).
  // Also, cta_group::2 requires both CTAs reach allocation/deallocation points together.
  cluster.sync();
  if (threadIdx.x < 32) {
    // Allocate the full TMEM slice to avoid allocator slice interleaving.
    tcgen05::tmem_alloc_cta2(&tmem_base, /*num_columns=*/512);
  }
  __syncthreads();
  cluster.sync();

  __shared__ alignas(16) uint8_t sScale[128][16];
  const int tid = static_cast<int>(threadIdx.x);
  for (int idx = tid; idx < 128 * 16; idx += static_cast<int>(blockDim.x)) {
    const int r = idx / 16;
    const int c = idx - r * 16;
    const uint8_t v = (pattern_kind == 0) ? static_cast<uint8_t>(r) : static_cast<uint8_t>(c);
    sScale[r][c] = v;
  }
  __syncthreads();
  cluster.sync();

  // Build the canonical SWIZZLE_NONE UMMA/UTCCP SMEM descriptor used for packed scale tiles.
  umma::SmemDescriptor desc{};
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.base_offset_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_NONE);
  const uint32_t s_addr = tcgen05::cast_smem_ptr_to_uint(&sScale[0][0]);
  desc.start_address_ = static_cast<uint16_t>(s_addr >> 4);  // u128
  desc.leading_byte_offset_ = 1;
  desc.stride_byte_offset_ = 8;
  const uint64_t desc_base = static_cast<uint64_t>(desc);

  // CUTLASS-probed base columns for cta_group::2 M_MMA=64 scale fragments:
  // - SFA: rank*8 + seg*2
  // Using UTCCP64 we copy two 32-row segments at once: seg01 at col0, seg23 at col4.
  const uint32_t tmem_rank_base =
      tcgen05::tmem_addr_add(tmem_base,
                             /*dp_add=*/0u,
                             /*col_add=*/static_cast<uint32_t>(rank) * 8u + static_cast<uint32_t>(dst_col_add));
  const uint32_t dst01 = tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/0u);
  const uint32_t dst23 = tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/4u);

  if (!skip_utccp) {
    if (use64) {
      // `warpx2` means the instruction must be executed by 2 participating warps.
      if (threadIdx.x < 64) {
        // 64x128b covers two 32-row segments per op.
        const uint64_t src01 = desc_base;
        // `desc_step_rows` is in units of 16B rows (u128) because `SmemDescriptor.start_address_`
        // is expressed in u128. A value of 64 corresponds to +64 rows = +1024 bytes.
        const uint64_t src23 = desc_base + static_cast<uint64_t>(desc_step_rows);
        if (schedule == 0) {
          tcgen05::utccp_cp_cta2_64x128b_warpx2_01_23(src01, dst01);
          tcgen05::utccp_cp_cta2_64x128b_warpx2_01_23(src23, dst23);
        } else {
          tcgen05::utccp_cp_cta2_64x128b_warpx2_02_13(src01, dst01);
          tcgen05::utccp_cp_cta2_64x128b_warpx2_02_13(src23, dst23);
        }
      }
    } else {
      // `warpx4` means the instruction must be executed by 4 participating warps.
      if (threadIdx.x < 128) {
        // 32x128b per segment.
        constexpr int kDescStep32 = 32;
        for (int seg = 0; seg < 4; ++seg) {
          const uint64_t src = desc_base + static_cast<uint64_t>(seg * kDescStep32);
          const uint32_t dst =
              tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(seg) * 2u);
          tcgen05::utccp_cp_cta2_32x128b_warpx4(src, dst);
        }
      }
    }
  }
  __syncthreads();
  cluster.sync();
  tcgen05::tmem_wait_st_sync();
  tcgen05::tmem_wait_ld_sync();

  // Targeted TMEM probe: read dp windows and per-seg column sub-indices from the 4 seg base pointers
  // (seg col stride = 2).
  // Store word0 (bytes 0..3) for each dp within each (seg, col_sub).
  if (!skip_dump) {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp < kProbeDpWindows) {
      const int dp_lane = lane;
      const int w = warp;  // dp window index
      const uint32_t dp_add = static_cast<uint32_t>(w * kProbeDp);
      const int base = rank * (kProbeSegs * kProbeColSubs * kProbeDpWindows * kProbeDp);
      for (int seg = 0; seg < kProbeSegs; ++seg) {
        const uint32_t seg_base =
            tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(seg) * 2u);
        for (int cs = 0; cs < kProbeColSubs; ++cs) {
          const uint32_t addr =
              tcgen05::tmem_addr_add(seg_base, /*dp_add=*/dp_add, /*col_add=*/static_cast<uint32_t>(cs));
          const uint32_t w0 = tcgen05::tmem_ld_32dp32b_x1(addr);
          const int out_base = base + (((seg * kProbeColSubs + cs) * kProbeDpWindows) + w) * kProbeDp;
          out_words[out_base + dp_lane] = w0;
        }
      }
    }
  }

  __syncthreads();
  cluster.sync();
  if (threadIdx.x < 32) {
    tcgen05::tmem_dealloc_cta2(tmem_base, /*num_columns=*/512);
    tcgen05::tmem_relinquish_alloc_permit_cta2();
  }
}

static void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
    std::exit(2);
  }
}

static uint8_t byte_from_word(uint32_t word, uint32_t sf_id) {
  const uint32_t shift = (sf_id & 3u) * 8u;
  return static_cast<uint8_t>((word >> shift) & 0xFFu);
}

struct AddrFields {
  uint32_t col = 0;
  uint32_t dp = 0;
  uint32_t sf_id = 0;
};

static AddrFields decode_addr(uint32_t addr) {
  AddrFields f;
  f.col = addr & 0xFFFFu;
  f.dp = (addr >> 16) & 0xFFu;
  f.sf_id = (addr >> 30) & 0x3u;
  return f;
}

static void run_device_probe(int use64, int schedule) {
  std::printf("\n=== Device UTCCP+TMEM dump (cta_group::2) ===\n");
  std::printf("use64=%d schedule=%d (0=01_23, 1=02_13)\n", use64, schedule);
  const int desc_step_rows = std::getenv("AISP_TMEM_SF_FRG_PROBE_DESC_STEP_ROWS")
                                 ? std::atoi(std::getenv("AISP_TMEM_SF_FRG_PROBE_DESC_STEP_ROWS"))
                                 : 64;
  std::printf("desc_step_rows=%d (added to SmemDescriptor.start_address_ for the second UTCCP64 op)\n",
              desc_step_rows);
  const int dst_col_add = std::getenv("AISP_TMEM_SF_FRG_PROBE_DST_COL_ADD")
                              ? std::atoi(std::getenv("AISP_TMEM_SF_FRG_PROBE_DST_COL_ADD"))
                              : 0;
  std::printf("dst_col_add=%d (added to rank*8 base column before UTCCP destination formation)\n", dst_col_add);
  const int verbose_dump = std::getenv("AISP_TMEM_SF_FRG_PROBE_VERBOSE") ? 1 : 0;
  const int skip_utccp = std::getenv("AISP_TMEM_SF_FRG_PROBE_SKIP_UTCCP") ? 1 : 0;
  const int skip_dump = std::getenv("AISP_TMEM_SF_FRG_PROBE_SKIP_DUMP") ? 1 : 0;
  if (skip_utccp) {
    std::printf("NOTE: skipping UTCCP (AISP_TMEM_SF_FRG_PROBE_SKIP_UTCCP=1)\n");
  }
  if (skip_dump) {
    std::printf("NOTE: skipping TMEM dump loads (AISP_TMEM_SF_FRG_PROBE_SKIP_DUMP=1)\n");
  }
  if (verbose_dump) {
    std::printf("NOTE: verbose dump enabled (AISP_TMEM_SF_FRG_PROBE_VERBOSE=1)\n");
  }

  // Cluster launches require explicitly allowing non-portable cluster sizes on some drivers.
  cuda_check(cudaFuncSetAttribute(utccp_cta2_scale_dump_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
             "cudaFuncSetAttribute(nonportable_cluster_size)");

  const int dump_elems = 2 * (kProbeSegs * kProbeColSubs * kProbeDpWindows * kProbeDp);
  uint32_t* d_dump_row = nullptr;
  uint32_t* d_dump_col = nullptr;
  cuda_check(cudaMalloc(&d_dump_row, sizeof(uint32_t) * static_cast<size_t>(dump_elems)), "cudaMalloc(d_dump_row)");
  cuda_check(cudaMalloc(&d_dump_col, sizeof(uint32_t) * static_cast<size_t>(dump_elems)), "cudaMalloc(d_dump_col)");
  cuda_check(cudaMemset(d_dump_row, 0, sizeof(uint32_t) * static_cast<size_t>(dump_elems)), "cudaMemset(row)");
  cuda_check(cudaMemset(d_dump_col, 0, sizeof(uint32_t) * static_cast<size_t>(dump_elems)), "cudaMemset(col)");

  auto launch = [&](uint32_t* out, int pattern_kind) {
    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(2, 1, 1);
    cfg.blockDim = dim3(128, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = nullptr;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    cuda_check(cudaLaunchKernelEx(&cfg,
                                 utccp_cta2_scale_dump_kernel,
                                 out,
                                 pattern_kind,
                                 use64,
                                 schedule,
                                 desc_step_rows,
                                 skip_utccp,
                                 skip_dump,
                                 dst_col_add),
               "cudaLaunchKernelEx(utccp_cta2_scale_dump_kernel)");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize()");
  };

  // Two passes to disambiguate (smem_row, smem_col) source coordinates.
  launch(d_dump_row, /*pattern_kind=*/0);
  launch(d_dump_col, /*pattern_kind=*/1);

  std::vector<uint32_t> h_row(static_cast<size_t>(dump_elems));
  std::vector<uint32_t> h_col(static_cast<size_t>(dump_elems));
  cuda_check(cudaMemcpy(h_row.data(), d_dump_row, sizeof(uint32_t) * static_cast<size_t>(dump_elems), cudaMemcpyDeviceToHost),
             "cudaMemcpy(row)");
  cuda_check(cudaMemcpy(h_col.data(), d_dump_col, sizeof(uint32_t) * static_cast<size_t>(dump_elems), cudaMemcpyDeviceToHost),
             "cudaMemcpy(col)");

  cuda_check(cudaFree(d_dump_row), "cudaFree(row)");
  cuda_check(cudaFree(d_dump_col), "cudaFree(col)");

  auto word_at = [&](int rank, int seg, int cs, int w, int dp_lane, bool is_row) -> uint32_t {
    if (rank < 0 || rank > 1 || seg < 0 || seg >= kProbeSegs || cs < 0 || cs >= kProbeColSubs || w < 0 ||
        w >= kProbeDpWindows || dp_lane < 0 || dp_lane >= kProbeDp) {
      return 0u;
    }
    const int base = rank * (kProbeSegs * kProbeColSubs * kProbeDpWindows * kProbeDp);
    const size_t idx = static_cast<size_t>(base + (((seg * kProbeColSubs + cs) * kProbeDpWindows) + w) * kProbeDp + dp_lane);
    return (is_row ? h_row : h_col)[idx];
  };

  auto dump_seg = [&](int rank, int seg, int cs, int w) {
    const int dp_base = w * kProbeDp;
    std::printf("rank=%d seg=%d col_sub=%d dp%d..%d: ", rank, seg, cs, dp_base, dp_base + kProbeDp - 1);
    for (int dp_lane = 0; dp_lane < kProbeDp; ++dp_lane) {
      const uint32_t w_row = word_at(rank, seg, cs, w, dp_lane, /*is_row=*/true);
      const uint32_t w_col = word_at(rank, seg, cs, w, dp_lane, /*is_row=*/false);
      const uint8_t r0 = byte_from_word(w_row, /*sf_id=*/0);
      const uint8_t c0 = byte_from_word(w_col, /*sf_id=*/0);
      std::printf("(%u,%u)%s", (unsigned)r0, (unsigned)c0, (dp_lane + 1 == kProbeDp) ? "" : " ");
    }
    std::printf("\n");
  };

  auto summarize_seg = [&](int rank, int seg, int cs, int w) {
    const int dp_base = w * kProbeDp;
    const uint32_t w_row0 = word_at(rank, seg, cs, w, /*dp_lane=*/0, /*is_row=*/true);
    const uint32_t w_col0 = word_at(rank, seg, cs, w, /*dp_lane=*/0, /*is_row=*/false);
    std::printf("rank=%d seg=%d col_sub=%d dp_base=%d  word0(row)=0x%08x word0(col)=0x%08x\n",
                rank, seg, cs, dp_base, w_row0, w_col0);
    for (int b = 0; b < 4; ++b) {
      bool seen_row[256] = {false};
      bool seen_col[256] = {false};
      int min_row = 255, max_row = 0, uniq_row = 0;
      int min_col = 255, max_col = 0, uniq_col = 0;
      const int row_base = static_cast<int>(byte_from_word(w_row0, static_cast<uint32_t>(b)));
      const int col_base = static_cast<int>(byte_from_word(w_col0, static_cast<uint32_t>(b)));
      bool row_linear1 = true;
      bool row_linear2 = true;  // common case: each logical row appears twice (16B row split into 2x 8B DP slots)
      bool col_const = true;
      for (int dp_lane = 0; dp_lane < kProbeDp; ++dp_lane) {
        const uint32_t w_row = word_at(rank, seg, cs, w, dp_lane, /*is_row=*/true);
        const uint32_t w_col = word_at(rank, seg, cs, w, dp_lane, /*is_row=*/false);
        const uint8_t r = byte_from_word(w_row, static_cast<uint32_t>(b));
        const uint8_t c = byte_from_word(w_col, static_cast<uint32_t>(b));
        if (!seen_row[r]) {
          seen_row[r] = true;
          ++uniq_row;
        }
        if (!seen_col[c]) {
          seen_col[c] = true;
          ++uniq_col;
        }
        min_row = std::min(min_row, static_cast<int>(r));
        max_row = std::max(max_row, static_cast<int>(r));
        min_col = std::min(min_col, static_cast<int>(c));
        max_col = std::max(max_col, static_cast<int>(c));
        if (static_cast<int>(r) != (dp_lane + row_base)) {
          row_linear1 = false;
        }
        if (static_cast<int>(r) != ((dp_lane >> 1) + row_base)) {
          row_linear2 = false;
        }
        if (static_cast<int>(c) != col_base) {
          col_const = false;
        }
      }
      const char* row_tag = row_linear1 ? "linear1" : (row_linear2 ? "linear2" : "nonlinear");
      std::printf("  byte=%d rows=[%d..%d] uniq=%d %s(base=%d)",
                  b, min_row, max_row, uniq_row, row_tag, row_base);
      std::printf(" cols=[%d..%d] uniq=%d %s", min_col, max_col, uniq_col, col_const ? "const" : "vary");
      if (col_const) {
        std::printf("(val=%d)", col_base);
      }
      std::printf("\n");
    }
  };

  std::printf(
      "\nSFA UTCCP placement probe: dump (smem_row, smem_col) for dp windows at each seg base.\n");
  std::printf("Interpretation (pattern_kind=0): if UTCCP places segX rows contiguously into dp0..31, then\n");
  std::printf("  seg0 should show smem_row ~= 0..31, seg1 ~= 32..63, seg2 ~= 64..95, seg3 ~= 96..127.\n");
  for (int rank = 0; rank < 2; ++rank) {
    for (int seg = 0; seg < kProbeSegs; ++seg) {
      for (int cs = 0; cs < kProbeColSubs; ++cs) {
        for (int w = 0; w < kProbeDpWindows; ++w) {
          summarize_seg(rank, seg, cs, w);
          if (verbose_dump) {
            dump_seg(rank, seg, cs, w);
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cute;

  // Optional: run device UTCCP+TMEM dump probe to validate scale placement.
  bool run_device = false;
  int use64 = 1;
  int schedule = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--run-device") {
      run_device = true;
    } else if (arg.rfind("--use64=", 0) == 0) {
      use64 = std::atoi(arg.c_str() + 8);
    } else if (arg.rfind("--schedule=", 0) == 0) {
      schedule = std::atoi(arg.c_str() + 11);
    }
  }

  // Probe CUTLASS TMEM layouts for SM100 block-scaled UMMA (N_SM=2).
  //
  // We need exact TMEM address deltas to transliterate CUTLASS' `tmem_sf_frg` / `tmem_frg_2sm`
  // layouts into our custom kernel without depending on CUTLASS at runtime.
  //
  // Key question: for N_SM=2 block-scaled UMMA, how do encoded TMEM addresses vary with:
  // - rank (CTA rank in the 2SM MMA group)
  // - u   (N packing / subpartition index in the UMMA_2SM accumulator layout)
  // - seg (K64 block index within a K256 tile; num_MMA_K=4)
  // - t   (external "tile" index; e.g., UnrollN=2)
  using CFrag = UMMA::tmem_frg_2sm<float>;
  using SFAFrag =
      UMMA::tmem_sf_frg<cutlass::float_ue4m3_t, 16, 2, true, UMMA::TmemAllocMode::ScaleFactorDuplicated4by1>;
  using SFBFrag =
      UMMA::tmem_sf_frg<cutlass::float_ue4m3_t, 16, 2, false, UMMA::TmemAllocMode::ScaleFactorDuplicated4by1>;
  using SFBFrag2x2 =
      UMMA::tmem_sf_frg<cutlass::float_ue4m3_t, 16, 2, false, UMMA::TmemAllocMode::ScaleFactorDuplicated2by2>;

  auto dump_addr = [](const char* label, uint32_t addr) {
    const uint32_t col = addr & 0xFFFFu;
    const uint32_t dp = (addr >> 16) & 0xFFu;
    const uint32_t idx = (addr >> 24) & 0xFFu;
    const uint32_t sf_id = (addr >> 30) & 0x3u;
    std::printf("%s addr=0x%08x col=%u dp=%u idx=%u sf_id=%u\n", label, addr, col, dp, idx, sf_id);
  };

  // --------------------------------------------------------------------------
  // M_MMA=128 (UMMA_2SM 4x1 accumulator atom) sanity.
  // --------------------------------------------------------------------------
  {
    std::printf("\n=== UMMA_2SM M_MMA=128 (sanity) ===\n");
    auto c = make_tensor<CFrag>(make_shape(make_shape(Int<128>{}, Int<128>{}), Int<1>{}, Int<2>{}, Int<2>{}));
    auto sfa = make_tensor<SFAFrag>(
        make_shape(make_shape(Int<128>{}, make_shape(Int<16>{}, Int<4>{})), Int<1>{}, Int<4>{}, Int<2>{}));
    auto sfb = make_tensor<SFBFrag>(
        make_shape(make_shape(Int<128>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}));
    auto sfb2x2 = make_tensor<SFBFrag2x2>(
        make_shape(make_shape(Int<128>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}));

    const uint32_t c_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(c).layout()) & 0xFFFF);
    const uint32_t sfa_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfa).layout()) & 0xFFFF);
    const uint32_t sfb_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb).layout()) & 0xFFFF);
    const uint32_t sfb2x2_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb2x2).layout()) & 0xFFFF);
    std::printf("tmem_colspan(masked): c=%u sfa=%u sfb4x1=%u sfb2x2=%u\n", c_cols, sfa_cols, sfb_cols, sfb2x2_cols);

    auto c_layout = c.layout();
    std::printf("C base (rank,u) at (m=0,n=0)\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, 0), 0, u, rank)));
        uint32_t addr = rotr32(off, 0);
        std::printf("rank=%d u=%d off=%u ", rank, u, off);
        dump_addr("", addr);
      }
    }

    auto sfa_layout = sfa.layout();
    std::printf("SFA base (rank,seg) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int seg = 0; seg < 4; ++seg) {
        uint32_t off = static_cast<uint32_t>(sfa_layout(make_coord(make_coord(0, make_coord(0, 0)), 0, seg, rank)));
        uint32_t addr = rotr32(off, 2);
        std::printf("rank=%d seg=%d off=%u ", rank, seg, off);
        dump_addr("", addr);
      }
    }

    auto sfb_layout = sfb.layout();
    std::printf("SFB4x1 base (rank,u,seg) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg = 0; seg < 4; ++seg) {
          uint32_t off =
              static_cast<uint32_t>(sfb_layout(make_coord(make_coord(0, make_coord(0, 0)), u, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          std::printf("rank=%d u=%d seg=%d off=%u ", rank, u, seg, off);
          dump_addr("", addr);
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // M_MMA=64 (UMMA_2SM 2x2 accumulator atom) target for cta_group::2 bring-up.
  // --------------------------------------------------------------------------
  {
    std::printf("\n=== UMMA_2SM M_MMA=64 (target) ===\n");
    auto c = make_tensor<CFrag>(make_shape(make_shape(Int<64>{}, Int<128>{}), Int<1>{}, Int<2>{}, Int<2>{}));
    auto c_t = make_tensor<CFrag>(
        make_shape(make_shape(Int<64>{}, Int<128>{}), Int<1>{}, Int<2>{}, Int<2>{}, Int<2>{}));
    auto sfa = make_tensor<SFAFrag>(
        make_shape(make_shape(Int<64>{}, make_shape(Int<16>{}, Int<4>{})), Int<1>{}, Int<4>{}, Int<2>{}));
    auto sfa_t = make_tensor<SFAFrag>(make_shape(
        make_shape(Int<64>{}, make_shape(Int<16>{}, Int<4>{})), Int<1>{}, Int<4>{}, Int<2>{}, Int<2>{}));
    auto sfb = make_tensor<SFBFrag>(
        make_shape(make_shape(Int<64>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}));
    auto sfb_t = make_tensor<SFBFrag>(make_shape(
        make_shape(Int<64>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}, Int<2>{}));
    auto sfb2x2 = make_tensor<SFBFrag2x2>(
        make_shape(make_shape(Int<64>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}));
    auto sfb2x2_t = make_tensor<SFBFrag2x2>(make_shape(
        make_shape(Int<64>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}, Int<2>{}));

    const uint32_t c_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(c).layout()) & 0xFFFF);
    const uint32_t c_cols_t = static_cast<uint32_t>(cosize(recast<uint32_t>(c_t).layout()) & 0xFFFF);
    const uint32_t sfa_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfa).layout()) & 0xFFFF);
    const uint32_t sfa_cols_t = static_cast<uint32_t>(cosize(recast<uint32_t>(sfa_t).layout()) & 0xFFFF);
    const uint32_t sfb_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb).layout()) & 0xFFFF);
    const uint32_t sfb_cols_t = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb_t).layout()) & 0xFFFF);
    const uint32_t sfb2x2_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb2x2).layout()) & 0xFFFF);
    const uint32_t sfb2x2_cols_t = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb2x2_t).layout()) & 0xFFFF);
    std::printf("tmem_colspan(masked): c=%u c[t]=%u sfa=%u sfa[t]=%u sfb4x1=%u sfb4x1[t]=%u sfb2x2=%u sfb2x2[t]=%u\n",
                c_cols, c_cols_t, sfa_cols, sfa_cols_t, sfb_cols, sfb_cols_t, sfb2x2_cols, sfb2x2_cols_t);

    auto c_layout = c.layout();
    auto c_t_layout = c_t.layout();
    std::printf("C base (rank,u) at (m=0,n=0)\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, 0), 0, u, rank)));
        uint32_t addr = rotr32(off, 0);
        std::printf("rank=%d u=%d off=%u ", rank, u, off);
        dump_addr("", addr);
      }
    }
    std::printf("C (m=0) n sweep (rank=0,u=0): n=0..15\n");
    for (int n = 0; n < 16; ++n) {
      uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, n), 0, /*u=*/0, /*rank=*/0)));
      uint32_t addr = rotr32(off, 0);
      std::printf("n=%d off=%u ", n, off);
      dump_addr("", addr);
    }
    std::printf("C (m=0) n sweep (rank=0,u=1): n=0..15\n");
    for (int n = 0; n < 16; ++n) {
      uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, n), 0, /*u=*/1, /*rank=*/0)));
      uint32_t addr = rotr32(off, 0);
      std::printf("n=%d off=%u ", n, off);
      dump_addr("", addr);
    }
    std::printf("C (m=0) n sweep (rank=0,u=0): n=64..79\n");
    for (int n = 64; n < 80; ++n) {
      uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, n), 0, /*u=*/0, /*rank=*/0)));
      uint32_t addr = rotr32(off, 0);
      std::printf("n=%d off=%u ", n, off);
      dump_addr("", addr);
    }
    std::printf("C (m=0) n sweep (rank=0,u=1): n=64..79\n");
    for (int n = 64; n < 80; ++n) {
      uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, n), 0, /*u=*/1, /*rank=*/0)));
      uint32_t addr = rotr32(off, 0);
      std::printf("n=%d off=%u ", n, off);
      dump_addr("", addr);
    }
    std::printf("C (rank,u) at selected m (n=0)\n");
    for (int m : {0, 1, 32, 63}) {
      for (int rank = 0; rank < 2; ++rank) {
        for (int u = 0; u < 2; ++u) {
          uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(m, 0), 0, u, rank)));
          uint32_t addr = rotr32(off, 0);
          std::printf("m=%d rank=%d u=%d off=%u ", m, rank, u, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("C base (rank,u,t) at (m=0,n=0)\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int t = 0; t < 2; ++t) {
          uint32_t off = static_cast<uint32_t>(c_t_layout(make_coord(make_coord(0, 0), 0, u, rank, t)));
          uint32_t addr = rotr32(off, 0);
          std::printf("rank=%d u=%d t=%d off=%u ", rank, u, t, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("C (rank,u,t) at m=63 (n=0)\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int t = 0; t < 2; ++t) {
          uint32_t off = static_cast<uint32_t>(c_t_layout(make_coord(make_coord(63, 0), 0, u, rank, t)));
          uint32_t addr = rotr32(off, 0);
          std::printf("m=63 rank=%d u=%d t=%d off=%u ", rank, u, t, off);
          dump_addr("", addr);
        }
      }
    }

    auto sfa_layout = sfa.layout();
    auto sfa_t_layout = sfa_t.layout();
    std::printf("SFA base (rank,seg) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int seg = 0; seg < 4; ++seg) {
        uint32_t off = static_cast<uint32_t>(sfa_layout(make_coord(make_coord(0, make_coord(0, 0)), 0, seg, rank)));
        uint32_t addr = rotr32(off, 2);
        std::printf("rank=%d seg=%d off=%u ", rank, seg, off);
        dump_addr("", addr);
      }
    }
    // Dump per-word addresses for a single logical (m=0, nsf=0) row. Each row holds 16 bytes
    // (vs=0..15), so words are at vs={0,4,8,12}. This is useful to transliterate the exact
    // word-stride/pattern CUTLASS uses for UE4M3 scale fragments (not necessarily contiguous
    // in TMEM col-space).
    std::printf("SFA word-addrs (rank,seg) at (m=0,nsf=0) for vs={0,4,8,12}\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int seg = 0; seg < 4; ++seg) {
        std::printf("rank=%d seg=%d: ", rank, seg);
        for (int vs : {0, 4, 8, 12}) {
          uint32_t off =
              static_cast<uint32_t>(sfa_layout(make_coord(make_coord(0, make_coord(vs, 0)), 0, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          const uint32_t col = addr & 0xFFFFu;
          const uint32_t dp = (addr >> 16) & 0xFFu;
          std::printf("vs=%d(col=%u,dp=%u,addr=0x%08x) ", vs, col, dp, addr);
        }
        std::printf("\n");
      }
    }
    std::printf("SFA nsf sweep (rank=0 seg=0 m=0 vs=0): nsf=0..3\n");
    for (int nsf = 0; nsf < 4; ++nsf) {
      uint32_t off = static_cast<uint32_t>(sfa_layout(make_coord(make_coord(0, make_coord(0, nsf)), 0, /*seg=*/0, /*rank=*/0)));
      uint32_t addr = rotr32(off, 2);
      std::printf("nsf=%d off=%u ", nsf, off);
      dump_addr("", addr);
    }
    std::printf("SFA (rank,seg) at selected m ((vs,nsf)=(0,0))\n");
    for (int m : {0, 1, 32, 63}) {
      for (int rank = 0; rank < 2; ++rank) {
        for (int seg : {0, 1}) {
          uint32_t off =
              static_cast<uint32_t>(sfa_layout(make_coord(make_coord(m, make_coord(0, 0)), 0, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          std::printf("m=%d rank=%d seg=%d off=%u ", m, rank, seg, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("SFA base (rank,seg,t) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int seg = 0; seg < 4; ++seg) {
        for (int t = 0; t < 2; ++t) {
          uint32_t off = static_cast<uint32_t>(
              sfa_t_layout(make_coord(make_coord(0, make_coord(0, 0)), 0, seg, rank, t)));
          uint32_t addr = rotr32(off, 2);
          std::printf("rank=%d seg=%d t=%d off=%u ", rank, seg, t, off);
          dump_addr("", addr);
        }
      }
    }

    auto sfb_layout = sfb.layout();
    auto sfb_t_layout = sfb_t.layout();
    std::printf("SFB4x1 base (rank,u,seg) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg = 0; seg < 4; ++seg) {
          uint32_t off =
              static_cast<uint32_t>(sfb_layout(make_coord(make_coord(0, make_coord(0, 0)), u, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          std::printf("rank=%d u=%d seg=%d off=%u ", rank, u, seg, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("SFB4x1 word-addrs (rank,u,seg) at (m=0,nsf=0) for vs={0,4,8,12}\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg = 0; seg < 4; ++seg) {
          std::printf("rank=%d u=%d seg=%d: ", rank, u, seg);
          for (int vs : {0, 4, 8, 12}) {
            uint32_t off = static_cast<uint32_t>(
                sfb_layout(make_coord(make_coord(0, make_coord(vs, 0)), u, seg, rank)));
            uint32_t addr = rotr32(off, 2);
            const uint32_t col = addr & 0xFFFFu;
            const uint32_t dp = (addr >> 16) & 0xFFu;
            std::printf("vs=%d(col=%u,dp=%u,addr=0x%08x) ", vs, col, dp, addr);
          }
          std::printf("\n");
        }
      }
    }
    std::printf("SFB4x1 nsf sweep (rank=0 u=0 seg=0 m=0 vs=0): nsf=0..3\n");
    for (int nsf = 0; nsf < 4; ++nsf) {
      uint32_t off = static_cast<uint32_t>(sfb_layout(make_coord(make_coord(0, make_coord(0, nsf)), /*u=*/0, /*seg=*/0, /*rank=*/0)));
      uint32_t addr = rotr32(off, 2);
      std::printf("nsf=%d off=%u ", nsf, off);
      dump_addr("", addr);
    }
    std::printf("SFB4x1 (rank,u,seg) at m=63 ((vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg : {0, 1}) {
          uint32_t off = static_cast<uint32_t>(
              sfb_layout(make_coord(make_coord(63, make_coord(0, 0)), u, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          std::printf("m=63 rank=%d u=%d seg=%d off=%u ", rank, u, seg, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("SFB4x1 base (rank,u,seg,t) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg = 0; seg < 4; ++seg) {
          for (int t = 0; t < 2; ++t) {
            uint32_t off = static_cast<uint32_t>(
                sfb_t_layout(make_coord(make_coord(0, make_coord(0, 0)), u, seg, rank, t)));
            uint32_t addr = rotr32(off, 2);
            std::printf("rank=%d u=%d seg=%d t=%d off=%u ", rank, u, seg, t, off);
            dump_addr("", addr);
          }
        }
      }
    }

    auto sfb2x2_layout = sfb2x2.layout();
    auto sfb2x2_t_layout = sfb2x2_t.layout();
    std::printf("SFB2x2 base (rank,u,seg) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg = 0; seg < 4; ++seg) {
          uint32_t off =
              static_cast<uint32_t>(sfb2x2_layout(make_coord(make_coord(0, make_coord(0, 0)), u, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          std::printf("rank=%d u=%d seg=%d off=%u ", rank, u, seg, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("SFB2x2 (rank,u,seg) at m=63 ((vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg : {0, 1}) {
          uint32_t off = static_cast<uint32_t>(
              sfb2x2_layout(make_coord(make_coord(63, make_coord(0, 0)), u, seg, rank)));
          uint32_t addr = rotr32(off, 2);
          std::printf("m=63 rank=%d u=%d seg=%d off=%u ", rank, u, seg, off);
          dump_addr("", addr);
        }
      }
    }
    std::printf("SFB2x2 base (rank,u,seg,t) at (m=0,(vs,nsf)=(0,0))\n");
    for (int rank = 0; rank < 2; ++rank) {
      for (int u = 0; u < 2; ++u) {
        for (int seg = 0; seg < 4; ++seg) {
          for (int t = 0; t < 2; ++t) {
            uint32_t off = static_cast<uint32_t>(
                sfb2x2_t_layout(make_coord(make_coord(0, make_coord(0, 0)), u, seg, rank, t)));
            uint32_t addr = rotr32(off, 2);
            std::printf("rank=%d u=%d seg=%d t=%d off=%u ", rank, u, seg, t, off);
            dump_addr("", addr);
          }
        }
      }
    }
  }

  if (run_device) {
    run_device_probe(use64, schedule);
  }

  return 0;
}
