#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/__ptx/instructions/cp_async_bulk_tensor.h>

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

// Tunable compile-time knobs (set via nvcc -D... flags).
#ifndef NVFP4_GROUP_GEMM_PIPELINE_STAGES
#define NVFP4_GROUP_GEMM_PIPELINE_STAGES 2
#endif

#ifndef NVFP4_GROUP_GEMM_TMEM_COLUMNS
#define NVFP4_GROUP_GEMM_TMEM_COLUMNS 512
#endif

#ifndef NVFP4_GROUP_GEMM_USE_UTCCP_64X128B
#define NVFP4_GROUP_GEMM_USE_UTCCP_64X128B 0
#endif

#ifndef NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA
// Optional per-operand override: use 64x128b UTCCP only for SFA.
#define NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA 0
#endif

#ifndef NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB
// Optional per-operand override: use 64x128b UTCCP only for SFB.
#define NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB 0
#endif

#ifndef NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE
// 0: legacy bring-up (02_13, src +32, dst seg0/seg1)
// 1: contiguous pairs (01_23, src +64, dst seg0/seg2)
// 2: contiguous pairs (02_13, src +64, dst seg0/seg2)
// 3: legacy bring-up (01_23, src +32, dst seg0/seg1)
#define NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE 0
#endif

// NOTE: The earlier experimental "N=256" (single UMMA covering two N128 tiles) path was removed.
// It produced incorrect results for `mxf4nvf4.block_scale.block16` (second N128 half was zero).

#ifndef NVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF
#define NVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF 0
#endif

#ifndef NVFP4_GROUP_GEMM_MULTICAST_A
// Debug knob: when TMA multicast is enabled, allow disabling the multicast load for A only
// to isolate correctness issues (defaults to enabled).
#define NVFP4_GROUP_GEMM_MULTICAST_A 1
#endif

#ifndef NVFP4_GROUP_GEMM_MULTICAST_SFA
// Debug knob: when TMA multicast is enabled, allow disabling the multicast load for SFA only
// to isolate correctness issues (defaults to enabled).
#define NVFP4_GROUP_GEMM_MULTICAST_SFA 1
#endif

#ifndef NVFP4_GROUP_GEMM_WS_UNROLL2_MMA
// When enabled (and UnrollN==2), use warp1 lane0 to issue the u=1 MMAs while warp0 lane0
// issues scale copies + u=0 MMAs. This can reduce the serial issue bottleneck on a single lane.
#define NVFP4_GROUP_GEMM_WS_UNROLL2_MMA 0
#endif

#ifndef NVFP4_GROUP_GEMM_WS_SFB1_SEGMENT_HELPERS
// Experimental: for WS_UNROLL2_MMA (UnrollN==2, cta_group::1), use helper warps to copy the
// SFB(u=1) scale segments 2 and 3 while warp1 lane0 copies segments 0 and 1 and issues u=1 MMAs.
// This reduces warp1's serial UTCCP issue pressure without introducing extra CTA-wide barriers.
#define NVFP4_GROUP_GEMM_WS_SFB1_SEGMENT_HELPERS 0
#endif

#ifndef NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
// Spin-wait backoff (in cycles) for the warp-specialized UnrollN=2 path.
// Smaller values can reduce latency but may increase contention.
#define NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES 8
#endif

#ifndef NVFP4_GROUP_GEMM_WS_TMA_PRODUCER
// Experimental: overlap TMA issue overhead with UMMA issue overhead by having an idle warp
// (warp2 lane0) issue TMA loads for the next stage while warp0/warp1 issue MMAs for the current
// stage. This is intended to mirror CUTLASS' producer/consumer pipeline structure.
#define NVFP4_GROUP_GEMM_WS_TMA_PRODUCER 0
#endif

#ifndef NVFP4_GROUP_GEMM_WS_SPLIT_U0_SEGS
// Experimental: for WS_UNROLL2_MMA (UnrollN==2, CtaGroup==1), use an additional warp
// (warp2 lane0) to issue the u=0 MMAs for seg=1..3 while warp0 lane0 issues seg=0.
// This reduces single-thread UMMA issue pressure at the cost of extra shared-memory spin/waits.
#define NVFP4_GROUP_GEMM_WS_SPLIT_U0_SEGS 0
#endif

#ifndef NVFP4_GROUP_GEMM_WS_SEGMENT_PARALLEL
// Experimental: for cta_group::1, UnrollN==1, use 4 warps to issue:
// - UTCCP scale copies (one K64 segment per warp)
// - UMMA (one K64 segment per warp)
// This reduces single-lane issue pressure and helps utilization when TMEM limits occupancy.
#define NVFP4_GROUP_GEMM_WS_SEGMENT_PARALLEL 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA1_COMMIT_BARRIER
// Experimental: for cta_group::1, use tcgen05.commit + mbarrier wait once before epilogue TMEM loads.
// Without an explicit commit, tcgen05.mma completion can manifest as stalls on the first TMEM loads.
#define NVFP4_GROUP_GEMM_CTA1_COMMIT_BARRIER 0
#endif

#ifndef NVFP4_GROUP_GEMM_STAGE1_PREFETCH
// Experimental: for PIPELINE_STAGES==1 (cta_group::1), overlap the next tile's TMA load with
// the current tile's asynchronous UMMA execution by reusing the single shared-memory stage
// immediately after issuing the MMAs. This mirrors CUTLASS' `num_ab_stage=1` scheduling intent.
// Disabled by default because it assumes UMMA consumes shared memory at issue time (not later).
#define NVFP4_GROUP_GEMM_STAGE1_PREFETCH 0
#endif

#ifndef NVFP4_GROUP_GEMM_WARP0_ONLY_MAINLOOP
// For cta_group::1 without multicast, use a warp0-only mainloop to reduce synchronization overhead.
// This is a performance knob: warp0-only can underutilize UMMA issue if tcgen05.mma requires
// warpgroup participation. This mode is only valid for UnrollN==1; UnrollN==2 requires
// full-CTA participation for correctness.
#define NVFP4_GROUP_GEMM_WARP0_ONLY_MAINLOOP 1
#endif

#ifndef NVFP4_GROUP_GEMM_USE_CUTLASS_TMEM_SF_FRG
// Experimental: use CUTLASS' `tmem_sf_frg` mapping for cta_group::2 + UnrollN=2 TMEM scale placement.
// Enabled by default for correctness bring-up now that the mapping is hard-wired from the probe.
#define NVFP4_GROUP_GEMM_USE_CUTLASS_TMEM_SF_FRG 1
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SF_DP_BANK
// Experimental: for cta_group::2 + UnrollN=2, select the block-scaled UMMA "scale bank" by
// setting a disjoint DP window (dp_add=64) on TSFA/TSFB base pointers.
// Keep enabled by default while we validate the correct bank/id semantics.
#define NVFP4_GROUP_GEMM_CTA2_SF_DP_BANK 1
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SFA_PAIR_SEGS
// cta_group::2: when using CUTLASS `tmem_sf_frg` mapping for SFA, decide whether to "pair" segs
// (seg0+seg1, seg2+seg3) when forming the TSFA base pointers passed to UMMA.
//
// On SM100a, CUTLASS' probed SFA `tmem_sf_frg` mapping uses per-seg column offsets {0,2,4,6},
// which are not 16B-aligned for seg1/seg3 and can force UTCCP64 for scale copies.
//
// In that UTCCP64 case, UMMA must use the aligned base pointers (seg0 for {seg0,seg1} and seg2
// for {seg2,seg3}) because the misaligned half-row bases are not valid scale-fragment operands.
//
// Enable pairing by default for correctness on B200 bring-up; disable only when forcing an
// aligned per-seg copy strategy (UTCCP32) that guarantees valid per-seg bases.
#define NVFP4_GROUP_GEMM_CTA2_SFA_PAIR_SEGS 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_ACCUM_DP_BANK
// Experimental: for cta_group::2 + UnrollN=2, remap the accumulator TMEM layout to free
// column space for scale-factor storage when col>=512 addressing is illegal.
//
// 0: CUTLASS probe mapping (rank*256 + u*128 in COL)
// 1: DP-banked mapping (rank*128 in COL, u=1 in DP+128)
#define NVFP4_GROUP_GEMM_CTA2_ACCUM_DP_BANK 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SFA_SF_ID
// Experimental: override the top-2 TMEM bits (used by UMMA's a_sf_id_) for TSFA pointers.
// Default 0 preserves the baseline byte-address/id semantics.
#define NVFP4_GROUP_GEMM_CTA2_SFA_SF_ID 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SFB_SF_ID
// Experimental: override the top-2 TMEM bits (used by UMMA's b_sf_id_) for TSFB pointers.
// Default 0 preserves the baseline byte-address/id semantics.
#define NVFP4_GROUP_GEMM_CTA2_SFB_SF_ID 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE
// Experimental: CUTLASS `tmem_sf_frg` scale-factor allocation mode for SFB when
// `cta_group::2 + UnrollN=2 + USE_CUTLASS_TMEM_SF_FRG` is enabled.
//
// 0: ScaleFactorDuplicated4by1 (CUTLASS default for SFB in many kernels; colspan=64 for N_SM=2, UnrollN=2)
// 1: ScaleFactorDuplicated2by2 (colspan=32 for N_SM=2, UnrollN=2)
//
// See `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu` for the measured TMEM word-column deltas.
//
// For our cta_group::2 kernel, the logical MMA tile is M=128 (2x 64-row CTAs). CUTLASS selects
// `ScaleFactorDuplicated2by2` for M==128 in `MMA_Traits<SM100_MMA_MXF4_2x1SM_SS...>`, so default
// to the 2x2 mapping here.
#define NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE 1
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SFB_PAIR_SEGS
// cta_group::2: when using CUTLASS `tmem_sf_frg` mapping for SFB, decide whether to "pair" segs
// (seg0+seg1, seg2+seg3) when forming TSFB base pointers passed to UMMA.
//
// When using ScaleFactorDuplicated2by2 (alloc mode 1), CUTLASS' probed SFB `tmem_sf_frg` mapping
// uses an 8B segment stride (per-seg col offsets {0,2,4,6}), which forces UTCCP64 and makes the
// seg1/seg3 bases invalid UMMA operands. Pair segs so UMMA always receives aligned bases.
//
// This is ignored for alloc mode 0 (4x1), where per-seg bases are 16B-aligned.
#define NVFP4_GROUP_GEMM_CTA2_SFB_PAIR_SEGS 1
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SFB_POPULATE_U1
// Correctness-first bring-up for `cta_group::2 + UnrollN=2`:
// Populate BOTH internal UMMA_2SM SFB subpartitions (u_int=0 and u_int=1) in TMEM.
//
// CUTLASS `tmem_sf_frg` exposes u_int as an explicit dimension with a small column delta:
// - ScaleFactorDuplicated4by1: u_int delta is +2 columns
// - ScaleFactorDuplicated2by2: u_int delta is +1 column
//
// Our earlier path assumed UTCCP implicitly filled u_int=1 when writing u_int=0. That is not
// guaranteed; explicitly populating u_int=1 makes correctness robust before performance tuning.
#define NVFP4_GROUP_GEMM_CTA2_SFB_POPULATE_U1 1
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFA
// Debug knob: for cta_group::2 bring-up, skip UTCCP copies for SFA to isolate traps.
#define NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFA 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFB
// Debug knob: for cta_group::2 bring-up, skip UTCCP copies for SFB to isolate traps.
#define NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFB 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_UTCCP_RANK0_ONLY
// Debug knob: issue cta_group::2 UTCCP scale copies from cluster rank0 only (to isolate rank-specific traps).
#define NVFP4_GROUP_GEMM_CTA2_UTCCP_RANK0_ONLY 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_WAIT_AFTER_UTCCP
// Correctness-first bring-up for `cta_group::2`:
// Fence UTCCP TMEM stores before issuing UMMA so the MMA doesn't observe partially-written
// scale-factor fragments (observed as NaN accumulators on B200).
//
// Once correctness is closed, set this to 0 and re-measure; CUTLASS typically relies on UMMA
// stalling as needed, but that has not been reliable in this bring-up kernel.
#define NVFP4_GROUP_GEMM_CTA2_WAIT_AFTER_UTCCP 1
#endif

#ifndef NVFP4_GROUP_GEMM_MMA_LANE0_ALL_WARPS
// Experimental: issue UMMA from lane0 of every warp (instead of thread0 only) in the full-CTA mainloop.
// CUTLASS uses per-warp `elect_one_sync()` for tcgen05.mma; this knob lets us validate whether tcgen05.mma
// needs multi-warp participation for good performance (and/or correctness on some tiles).
#define NVFP4_GROUP_GEMM_MMA_LANE0_ALL_WARPS 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_MMA_SEGMENT_PARALLEL
// Experimental (cta_group::2 bring-up): issue UMMA in a "segment-parallel" schedule where warp `w`
// issues only K-segment `seg=w` (lane0 only). This avoids multi-warp over-issuing while still
// providing multi-warp participation for tcgen05.mma.
#define NVFP4_GROUP_GEMM_CTA2_MMA_SEGMENT_PARALLEL 0
#endif

#ifndef NVFP4_GROUP_GEMM_CTA2_MMA_ALL_THREADS
// Experimental (cta_group::2 bring-up): issue UMMA from the full CTA (all 128 threads).
//
// We already observed that `tcgen05.cp.*.cta_group::2.*.warpx4` behaves like a warpgroup collective
// and can trap when issued from a single lane. If `tcgen05.mma.cta_group::2` has similar collective
// participation requirements, issuing it from only one lane can yield undefined accumulator contents
// (NaNs/Inf) without necessarily trapping.
#define NVFP4_GROUP_GEMM_CTA2_MMA_ALL_THREADS 0
#endif

#ifndef NVFP4_GROUP_GEMM_DEBUG_STAGE
#define NVFP4_GROUP_GEMM_DEBUG_STAGE 0
#endif

#ifndef NVFP4_GROUP_GEMM_EPILOGUE_LD_X16
#define NVFP4_GROUP_GEMM_EPILOGUE_LD_X16 0
#endif

#ifndef NVFP4_GROUP_GEMM_EPILOGUE_LD_X32
#define NVFP4_GROUP_GEMM_EPILOGUE_LD_X32 0
#endif

namespace {

namespace cg = cooperative_groups;
namespace cptx = cuda::ptx;

__device__ __forceinline__ int8_t fp4_e2m1_to_intq(uint8_t nibble) {
  // Signed symmetric lookup in quantized integer space for E2M1 values.
  // The final FP value is 0.25 * intq (matches the Python LUT of {0,0.5,1,1.5,2,3,4,6,...}).
  constexpr int8_t kLut[16] = {
      0,  1,  2,  3,  //
      4,  6,  8,  12, //
      0,  -1, -2, -3, //
      -4, -6, -8, -12 //
  };
  return kLut[nibble & 0x0F];
}

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {
  return (a + b - 1) / b;
}

__host__ __device__ __forceinline__ size_t shared_bytes_for_tile(int block_m, int block_n, int kpack_tile_bytes, int scale_tile) {
  // A/B are packed bytes.
  // Scales are fp16: (block_m + block_n) * scale_tile halfs.
  // Add padding for alignment.
  return static_cast<size_t>(block_m) * static_cast<size_t>(kpack_tile_bytes) +
         static_cast<size_t>(block_n) * static_cast<size_t>(kpack_tile_bytes) +
         static_cast<size_t>(block_m + block_n) * static_cast<size_t>(scale_tile) * sizeof(half) +
         32;
}

__global__ void nvfp4_group_gemm_scalar_kernel(
    const uint64_t* __restrict__ a_ptrs,
    const uint64_t* __restrict__ b_ptrs,
    const uint64_t* __restrict__ sfa_ptrs,
    const uint64_t* __restrict__ sfb_ptrs,
    const uint64_t* __restrict__ c_ptrs,
    const int32_t* __restrict__ m_sizes,
    const int32_t* __restrict__ n_sizes,
    const int32_t* __restrict__ k_halves,
    const int32_t* __restrict__ k_scales,
    int block_m,
    int block_n,
    int kpack_tile_bytes,
    int scale_tile) {
  const int group_idx = static_cast<int>(blockIdx.z);

  const uint8_t* const a_packed = reinterpret_cast<const uint8_t*>(a_ptrs[group_idx]);
  const uint8_t* const b_packed = reinterpret_cast<const uint8_t*>(b_ptrs[group_idx]);
  const half* const sfa_half = reinterpret_cast<const half*>(sfa_ptrs[group_idx]);
  const half* const sfb_half = reinterpret_cast<const half*>(sfb_ptrs[group_idx]);
  half* const c_out = reinterpret_cast<half*>(c_ptrs[group_idx]);

	  const int m_size = m_sizes[group_idx];
	  const int n_size = n_sizes[group_idx];
  const int k_half = k_halves[group_idx];   // packed bytes == K/2
  const int k_scale = k_scales[group_idx];  // K/16

  const int local_n = static_cast<int>(threadIdx.x);
  const int local_m = static_cast<int>(threadIdx.y);
  const int global_m = static_cast<int>(blockIdx.y) * block_m + local_m;
  const int global_n = static_cast<int>(blockIdx.x) * block_n + local_n;

  if (static_cast<int>(blockIdx.y) * block_m >= m_size || static_cast<int>(blockIdx.x) * block_n >= n_size) {
    return;
  }

  const int linear_tid = local_m * block_n + local_n;
  const int threads = block_m * block_n;

  extern __shared__ unsigned char smem_raw[];
  unsigned char* smem_ptr = smem_raw;

  uint8_t* const sh_a = reinterpret_cast<uint8_t*>(smem_ptr);
  smem_ptr += static_cast<size_t>(block_m) * static_cast<size_t>(kpack_tile_bytes);

  uint8_t* const sh_b = reinterpret_cast<uint8_t*>(smem_ptr);
  smem_ptr += static_cast<size_t>(block_n) * static_cast<size_t>(kpack_tile_bytes);

  // Align for half.
  uintptr_t aligned_ptr = reinterpret_cast<uintptr_t>(smem_ptr);
  aligned_ptr = (aligned_ptr + alignof(half) - 1U) & ~(static_cast<uintptr_t>(alignof(half) - 1U));
  half* const sh_sa = reinterpret_cast<half*>(aligned_ptr);
  aligned_ptr += static_cast<size_t>(block_m) * static_cast<size_t>(scale_tile) * sizeof(half);
  half* const sh_sb = reinterpret_cast<half*>(aligned_ptr);

  float acc = 0.0f;

  for (int kp_base = 0; kp_base < k_half; kp_base += kpack_tile_bytes) {
    const int valid_kpack = ((k_half - kp_base) < kpack_tile_bytes) ? (k_half - kp_base) : kpack_tile_bytes;
    const int valid_scale = (valid_kpack + 7) >> 3;  // 8 packed bytes == 16 FP4 values

    // Load A packed bytes.
    for (int idx = linear_tid; idx < block_m * kpack_tile_bytes; idx += threads) {
      const int row = idx / kpack_tile_bytes;
      const int kk = idx - row * kpack_tile_bytes;
      const int gm = static_cast<int>(blockIdx.y) * block_m + row;
      const int gk = kp_base + kk;
      uint8_t value = 0;
      if (gm < m_size && kk < valid_kpack) {
        value = a_packed[static_cast<size_t>(gm) * static_cast<size_t>(k_half) + static_cast<size_t>(gk)];
      }
      sh_a[idx] = value;
    }

    // Load B packed bytes.
    for (int idx = linear_tid; idx < block_n * kpack_tile_bytes; idx += threads) {
      const int row = idx / kpack_tile_bytes;
      const int kk = idx - row * kpack_tile_bytes;
      const int gn = static_cast<int>(blockIdx.x) * block_n + row;
      const int gk = kp_base + kk;
      uint8_t value = 0;
      if (gn < n_size && kk < valid_kpack) {
        value = b_packed[static_cast<size_t>(gn) * static_cast<size_t>(k_half) + static_cast<size_t>(gk)];
      }
      sh_b[idx] = value;
    }

    // Load A/B scale factors (FP16, K/16).
    for (int idx = linear_tid; idx < block_m * scale_tile; idx += threads) {
      const int row = idx / scale_tile;
      const int s = idx - row * scale_tile;
      const int gm = static_cast<int>(blockIdx.y) * block_m + row;
      const int gs = (kp_base >> 3) + s;  // kp_base bytes -> (kp_base*2)/16 == kp_base/8
      half value = __float2half_rn(0.0f);
      if (gm < m_size && s < valid_scale && gs < k_scale) {
        value = sfa_half[static_cast<size_t>(gm) * static_cast<size_t>(k_scale) + static_cast<size_t>(gs)];
      }
      sh_sa[idx] = value;
    }

    for (int idx = linear_tid; idx < block_n * scale_tile; idx += threads) {
      const int row = idx / scale_tile;
      const int s = idx - row * scale_tile;
      const int gn = static_cast<int>(blockIdx.x) * block_n + row;
      const int gs = (kp_base >> 3) + s;
      half value = __float2half_rn(0.0f);
      if (gn < n_size && s < valid_scale && gs < k_scale) {
        value = sfb_half[static_cast<size_t>(gn) * static_cast<size_t>(k_scale) + static_cast<size_t>(gs)];
      }
      sh_sb[idx] = value;
    }

    __syncthreads();

    if (global_m < m_size && global_n < n_size) {
      const uint8_t* const a_row = sh_a + static_cast<size_t>(local_m) * static_cast<size_t>(kpack_tile_bytes);
      const uint8_t* const b_row = sh_b + static_cast<size_t>(local_n) * static_cast<size_t>(kpack_tile_bytes);
      const half* const sa_row = sh_sa + static_cast<size_t>(local_m) * static_cast<size_t>(scale_tile);
      const half* const sb_row = sh_sb + static_cast<size_t>(local_n) * static_cast<size_t>(scale_tile);

#pragma unroll
      for (int s = 0; s < 16; ++s) {
        if (s >= valid_scale) {
          break;
        }
        const float scale = __half2float(sa_row[s]) * __half2float(sb_row[s]);
        const int kk_base = s << 3;  // 8 packed bytes -> 16 values
        const int remaining = valid_kpack - kk_base;
        const int active = remaining > 8 ? 8 : remaining;

#pragma unroll
        for (int t = 0; t < 8; ++t) {
          if (t >= active) {
            break;
          }
          const uint8_t pa = a_row[kk_base + t];
          const uint8_t pb = b_row[kk_base + t];

          const int a0 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>(pa & 0x0F)));
          const int a1 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>((pa >> 4) & 0x0F)));
          const int b0 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>(pb & 0x0F)));
          const int b1 = static_cast<int>(fp4_e2m1_to_intq(static_cast<uint8_t>((pb >> 4) & 0x0F)));

          acc += scale * (0.25f * static_cast<float>((a0 * b0) + (a1 * b1)));
        }
      }
    }

    __syncthreads();
  }

  if (global_m < m_size && global_n < n_size) {
    c_out[static_cast<size_t>(global_m) * static_cast<size_t>(n_size) + static_cast<size_t>(global_n)] =
        __float2half_rn(acc);
  }
}

}  // namespace

void nvfp4_group_gemm_forward_grouped_cuda(
    torch::Tensor a_ptrs,
    torch::Tensor b_ptrs,
    torch::Tensor sfa_ptrs,
    torch::Tensor sfb_ptrs,
    torch::Tensor c_ptrs,
    torch::Tensor m_sizes,
    torch::Tensor n_sizes,
    torch::Tensor k_halves,
    torch::Tensor k_scales,
    int max_m_size,
    int max_n_size,
    int block_m,
    int block_n,
    int kpack_tile_bytes) {
  TORCH_CHECK(a_ptrs.is_cuda(), "a_ptrs must be CUDA tensor");
  TORCH_CHECK(b_ptrs.is_cuda(), "b_ptrs must be CUDA tensor");
  TORCH_CHECK(sfa_ptrs.is_cuda(), "sfa_ptrs must be CUDA tensor");
  TORCH_CHECK(sfb_ptrs.is_cuda(), "sfb_ptrs must be CUDA tensor");
  TORCH_CHECK(c_ptrs.is_cuda(), "c_ptrs must be CUDA tensor");
  TORCH_CHECK(m_sizes.is_cuda(), "m_sizes must be CUDA tensor");
  TORCH_CHECK(n_sizes.is_cuda(), "n_sizes must be CUDA tensor");
  TORCH_CHECK(k_halves.is_cuda(), "k_halves must be CUDA tensor");
  TORCH_CHECK(k_scales.is_cuda(), "k_scales must be CUDA tensor");

  TORCH_CHECK(a_ptrs.scalar_type() == torch::kInt64, "a_ptrs must be torch.int64");
  TORCH_CHECK(b_ptrs.scalar_type() == torch::kInt64, "b_ptrs must be torch.int64");
  TORCH_CHECK(sfa_ptrs.scalar_type() == torch::kInt64, "sfa_ptrs must be torch.int64");
  TORCH_CHECK(sfb_ptrs.scalar_type() == torch::kInt64, "sfb_ptrs must be torch.int64");
  TORCH_CHECK(c_ptrs.scalar_type() == torch::kInt64, "c_ptrs must be torch.int64");
  TORCH_CHECK(m_sizes.scalar_type() == torch::kInt, "m_sizes must be torch.int32");
  TORCH_CHECK(n_sizes.scalar_type() == torch::kInt, "n_sizes must be torch.int32");
  TORCH_CHECK(k_halves.scalar_type() == torch::kInt, "k_halves must be torch.int32");
  TORCH_CHECK(k_scales.scalar_type() == torch::kInt, "k_scales must be torch.int32");

  TORCH_CHECK(a_ptrs.dim() == 1, "a_ptrs must be 1D");
  TORCH_CHECK(b_ptrs.dim() == 1, "b_ptrs must be 1D");
  TORCH_CHECK(sfa_ptrs.dim() == 1, "sfa_ptrs must be 1D");
  TORCH_CHECK(sfb_ptrs.dim() == 1, "sfb_ptrs must be 1D");
  TORCH_CHECK(c_ptrs.dim() == 1, "c_ptrs must be 1D");
  TORCH_CHECK(m_sizes.dim() == 1, "m_sizes must be 1D");
  TORCH_CHECK(n_sizes.dim() == 1, "n_sizes must be 1D");
  TORCH_CHECK(k_halves.dim() == 1, "k_halves must be 1D");
  TORCH_CHECK(k_scales.dim() == 1, "k_scales must be 1D");

  TORCH_CHECK(a_ptrs.is_contiguous(), "a_ptrs must be contiguous");
  TORCH_CHECK(b_ptrs.is_contiguous(), "b_ptrs must be contiguous");
  TORCH_CHECK(sfa_ptrs.is_contiguous(), "sfa_ptrs must be contiguous");
  TORCH_CHECK(sfb_ptrs.is_contiguous(), "sfb_ptrs must be contiguous");
  TORCH_CHECK(c_ptrs.is_contiguous(), "c_ptrs must be contiguous");
  TORCH_CHECK(m_sizes.is_contiguous(), "m_sizes must be contiguous");
  TORCH_CHECK(n_sizes.is_contiguous(), "n_sizes must be contiguous");
  TORCH_CHECK(k_halves.is_contiguous(), "k_halves must be contiguous");
  TORCH_CHECK(k_scales.is_contiguous(), "k_scales must be contiguous");

  const int groups = static_cast<int>(m_sizes.numel());
  TORCH_CHECK(groups > 0, "grouped call requires at least one group");
  TORCH_CHECK(b_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(sfa_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(sfb_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(c_ptrs.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(n_sizes.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(k_halves.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");
  TORCH_CHECK(k_scales.numel() == m_sizes.numel(), "all grouped tensors must have matching lengths");

  TORCH_CHECK(max_m_size > 0, "max_m_size must be > 0");
  TORCH_CHECK(max_n_size > 0, "max_n_size must be > 0");
  TORCH_CHECK(block_m > 0 && block_n > 0, "block sizes must be > 0");
  TORCH_CHECK(block_m * block_n <= 1024, "block_m * block_n must be <= 1024");
  TORCH_CHECK(kpack_tile_bytes > 0, "kpack_tile_bytes must be > 0");
  TORCH_CHECK((kpack_tile_bytes % 8) == 0, "kpack_tile_bytes must be divisible by 8");

  const int scale_tile = (kpack_tile_bytes + 7) >> 3;

  const dim3 block(static_cast<unsigned int>(block_n), static_cast<unsigned int>(block_m), 1);
  const dim3 grid(static_cast<unsigned int>(ceil_div_int(max_n_size, block_n)),
                  static_cast<unsigned int>(ceil_div_int(max_m_size, block_m)),
                  static_cast<unsigned int>(groups));

  const size_t shared_bytes = shared_bytes_for_tile(block_m, block_n, kpack_tile_bytes, scale_tile);

  nvfp4_group_gemm_scalar_kernel<<<grid, block, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint64_t*>(a_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(b_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(sfa_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(sfb_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const uint64_t*>(c_ptrs.data_ptr<int64_t>()),
      m_sizes.data_ptr<int32_t>(),
      n_sizes.data_ptr<int32_t>(),
      k_halves.data_ptr<int32_t>(),
      k_scales.data_ptr<int32_t>(),
      block_m,
      block_n,
      kpack_tile_bytes,
      scale_tile);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {

// -----------------------------------------------------------------------------
// Minimal UMMA descriptor structs (transliterated from CUTLASS/CuTe headers).
// -----------------------------------------------------------------------------

namespace umma {

enum class Major : uint8_t {
  K = 0,
  MN = 1,
};

enum class LayoutType : uint8_t {
  SWIZZLE_NONE = 0,
  SWIZZLE_128B_BASE32B = 1,
  SWIZZLE_128B = 2,
  SWIZZLE_64B = 4,
  SWIZZLE_32B = 6,
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
  struct {
    uint32_t lo;
    uint32_t hi;
  };
  __host__ __device__ constexpr operator uint64_t() const noexcept { return desc_; }
};

union InstrDescriptorBlockScaled {
  uint32_t desc_ = 0;
  struct {
    uint16_t sparse_id2_ : 2,
             sparse_flag_ : 1,
             : 1,
             b_sf_id_ : 2,
             : 1,
             a_format_ : 3,
             b_format_ : 3,
             a_negate_ : 1,
             b_negate_ : 1,
             a_major_ : 1;
    uint16_t b_major_ : 1,
             n_dim_ : 6,
             scale_format_ : 1,
             m_dim_ : 5,
             a_sf_id_ : 2,
             k_size_ : 1;
  };
  __host__ __device__ constexpr operator uint32_t() const noexcept { return desc_; }
};

__device__ __forceinline__ uint64_t make_runtime_instr_desc_block_scaled(
    InstrDescriptorBlockScaled desc_i, uint32_t tmem_sfa_addr, uint32_t tmem_sfb_addr) {
  // The first 2-bits of TMEM address includes byte address.
  desc_i.a_sf_id_ = (tmem_sfa_addr & 0xC0000000u) >> 30;
  desc_i.b_sf_id_ = (tmem_sfb_addr & 0xC0000000u) >> 30;
  // Upper 32b contains the instruction descriptor. Lower 32b unused for dense MMA.
  return (static_cast<uint64_t>(static_cast<uint32_t>(desc_i)) << 32);
}

}  // namespace umma

// -----------------------------------------------------------------------------
// tcgen05 PTX helpers (transliterated from CUTLASS/CuTe headers).
// -----------------------------------------------------------------------------

namespace tcgen05 {

// CUTLASS: cute/arch/copy_sm100_tma.hpp
// Clear the peer bit so both CTAs in a cta_group::2 cluster update CTA0's mbarrier state.
constexpr uint32_t Sm100MmaPeerBitMask = 0xFEFFFFFFu;

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void* ptr) {
  // Prefer NVCC builtin to avoid PTX inline-asm operand width issues on SM100.
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ bool elect_one_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const unsigned mask = __activemask();
  const int leader = __ffs(static_cast<int>(mask)) - 1;
  return ((static_cast<int>(threadIdx.x) & 31) == leader);
#else
  return (threadIdx.x == 0);
#endif
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

__device__ __forceinline__ void tmem_alloc_cta1(uint32_t* dst_ptr_smem, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t dst_intptr = cast_smem_ptr_to_uint(dst_ptr_smem);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(dst_intptr), "r"(num_columns));
#else
  (void)dst_ptr_smem;
  (void)num_columns;
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

__device__ __forceinline__ void tmem_dealloc_cta1(uint32_t tmem_ptr, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               "tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1; \n\t"
               "}"
               :
               : "r"(tmem_ptr), "r"(num_columns));
#else
  (void)tmem_ptr;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_dealloc_cta2(uint32_t tmem_ptr, int num_columns) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               "tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1; \n\t"
               "}"
               :
               : "r"(tmem_ptr), "r"(num_columns));
#else
  (void)tmem_ptr;
  (void)num_columns;
#endif
}

__device__ __forceinline__ void tmem_relinquish_alloc_permit_cta1() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
#endif
}

__device__ __forceinline__ void tmem_relinquish_alloc_permit_cta2() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::);
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_128x128b(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta1_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void utccp_cp_cta2_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;" : : "r"(dst_tmem_addr), "l"(src_smem_desc) : "memory");
#else
  (void)src_smem_desc;
  (void)dst_tmem_addr;
#endif
}

__device__ __forceinline__ void mma_cta1_mxf4nvf4_block16(
    uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c, uint32_t accumulate, uint32_t idesc_hi, uint32_t tsfa_addr,
    uint32_t tsfb_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
               "}\n"
               :
               : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_hi), "r"(accumulate), "r"(tsfa_addr),
                 "r"(tsfb_addr));
#else
  (void)desc_a;
  (void)desc_b;
  (void)tmem_c;
  (void)accumulate;
  (void)idesc_hi;
  (void)tsfa_addr;
  (void)tsfb_addr;
#endif
}

__device__ __forceinline__ void mma_cta2_mxf4nvf4_block16(
    uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c, uint32_t accumulate, uint32_t idesc_hi, uint32_t tsfa_addr,
    uint32_t tsfb_addr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
               "}\n"
               :
               : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc_hi), "r"(accumulate), "r"(tsfa_addr),
                 "r"(tsfb_addr));
#else
  (void)desc_a;
  (void)desc_b;
  (void)tmem_c;
  (void)accumulate;
  (void)idesc_hi;
  (void)tsfa_addr;
  (void)tsfb_addr;
#endif
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_alloc(uint32_t* dst_ptr_smem, int num_columns) {
  if constexpr (CtaGroup == 2) {
    tmem_alloc_cta2(dst_ptr_smem, num_columns);
  } else {
    tmem_alloc_cta1(dst_ptr_smem, num_columns);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_dealloc(uint32_t tmem_ptr, int num_columns) {
  if constexpr (CtaGroup == 2) {
    tmem_dealloc_cta2(tmem_ptr, num_columns);
  } else {
    tmem_dealloc_cta1(tmem_ptr, num_columns);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
  if constexpr (CtaGroup == 2) {
    tmem_relinquish_alloc_permit_cta2();
  } else {
    tmem_relinquish_alloc_permit_cta1();
  }
}

template <int CtaGroup>
__device__ __forceinline__ void utccp_cp_32x128b_warpx4(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
  if constexpr (CtaGroup == 2) {
    utccp_cp_cta2_32x128b_warpx4(src_smem_desc, dst_tmem_addr);
  } else {
    utccp_cp_cta1_32x128b_warpx4(src_smem_desc, dst_tmem_addr);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void utccp_cp_64x128b_warpx2_02_13(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
  if constexpr (CtaGroup == 2) {
    utccp_cp_cta2_64x128b_warpx2_02_13(src_smem_desc, dst_tmem_addr);
  } else {
    utccp_cp_cta1_64x128b_warpx2_02_13(src_smem_desc, dst_tmem_addr);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void utccp_cp_64x128b_warpx2_01_23(uint64_t src_smem_desc, uint32_t dst_tmem_addr) {
  if constexpr (CtaGroup == 2) {
    utccp_cp_cta2_64x128b_warpx2_01_23(src_smem_desc, dst_tmem_addr);
  } else {
    utccp_cp_cta1_64x128b_warpx2_01_23(src_smem_desc, dst_tmem_addr);
  }
}

template <int CtaGroup>
__device__ __forceinline__ void mma_mxf4nvf4_block16(
    uint64_t desc_a, uint64_t desc_b, uint32_t tmem_c, uint32_t accumulate, uint32_t idesc_hi, uint32_t tsfa_addr,
    uint32_t tsfb_addr) {
  if constexpr (CtaGroup == 2) {
    mma_cta2_mxf4nvf4_block16(desc_a, desc_b, tmem_c, accumulate, idesc_hi, tsfa_addr, tsfb_addr);
  } else {
    mma_cta1_mxf4nvf4_block16(desc_a, desc_b, tmem_c, accumulate, idesc_hi, tsfa_addr, tsfb_addr);
  }
}

__device__ __forceinline__ uint32_t tmem_ld_32dp32b_x1_pack16b(uint32_t src_addr) {
  uint32_t dst0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32"
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

__device__ __forceinline__ void tmem_ld_32dp32b_x8(
    uint32_t src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5,
    uint32_t& dst6, uint32_t& dst7) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
               "{%0, %1, %2, %3, %4, %5, %6, %7},"
               "[%8];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
               : "r"(src_addr)
               : "memory");
#else
  dst0 = 0;
  dst1 = 0;
  dst2 = 0;
  dst3 = 0;
  dst4 = 0;
  dst5 = 0;
  dst6 = 0;
  dst7 = 0;
  (void)src_addr;
#endif
}

__device__ __forceinline__ void tmem_ld_32dp32b_x8_pack16b(
    uint32_t src_addr, uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3, uint32_t& dst4, uint32_t& dst5,
    uint32_t& dst6, uint32_t& dst7) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.pack::16b.b32"
               "{%0, %1, %2, %3, %4, %5, %6, %7},"
               "[%8];\n"
               : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3), "=r"(dst4), "=r"(dst5), "=r"(dst6), "=r"(dst7)
               : "r"(src_addr)
               : "memory");
#else
  dst0 = 0;
  dst1 = 0;
  dst2 = 0;
  dst3 = 0;
  dst4 = 0;
  dst5 = 0;
  dst6 = 0;
  dst7 = 0;
  (void)src_addr;
#endif
}

__device__ __forceinline__ void tmem_ld_32dp32b_x16(
    uint32_t src_addr,
    uint32_t& dst0,
    uint32_t& dst1,
    uint32_t& dst2,
    uint32_t& dst3,
    uint32_t& dst4,
    uint32_t& dst5,
    uint32_t& dst6,
    uint32_t& dst7,
    uint32_t& dst8,
    uint32_t& dst9,
    uint32_t& dst10,
    uint32_t& dst11,
    uint32_t& dst12,
    uint32_t& dst13,
    uint32_t& dst14,
    uint32_t& dst15) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32"
               "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},"
               "[%16];\n"
               : "=r"(dst0),
                 "=r"(dst1),
                 "=r"(dst2),
                 "=r"(dst3),
                 "=r"(dst4),
                 "=r"(dst5),
                 "=r"(dst6),
                 "=r"(dst7),
                 "=r"(dst8),
                 "=r"(dst9),
                 "=r"(dst10),
                 "=r"(dst11),
                 "=r"(dst12),
                 "=r"(dst13),
                 "=r"(dst14),
                 "=r"(dst15)
               : "r"(src_addr)
               : "memory");
#else
  dst0 = 0;
  dst1 = 0;
  dst2 = 0;
  dst3 = 0;
  dst4 = 0;
  dst5 = 0;
  dst6 = 0;
  dst7 = 0;
  dst8 = 0;
  dst9 = 0;
  dst10 = 0;
  dst11 = 0;
  dst12 = 0;
  dst13 = 0;
  dst14 = 0;
  dst15 = 0;
  (void)src_addr;
#endif
}

__device__ __forceinline__ void tmem_ld_32dp32b_x32(
    uint32_t src_addr,
    uint32_t& dst0,
    uint32_t& dst1,
    uint32_t& dst2,
    uint32_t& dst3,
    uint32_t& dst4,
    uint32_t& dst5,
    uint32_t& dst6,
    uint32_t& dst7,
    uint32_t& dst8,
    uint32_t& dst9,
    uint32_t& dst10,
    uint32_t& dst11,
    uint32_t& dst12,
    uint32_t& dst13,
    uint32_t& dst14,
    uint32_t& dst15,
    uint32_t& dst16,
    uint32_t& dst17,
    uint32_t& dst18,
    uint32_t& dst19,
    uint32_t& dst20,
    uint32_t& dst21,
    uint32_t& dst22,
    uint32_t& dst23,
    uint32_t& dst24,
    uint32_t& dst25,
    uint32_t& dst26,
    uint32_t& dst27,
    uint32_t& dst28,
    uint32_t& dst29,
    uint32_t& dst30,
    uint32_t& dst31) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32"
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
      "%23, %24, %25, %26, %27, %28, %29, %30, %31},"
      "[%32];\n"
      : "=r"(dst0),
        "=r"(dst1),
        "=r"(dst2),
        "=r"(dst3),
        "=r"(dst4),
        "=r"(dst5),
        "=r"(dst6),
        "=r"(dst7),
        "=r"(dst8),
        "=r"(dst9),
        "=r"(dst10),
        "=r"(dst11),
        "=r"(dst12),
        "=r"(dst13),
        "=r"(dst14),
        "=r"(dst15),
        "=r"(dst16),
        "=r"(dst17),
        "=r"(dst18),
        "=r"(dst19),
        "=r"(dst20),
        "=r"(dst21),
        "=r"(dst22),
        "=r"(dst23),
        "=r"(dst24),
        "=r"(dst25),
        "=r"(dst26),
        "=r"(dst27),
        "=r"(dst28),
        "=r"(dst29),
        "=r"(dst30),
        "=r"(dst31)
      : "r"(src_addr)
      : "memory");
#else
  dst0 = 0;
  dst1 = 0;
  dst2 = 0;
  dst3 = 0;
  dst4 = 0;
  dst5 = 0;
  dst6 = 0;
  dst7 = 0;
  dst8 = 0;
  dst9 = 0;
  dst10 = 0;
  dst11 = 0;
  dst12 = 0;
  dst13 = 0;
  dst14 = 0;
  dst15 = 0;
  dst16 = 0;
  dst17 = 0;
  dst18 = 0;
  dst19 = 0;
  dst20 = 0;
  dst21 = 0;
  dst22 = 0;
  dst23 = 0;
  dst24 = 0;
  dst25 = 0;
  dst26 = 0;
  dst27 = 0;
  dst28 = 0;
  dst29 = 0;
  dst30 = 0;
  dst31 = 0;
  (void)src_addr;
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

__device__ __forceinline__ void tmem_wait_ld_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
#endif
}

__device__ __forceinline__ void tmem_wait_st_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
#endif
}

// TMEM pointers are encoded as {col:16, dp:8, idx:8}. When forming derived addresses,
// preserve the high idx bits from the allocated base pointer rather than assuming idx=0.
__device__ __forceinline__ uint32_t tmem_addr_add(uint32_t base, uint32_t dp_add, uint32_t col_add) {
  const uint32_t base_col = base & 0x0000FFFFu;
  const uint32_t base_dp = (base >> 16) & 0x000000FFu;
  const uint32_t base_idx = base & 0xFF000000u;
  const uint32_t col = base_col + col_add;
  // dp is an 8-bit field; always wrap to avoid corrupting the idx bits on overflow.
  const uint32_t dp = (base_dp + dp_add) & 0x000000FFu;
  return base_idx | (dp << 16) | (col & 0x0000FFFFu);
}

__device__ __forceinline__ uint32_t tmem_set_top2_bits(uint32_t addr, uint32_t top2) {
  return (addr & 0x3FFFFFFFu) | ((top2 & 0x3u) << 30);
}

// CUTLASS tmem_ptr uses subword-aware pointer arithmetic:
//   addr' = addr + rotr(logical_offset, OffsetShift)
// For UE4M3/F8 (8-bit storage), OffsetShift = 2.
template <int OffsetShift>
__device__ __forceinline__ uint32_t tmem_addr_add_subword(uint32_t base, uint32_t logical_offset) {
  static_assert(OffsetShift > 0 && OffsetShift < 32, "OffsetShift must be in (0, 32)");
  const uint32_t rotated =
      (logical_offset >> OffsetShift) | (logical_offset << (32 - OffsetShift));
  return base + rotated;
}

__device__ __forceinline__ uint32_t tmem_addr_add_ue4m3(uint32_t base, uint32_t logical_offset) {
  return tmem_addr_add_subword<2>(base, logical_offset);
}

}  // namespace tcgen05

// -----------------------------------------------------------------------------
// TMA descriptor encoding (host-side). Stored as 16x int64 per descriptor.
// -----------------------------------------------------------------------------

using EncodeFn = PFN_cuTensorMapEncodeTiled_v12000;

static EncodeFn load_cuTensorMapEncodeTiled() {
  void* func_ptr = nullptr;
  cudaDriverEntryPointQueryResult query_result{};

  cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &func_ptr, 13000, cudaEnableDefault,
                                                     &query_result);
  if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess) {
    err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &func_ptr, 12000, cudaEnableDefault, &query_result);
  }
  if (err != cudaSuccess || query_result != cudaDriverEntryPointSuccess || func_ptr == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<EncodeFn>(func_ptr);
}

static void encode_2d_tensor_map_or_throw(
    CUtensorMap* out_desc,
    EncodeFn encode,
    CUtensorMapDataType dtype,
    void* base,
    uint64_t dim0,
    uint64_t dim1,
    uint64_t stride0_bytes,
    uint32_t box0,
    uint32_t box1,
    CUtensorMapSwizzle swizzle_mode,
    CUtensorMapL2promotion promotion) {
  constexpr uint32_t rank = 2;
  // CUDA tensor maps treat dimension 0 as the innermost (contiguous) dimension.
  // `globalStrides[0]` is the stride in bytes to advance dimension 1 by 1.
  uint64_t dims[rank] = {dim0, dim1};
  uint64_t stride[rank - 1] = {stride0_bytes};
  uint32_t box[rank] = {box0, box1};
  uint32_t elem_stride[rank] = {1, 1};

  constexpr auto interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  constexpr auto oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  auto fn = encode ? encode : cuTensorMapEncodeTiled;
  CUresult res = fn(out_desc, dtype, rank, base, dims, stride, box, elem_stride, interleave,
                    swizzle_mode, promotion, oob_fill);
  if (res != CUDA_SUCCESS) {
    const char* err_str = nullptr;
    const char* err_name = nullptr;
    cuGetErrorString(res, &err_str);
    cuGetErrorName(res, &err_name);
    TORCH_CHECK(false, "cuTensorMapEncodeTiled failed: ", (err_str ? err_str : "unknown"), " (",
                (err_name ? err_name : "unknown"), ", ", static_cast<int>(res), ")");
  }
}

static CUtensorMapL2promotion parse_tma_l2_promotion_from_env() {
  const char* env = std::getenv("AISP_NVFP4_GROUP_GEMM_TMA_L2_PROMOTION");
  if (env == nullptr || env[0] == '\0') {
    return CU_TENSOR_MAP_L2_PROMOTION_NONE;
  }
  const int mode = std::atoi(env);
  switch (mode) {
    case 0:
      return CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case 1:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case 2:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case 3:
      return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    default:
      TORCH_CHECK(false,
                  "AISP_NVFP4_GROUP_GEMM_TMA_L2_PROMOTION must be 0..3: "
                  "0=NONE, 1=L2_64B, 2=L2_128B, 3=L2_256B. Got ",
                  mode);
      return CU_TENSOR_MAP_L2_PROMOTION_NONE;  // Unreachable.
  }
}

static std::pair<torch::Tensor, torch::Tensor> build_ab_tma_descs_cuda(
    torch::Tensor a_ptrs_cpu, torch::Tensor b_ptrs_cpu, torch::Tensor m_sizes_cpu, torch::Tensor n_sizes_cpu,
    torch::Tensor k_halves_cpu, int64_t a_box_height_rows, int64_t b_box_height_rows) {
  TORCH_CHECK(!a_ptrs_cpu.is_cuda(), "a_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!b_ptrs_cpu.is_cuda(), "b_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!m_sizes_cpu.is_cuda(), "m_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!n_sizes_cpu.is_cuda(), "n_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!k_halves_cpu.is_cuda(), "k_halves_cpu must be CPU tensor");

  TORCH_CHECK(a_ptrs_cpu.scalar_type() == torch::kInt64, "a_ptrs_cpu must be torch.int64");
  TORCH_CHECK(b_ptrs_cpu.scalar_type() == torch::kInt64, "b_ptrs_cpu must be torch.int64");
  TORCH_CHECK(m_sizes_cpu.scalar_type() == torch::kInt, "m_sizes_cpu must be torch.int32");
  TORCH_CHECK(n_sizes_cpu.scalar_type() == torch::kInt, "n_sizes_cpu must be torch.int32");
  TORCH_CHECK(k_halves_cpu.scalar_type() == torch::kInt, "k_halves_cpu must be torch.int32");

  TORCH_CHECK(a_box_height_rows == 64 || a_box_height_rows == 128,
              "a_box_height_rows must be 64 or 128. Got a_box_height_rows=", a_box_height_rows);
  TORCH_CHECK(b_box_height_rows == 64 || b_box_height_rows == 128 || b_box_height_rows == 256,
              "b_box_height_rows must be 64, 128, or 256. Got b_box_height_rows=", b_box_height_rows);
  const uint32_t a_box_height = static_cast<uint32_t>(a_box_height_rows);
  const uint32_t b_box_height = static_cast<uint32_t>(b_box_height_rows);

  const int64_t groups = m_sizes_cpu.numel();
  TORCH_CHECK(groups > 0, "groups must be > 0");
  TORCH_CHECK(a_ptrs_cpu.numel() == groups, "a_ptrs_cpu length mismatch");
  TORCH_CHECK(b_ptrs_cpu.numel() == groups, "b_ptrs_cpu length mismatch");
  TORCH_CHECK(n_sizes_cpu.numel() == groups, "n_sizes_cpu length mismatch");
  TORCH_CHECK(k_halves_cpu.numel() == groups, "k_halves_cpu length mismatch");

  auto encode = load_cuTensorMapEncodeTiled();
  TORCH_CHECK(encode != nullptr, "cuTensorMapEncodeTiled unavailable on this runtime");
  const CUtensorMapL2promotion promotion = parse_tma_l2_promotion_from_env();

  static_assert(sizeof(CUtensorMap) == 128, "Unexpected CUtensorMap size");

  std::vector<CUtensorMap> a_descs_host(static_cast<size_t>(groups));
  std::vector<CUtensorMap> b_descs_host(static_cast<size_t>(groups));

  auto* a_ptrs = a_ptrs_cpu.data_ptr<int64_t>();
  auto* b_ptrs = b_ptrs_cpu.data_ptr<int64_t>();
  auto* ms = m_sizes_cpu.data_ptr<int32_t>();
  auto* ns = n_sizes_cpu.data_ptr<int32_t>();
  auto* ks = k_halves_cpu.data_ptr<int32_t>();

  for (int64_t i = 0; i < groups; ++i) {
    const int32_t m = ms[i];
    const int32_t n = ns[i];
    const int32_t k_bytes = ks[i];
    TORCH_CHECK(m > 0 && n > 0 && k_bytes > 0, "Invalid sizes for group ", i);
    // The caller passes explicit padded heights for TMA encoding (setup-time). Do not re-pad
    // here, otherwise the tensormap height can exceed the backing allocation (and TMA reads
    // become out-of-bounds, which is undefined).
    const int32_t m_padded = m;
    const int32_t n_padded = n;

    // Encode A/B tensor maps as packed U4 (NVFP4) rather than raw bytes. This matches CUTLASS'
    // FP4 TMA encoding and avoids extra unpack/format overhead.
    const uint64_t k_elems = static_cast<uint64_t>(k_bytes) * 2ull;  // 2 fp4 values per byte
    constexpr CUtensorMapDataType kAbDtype = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    constexpr uint32_t kBoxK = 256u;  // MMA tile K=256 elements (4-bit)

    encode_2d_tensor_map_or_throw(&a_descs_host[static_cast<size_t>(i)], encode, kAbDtype,
                                 reinterpret_cast<void*>(a_ptrs[i]),
                                 /*dim0=*/k_elems,
                                 /*dim1=*/static_cast<uint64_t>(m_padded),
                                 /*stride0_bytes=*/static_cast<uint64_t>(k_bytes),
                                 /*box0=*/kBoxK,
                                 /*box1=*/a_box_height,
                                 CU_TENSOR_MAP_SWIZZLE_128B,
                                 promotion);

    encode_2d_tensor_map_or_throw(&b_descs_host[static_cast<size_t>(i)], encode, kAbDtype,
                                 reinterpret_cast<void*>(b_ptrs[i]),
                                 /*dim0=*/k_elems,
                                 /*dim1=*/static_cast<uint64_t>(n_padded),
                                 /*stride0_bytes=*/static_cast<uint64_t>(k_bytes),
                                 /*box0=*/kBoxK,
                                 /*box1=*/b_box_height,
                                 CU_TENSOR_MAP_SWIZZLE_128B,
                                 promotion);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor a_descs = torch::empty({groups, 16}, opts);
  torch::Tensor b_descs = torch::empty({groups, 16}, opts);

  cudaError_t err_a =
      cudaMemcpy(a_descs.data_ptr(), a_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_a == cudaSuccess, "cudaMemcpy a_descs failed: ", cudaGetErrorString(err_a));
  cudaError_t err_b =
      cudaMemcpy(b_descs.data_ptr(), b_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_b == cudaSuccess, "cudaMemcpy b_descs failed: ", cudaGetErrorString(err_b));

  return {a_descs, b_descs};
}

static std::pair<torch::Tensor, torch::Tensor> build_scale_tma_descs_cuda(
    torch::Tensor sfa_ptrs_cpu, torch::Tensor sfb_ptrs_cpu, torch::Tensor m_sizes_cpu, torch::Tensor n_sizes_cpu,
    torch::Tensor k_halves_cpu, int64_t sfa_box_height_rows, int64_t sfb_box_height_rows) {
  TORCH_CHECK(!sfa_ptrs_cpu.is_cuda(), "sfa_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!sfb_ptrs_cpu.is_cuda(), "sfb_ptrs_cpu must be CPU tensor");
  TORCH_CHECK(!m_sizes_cpu.is_cuda(), "m_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!n_sizes_cpu.is_cuda(), "n_sizes_cpu must be CPU tensor");
  TORCH_CHECK(!k_halves_cpu.is_cuda(), "k_halves_cpu must be CPU tensor");

  TORCH_CHECK(sfa_ptrs_cpu.scalar_type() == torch::kInt64, "sfa_ptrs_cpu must be torch.int64");
  TORCH_CHECK(sfb_ptrs_cpu.scalar_type() == torch::kInt64, "sfb_ptrs_cpu must be torch.int64");
  TORCH_CHECK(m_sizes_cpu.scalar_type() == torch::kInt, "m_sizes_cpu must be torch.int32");
  TORCH_CHECK(n_sizes_cpu.scalar_type() == torch::kInt, "n_sizes_cpu must be torch.int32");
  TORCH_CHECK(k_halves_cpu.scalar_type() == torch::kInt, "k_halves_cpu must be torch.int32");

  TORCH_CHECK(sfa_box_height_rows == 64 || sfa_box_height_rows == 128,
              "sfa_box_height_rows must be 64 or 128. Got sfa_box_height_rows=", sfa_box_height_rows);
  const uint32_t sfa_box_height = static_cast<uint32_t>(sfa_box_height_rows);

  TORCH_CHECK(sfb_box_height_rows == 64 || sfb_box_height_rows == 128 || sfb_box_height_rows == 256,
              "sfb_box_height_rows must be 64, 128, or 256. Got sfb_box_height_rows=", sfb_box_height_rows);
  const uint32_t sfb_box_height = static_cast<uint32_t>(sfb_box_height_rows);

  const int64_t groups = m_sizes_cpu.numel();
  TORCH_CHECK(groups > 0, "groups must be > 0");
  TORCH_CHECK(sfa_ptrs_cpu.numel() == groups, "sfa_ptrs_cpu length mismatch");
  TORCH_CHECK(sfb_ptrs_cpu.numel() == groups, "sfb_ptrs_cpu length mismatch");
  TORCH_CHECK(n_sizes_cpu.numel() == groups, "n_sizes_cpu length mismatch");
  TORCH_CHECK(k_halves_cpu.numel() == groups, "k_halves_cpu length mismatch");

  auto encode = load_cuTensorMapEncodeTiled();
  TORCH_CHECK(encode != nullptr, "cuTensorMapEncodeTiled unavailable on this runtime");
  const CUtensorMapL2promotion promotion = parse_tma_l2_promotion_from_env();

  static_assert(sizeof(CUtensorMap) == 128, "Unexpected CUtensorMap size");

  std::vector<CUtensorMap> sfa_descs_host(static_cast<size_t>(groups));
  std::vector<CUtensorMap> sfb_descs_host(static_cast<size_t>(groups));

  auto* sfa_ptrs = sfa_ptrs_cpu.data_ptr<int64_t>();
  auto* sfb_ptrs = sfb_ptrs_cpu.data_ptr<int64_t>();
  auto* ms = m_sizes_cpu.data_ptr<int32_t>();
  auto* ns = n_sizes_cpu.data_ptr<int32_t>();
  auto* ks = k_halves_cpu.data_ptr<int32_t>();

  for (int64_t i = 0; i < groups; ++i) {
    const int32_t m = ms[i];
    const int32_t n = ns[i];
    const int32_t k_bytes = ks[i];
    TORCH_CHECK(m > 0 && n > 0 && k_bytes > 0, "Invalid sizes for group ", i);

    constexpr int kTileBytes = 128;
    constexpr int sfaRowsPerTile = 128;
    constexpr int sfbRowsPerTile = 128;
    constexpr int sfRowBytes = 16;

    const int32_t k_tiles = ceil_div_int(k_bytes, kTileBytes);
    const int32_t m_tiles = ceil_div_int(m, 128);
    const int32_t n_tiles = ceil_div_int(n, 128);

    const int64_t sfa_height = static_cast<int64_t>(m_tiles) * static_cast<int64_t>(k_tiles) * sfaRowsPerTile;
    const int64_t sfb_height = static_cast<int64_t>(n_tiles) * static_cast<int64_t>(k_tiles) * sfbRowsPerTile;

    encode_2d_tensor_map_or_throw(&sfa_descs_host[static_cast<size_t>(i)], encode, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                 reinterpret_cast<void*>(sfa_ptrs[i]),
                                 /*dim0=*/static_cast<uint64_t>(sfRowBytes),
                                 /*dim1=*/static_cast<uint64_t>(sfa_height),
                                 /*stride0_bytes=*/static_cast<uint64_t>(sfRowBytes),
                                 /*box0=*/sfRowBytes,
                                 /*box1=*/sfa_box_height,
                                 CU_TENSOR_MAP_SWIZZLE_NONE,
                                 promotion);

    encode_2d_tensor_map_or_throw(&sfb_descs_host[static_cast<size_t>(i)], encode, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                 reinterpret_cast<void*>(sfb_ptrs[i]),
                                 /*dim0=*/static_cast<uint64_t>(sfRowBytes),
                                 /*dim1=*/static_cast<uint64_t>(sfb_height),
                                 /*stride0_bytes=*/static_cast<uint64_t>(sfRowBytes),
                                 /*box0=*/sfRowBytes,
                                 /*box1=*/sfb_box_height,
                                 CU_TENSOR_MAP_SWIZZLE_NONE,
                                 promotion);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor sfa_descs = torch::empty({groups, 16}, opts);
  torch::Tensor sfb_descs = torch::empty({groups, 16}, opts);

  cudaError_t err_sfa =
      cudaMemcpy(sfa_descs.data_ptr(), sfa_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_sfa == cudaSuccess, "cudaMemcpy sfa_descs failed: ", cudaGetErrorString(err_sfa));
  cudaError_t err_sfb =
      cudaMemcpy(sfb_descs.data_ptr(), sfb_descs_host.data(), static_cast<size_t>(groups) * sizeof(CUtensorMap),
                 cudaMemcpyHostToDevice);
  TORCH_CHECK(err_sfb == cudaSuccess, "cudaMemcpy sfb_descs failed: ", cudaGetErrorString(err_sfb));

  return {sfa_descs, sfb_descs};
}

// -----------------------------------------------------------------------------
// tcgen05 kernel (correctness-first; single-stage). CtaGroup is 1 or 2.
// -----------------------------------------------------------------------------

namespace cuda_device = cuda::device::experimental;

template <int CtaGroup, int UnrollN, int CtaTileM, bool EnableTmaMulticast>
__global__ void nvfp4_group_gemm_tcgen05_kernel(
    const uint64_t* __restrict__ a_ptrs,
    const uint64_t* __restrict__ b_ptrs,
    const uint64_t* __restrict__ sfa_ptrs,  // points to packed [m_tiles,k_tiles,128,16] uint8 scales
    const uint64_t* __restrict__ sfb_ptrs,  // points to packed [n_tiles,k_tiles,128,16] uint8 scales
    const uint64_t* __restrict__ c_ptrs,
    const int32_t* __restrict__ m_sizes,
    const int32_t* __restrict__ n_sizes,
    const int32_t* __restrict__ k_halves,
    const int32_t* __restrict__ k_scales,
    const uint64_t* __restrict__ a_descs_u64,  // [groups,16]
    const uint64_t* __restrict__ b_descs_u64,  // [groups,16]
    const uint64_t* __restrict__ sfa_descs_u64,  // [groups,16]
    const uint64_t* __restrict__ sfb_descs_u64,  // [groups,16]
    // Optional packed-CTA mapping (GPU MODE-style): grid=(1,1,total_ctas), blockIdx.z selects a CTA.
    // When these are null, we use the legacy max-based grid mapping: grid=(n_tiles, m_tiles, groups).
    const int32_t* __restrict__ cta_group_idx_map,  // [total_ctas] or nullptr
    const int32_t* __restrict__ cta_tile_m_map,     // [total_ctas] or nullptr
    const int32_t* __restrict__ cta_tile_n_map,     // [total_ctas] or nullptr
    int cta2_desc_a_row_offset_rows,
    int cta2_desc_b_row_offset_rows,
    int cta2_desc_sfa_row_offset_rows,
    int cta2_epilogue_row_base_rows,
    int cta2_epilogue_addr_mode,
    int cta2_sfb_slot_mode,
    int cta2_tmem_c_word_offset,
    int cta2_tmem_sf_word_offset,
    int cta2_tmem_sf_rank_word_offset,
    int cta2_tsfa_word_offset,
    int cta2_tsfb_word_offset,
    int cta2_sfa_sf_id,
    int cta2_sfb_sf_id,
    int debug_tmem_dump,
    int debug_tmem_only_rank,
    int debug_tmem_idx_add,
    int cta2_partition_b,
    int debug_print_ptrs,
    int cta2_idesc_m_dim_override,
    int cta2_idesc_n_dim_override,
    int cluster_dim_x
) {
  // cta_group::2 note:
  // For tcgen05 cta_group::2, CUTLASS' `tmem_frg_2sm<float>` shows two useful regimes:
  // - M_MMA=128 per CTA: UnrollN=2 consumes the full TMEM DP/COL address space, leaving no
  //   non-overlapping space for scale factors in the same TMEM allocation.
  // - M_MMA=64 per CTA : Accumulators occupy DP 0..63, leaving DP 64..127 available for scale
  //   factors (dp-banked) while staying within COL 0..511.
  //
  // We therefore use cta_group::2 with CtaTileM=64 (per-CTA M=64, cluster tile M=128).
  // See `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu` for the CUTLASS address probes.
  // CtaTileM controls the per-CTA M dimension (and the per-CTA A tile height). For cta_group::1
  // we intentionally keep the packed CTA map's tile_m indexing in 128-row units
  // (CLUSTER_TILE_M=128) and use CtaTileM=64 only for tail tiles where remaining rows <= 64.
  // This avoids runtime-masked UMMA descriptor tweaks (which can trigger illegal instruction
  // faults on SM100 for block-scaled UMMA).
  constexpr int CTA_TILE_M = CtaTileM;
  constexpr int CLUSTER_TILE_M = (CtaGroup == 2) ? 128 : 128;
  constexpr int TILE_M = CTA_TILE_M;
  constexpr int TILE_N = 128;
  constexpr int TILE_N_MMA = TILE_N;
  constexpr int K_TILE_BYTES = 128;   // 256 FP4 elems
  constexpr int K_SEG_BYTES = 32;     // 64 FP4 elems
  // TMEM allocation: use the full 512-column TMEM slice, matching the GPU MODE reference.
  // This avoids allocator interleaving behavior for smaller slices and keeps the layout consistent
  // across bring-up and tuned variants.
  constexpr int TMEM_COLUMNS = NVFP4_GROUP_GEMM_TMEM_COLUMNS;
  static_assert(TMEM_COLUMNS >= 32 && TMEM_COLUMNS <= 512 && ((TMEM_COLUMNS & (TMEM_COLUMNS - 1)) == 0),
                "NVFP4_GROUP_GEMM_TMEM_COLUMNS must be a power of 2 in [32, 512] (tcgen05.alloc requirement).");
  constexpr int SF_BYTES_PER_ROW = 16;  // (mm4, kk4) packed as 4x4 bytes
  // Scale tiles are stored in the CUTLASS SM100 "blockscaled" packed layout:
  //   [seg(0..3) * 32 + mm32, mm4*4 + kk4] -> [128, 16]
  // so the scale tile height is always 128 rows regardless of cta_group. Any rank-specific
  // selection for UMMA_2SM is handled via the TMEM fragment mapping, not by changing the
  // global row coordinate / shared tile height.
  constexpr int SFA_ROWS = 128;
  constexpr int SFB_ROWS = 128;
  constexpr int SFA_TILE_BYTES = SFA_ROWS * SF_BYTES_PER_ROW;
  constexpr int SFB_TILE_BYTES = SFB_ROWS * SF_BYTES_PER_ROW;
  // Scale-factor layout in TMEM:
  // CUTLASS uses `tmem_sf_frg<ue4m3, SFVecSize=16, N_SM=2, ...>` for cta_group::2 block-scaled UMMA.
  // The important nuance is that K64 "segment" selection is encoded via the TMEM scale-id (sf_id)
  // bits (addr[31:30]) using `tmem_ptr` subword addressing (rotr(offset, 2)), not by advancing
  // the TMEM `col` field for each seg.
  //
  // Our local probe (`labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu`) reports the encoded element
  // offsets and corresponding TMEM word-column deltas for the N_SM=2 case:
  //   - SFA (no external t): col(rank, seg) = rank*8 + seg*2
  //   - SFB (4x1 alloc): col(rank, u_int, seg, t) = rank*16 + seg*4 + u_int*2 + t*32
  //   - SFB (2x2 alloc): col(rank, u_int, seg, t) = rank*8 + seg*2 + u_int*1 + t*16
  // The final TMEM address for ue4m3 is: addr = base + rotr(off, 2).
  //
  // Legacy (non-CUTLASS) TMEM mapping uses a 4-column stride per K64 segment.
  constexpr uint32_t SF_COLS_PER_KBLOCK_PER_MN = 4u;
  constexpr uint32_t SF_COLS_PER_TILE_PER_MN = SF_COLS_PER_KBLOCK_PER_MN * 4u;  // 4 K64 blocks.
  // CUTLASS `tmem_sf_frg` (UMMA_2SM M_MMA=64) uses a 2-column stride per K64 segment.
  constexpr uint32_t SF_COLS_PER_KBLOCK_PER_MN_CUTLASS = 2u;
  constexpr uint32_t SF_COLS_PER_TILE_PER_MN_CUTLASS = SF_COLS_PER_KBLOCK_PER_MN_CUTLASS * 4u;
  // Bring-up knob:
  //   0 = full (TMA + scales + UTCCP + MMA + epilogue)
  //   2 = TMA-only sanity (alloc TMEM, load A/B once, then dealloc + return)
  //   3 = TMA + scales + UTCCP (no MMA/epilogue; one K-tile, then dealloc + return)
  //   4 = TMA + scales + UTCCP + MMA (no epilogue; one K-tile, then dealloc + return)
  constexpr int DEBUG_STAGE = NVFP4_GROUP_GEMM_DEBUG_STAGE;
  // Shared-memory stages: 2-stage double-buffered pipeline across K tiles.
  constexpr int PIPELINE_STAGES = NVFP4_GROUP_GEMM_PIPELINE_STAGES;
  static_assert(PIPELINE_STAGES >= 1 && PIPELINE_STAGES <= 4, "PIPELINE_STAGES must be 1..4");
  // Warp-specialized issue path for UnrollN=2:
  // - warp0 lane0: TMA + UTCCP scales + MMAs for u=0
  // - warp1 lane0: MMAs for u=1 (after warp0 has copied scales)
  //
  // NOTE: disabled for DEBUG_STAGE bring-up modes to avoid deadlocks when loops break early.
  constexpr bool WS_UNROLL2_MMA =
      (CtaGroup == 1) && (UnrollN == 2) && (NVFP4_GROUP_GEMM_WS_UNROLL2_MMA != 0) && (DEBUG_STAGE == 0) &&
      (!EnableTmaMulticast);
  constexpr bool WS_SPLIT_U0_SEGS = WS_UNROLL2_MMA && (NVFP4_GROUP_GEMM_WS_SPLIT_U0_SEGS != 0);
  constexpr bool WS_SEGMENT_PARALLEL =
      (CtaGroup == 1) && (UnrollN == 1) && (NVFP4_GROUP_GEMM_WS_SEGMENT_PARALLEL != 0) && (DEBUG_STAGE == 0) &&
      (!EnableTmaMulticast) && (NVFP4_GROUP_GEMM_USE_UTCCP_64X128B == 0) &&
      (NVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF == 0);
  constexpr bool CTA2_MMA_ALL_THREADS =
      (CtaGroup == 2) && (NVFP4_GROUP_GEMM_CTA2_MMA_ALL_THREADS != 0) && (DEBUG_STAGE == 0);
  constexpr bool CTA2_MMA_SEGMENT_PARALLEL =
      (CtaGroup == 2) && (NVFP4_GROUP_GEMM_CTA2_MMA_SEGMENT_PARALLEL != 0) && (DEBUG_STAGE == 0);
  constexpr bool WS_TMA_PRODUCER = (CtaGroup == 1) && (NVFP4_GROUP_GEMM_WS_TMA_PRODUCER != 0) && (DEBUG_STAGE == 0) &&
                                   (PIPELINE_STAGES > 1) && (!EnableTmaMulticast) && (!WS_SPLIT_U0_SEGS);
  constexpr bool WS_SFB1_SEGMENT_HELPERS =
      WS_UNROLL2_MMA && (NVFP4_GROUP_GEMM_WS_SFB1_SEGMENT_HELPERS != 0) && (!WS_SPLIT_U0_SEGS) &&
      (!WS_TMA_PRODUCER) && (NVFP4_GROUP_GEMM_USE_UTCCP_64X128B == 0) &&
      (NVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF == 0);
  constexpr bool STAGE1_PREFETCH = (PIPELINE_STAGES == 1) && (NVFP4_GROUP_GEMM_STAGE1_PREFETCH != 0) &&
                                   (!WS_UNROLL2_MMA) && (!WS_SPLIT_U0_SEGS) &&
                                   (!EnableTmaMulticast);
  constexpr bool CTA1_COMMIT_BARRIER =
      (CtaGroup == 1) && (NVFP4_GROUP_GEMM_CTA1_COMMIT_BARRIER != 0) && (DEBUG_STAGE == 0);

  static_assert(CtaGroup == 1 || CtaGroup == 2, "CtaGroup must be 1 or 2");
  static_assert(UnrollN == 1 || UnrollN == 2, "UnrollN must be 1 or 2");
  static_assert((CtaGroup != 2) || (CtaTileM == 64),
                "cta_group::2 requires CtaTileM=64 (M_MMA=64 per CTA, cluster tile M=128).");
  static_assert((CtaGroup != 1) || (CtaTileM == 128),
                "cta_group::1 path expects CtaTileM=128 (M tile).");
  if constexpr (CtaGroup == 2) {
    // cta_group::2 supports UnrollN in {1,2}; UnrollN=2 remains experimental and is validated
    // via runtime correctness checks in the microbench/harness flow.
  }

  // Optional cluster-mode optimization: multicast A + SFA across CTAs that share (tile_m, k_tile).
  // This reduces redundant L2 traffic for A/SFA across the N tiles of the same M tile.
  //
  // IMPORTANT: Keep multicast a compile-time specialization to avoid polluting the non-multicast
  // kernel with extra register pressure (we're latency-bound and very sensitive to occupancy).
  static_assert(!EnableTmaMulticast || (CtaGroup == 1), "TMA multicast is supported only for cta_group::1.");
  if constexpr (EnableTmaMulticast) {
    // Host-side validation should prevent this, but keep a defensive runtime check.
    if (cluster_dim_x <= 1) {
      return;
    }
  }
  constexpr bool use_tma_multicast = EnableTmaMulticast;
  uint16_t tma_multicast_mask = 0;
  if constexpr (EnableTmaMulticast) {
    tma_multicast_mask = static_cast<uint16_t>((1u << static_cast<unsigned>(cluster_dim_x)) - 1u);
  }

  int group_idx = 0;
  const int cluster_rank =
      (CtaGroup == 2 || EnableTmaMulticast) ? static_cast<int>(tcgen05::block_rank_in_cluster()) : 0;
  const int cluster_rank_b = cluster_rank;
  cg::cluster_group cluster = cg::this_cluster();
  int tile_n = 0;
  int tile_m = 0;
  if (cta_group_idx_map != nullptr && cta_tile_m_map != nullptr && cta_tile_n_map != nullptr) {
    // Packed-CTA mode: the host launches exactly the required CTAs per group (no early-return CTAs).
    // Non-cluster packed launch uses grid.z as the linear CTA dimension. Cluster launch needs
    // grid.x for clustering, so it passes the linear CTA index via blockIdx.x.
    const int cta_linear = (cluster_dim_x > 1) ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.z);
    group_idx = cta_group_idx_map[cta_linear];
    tile_m = cta_tile_m_map[cta_linear];
    tile_n = cta_tile_n_map[cta_linear];
  } else {
    // Legacy max-based grid: (tile_n, tile_m, group_idx).
    group_idx = static_cast<int>(blockIdx.z);
    // For cluster-mode launches, `blockIdx.x` indexes CTAs along N. Each CTA covers `UnrollN`
    // adjacent N tiles, so convert CTA index -> tile_n start.
    tile_n = (CtaGroup == 2) ? ((static_cast<int>(blockIdx.x) >> 1) * UnrollN)
                             : (static_cast<int>(blockIdx.x) * UnrollN);
    tile_m = static_cast<int>(blockIdx.y);
  }
  // cta_group::2 bring-up: legacy SFB-slot mapping knob (kept for future UTCCP schedule experiments).
  (void)cta2_sfb_slot_mode;

  const int m_size = m_sizes[group_idx];
  const int n_size = n_sizes[group_idx];
  const int k_bytes_total = k_halves[group_idx];
  const int k_tiles_total = ceil_div_int(k_bytes_total, K_TILE_BYTES);
  const int n_tiles_group = ceil_div_int(n_size, TILE_N);
  const bool ws_u1_active = WS_UNROLL2_MMA && ((tile_n + 1) < n_tiles_group);
  const int cta2_partition_b_mode =
      (CtaGroup == 2 && UnrollN == 2) ? 0 : cta2_partition_b;

	  const int m_offset_cluster = tile_m * CLUSTER_TILE_M;
	  const int n_offset = tile_n * TILE_N;

  // Cluster-wide bounds check: must be uniform across CTAs in the cluster to avoid divergence
  // across required cta_group::2 synchronization points.
  if (m_offset_cluster >= m_size || n_offset >= n_size) {
    return;
  }

	  const uint8_t* const a_gmem = reinterpret_cast<const uint8_t*>(a_ptrs[group_idx]);
  const uint8_t* const b_gmem = reinterpret_cast<const uint8_t*>(b_ptrs[group_idx]);
  half* const c_out = reinterpret_cast<half*>(c_ptrs[group_idx]);
  // Legacy bring-up parameters (kept for host-side experimentation). The current 256x128 2SM path
  // intentionally ignores descriptor-row remaps in favor of direct per-rank tile offsets.
  (void)cta2_desc_a_row_offset_rows;
  (void)cta2_desc_b_row_offset_rows;
  (void)cta2_desc_sfa_row_offset_rows;
  (void)cta2_epilogue_row_base_rows;
  (void)cta2_epilogue_addr_mode;
  (void)debug_tmem_dump;
  (void)debug_tmem_only_rank;
  (void)debug_tmem_idx_add;
  (void)debug_print_ptrs;
  (void)cta2_idesc_m_dim_override;
  (void)cta2_idesc_n_dim_override;

  // Shared storage (double-buffered across K tiles).
  __shared__ alignas(128) uint8_t sA[PIPELINE_STAGES][CTA_TILE_M][K_TILE_BYTES];
  __shared__ alignas(128) uint8_t sB[PIPELINE_STAGES][UnrollN][TILE_N][K_TILE_BYTES];
  __shared__ alignas(16) uint8_t sSFA[PIPELINE_STAGES][SFA_ROWS][SF_BYTES_PER_ROW];
  __shared__ alignas(16) uint8_t sSFB[PIPELINE_STAGES][UnrollN][SFB_ROWS][SF_BYTES_PER_ROW];
  // UMMA completion barrier (cta_group::2 only). tcgen05.mma is asynchronous; tcgen05.commit +
  // mbarrier wait ensures the accumulator tile is fully materialized in TMEM before epilogue loads.
  __shared__ alignas(8) uint64_t umma_done_barrier;

  // Barrier for TMA loads (per-stage). We only use the barrier from thread0 and then
  // synchronize the full CTA with __syncthreads() when data is needed.
  using block_barrier = cuda::barrier<cuda::thread_scope_block>;
  struct alignas(alignof(block_barrier)) BarrierStorage {
    unsigned char bytes[sizeof(block_barrier)];
  };
  __shared__ BarrierStorage bar_storage[PIPELINE_STAGES];
  block_barrier* bars[PIPELINE_STAGES];
#pragma unroll
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    bars[stage] = reinterpret_cast<block_barrier*>(bar_storage[stage].bytes);
  }

  if (threadIdx.x == 0) {
    // Barrier expected-arrival policy:
    // - non-multicast: only thread0 arrives+writes tx count (fast path)
    // - multicast: follow the cluster-safe pattern where every CTA thread joins the barrier
    //   generation before rank0 issues the multicast transaction.
    const int expected_arrivals = EnableTmaMulticast ? static_cast<int>(blockDim.x) : 1;
#pragma unroll
    for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
      init(bars[stage], expected_arrivals);
    }
    cuda_device::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  if constexpr (CtaGroup == 2 || CTA1_COMMIT_BARRIER) {
    // UMMA completion barrier:
    // - cta_group::1: per-CTA barrier is sufficient.
    // - cta_group::2: both CTAs must update the *same* barrier, otherwise each barrier only sees
    //   one arrival and `try_wait` deadlocks. CUTLASS clears the peer bit so both CTAs target
    //   CTA0's barrier (Sm100MmaPeerBitMask).
    //
    // mbarrier parity starts at 1 after init; the first completion flips parity to 0.
    if (threadIdx.x == 0) {
      uint32_t bar_addr = tcgen05::cast_smem_ptr_to_uint(&umma_done_barrier);
      if constexpr (CtaGroup == 2) {
        bar_addr &= tcgen05::Sm100MmaPeerBitMask;
        if (cluster_rank == 0) {
          asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                       :
                       : "r"(bar_addr), "r"(2)
                       : "memory");
          asm volatile("fence.mbarrier_init.release.cluster;\n" : : : "memory");
        }
      } else {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                     :
                     : "r"(bar_addr), "r"(1)
                     : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;\n" : : : "memory");
      }
    }
    __syncthreads();
    if constexpr (CtaGroup == 2) {
      cg::this_cluster().sync();
    }
  }

  // TMA expects the tensor map descriptor in global/const memory (not shared). Our descriptors are
  // stored in CUDA global memory as a packed [groups,16] int64 tensor (16 * 8 bytes = 128 bytes).
  const CUtensorMap* const a_descs = reinterpret_cast<const CUtensorMap*>(a_descs_u64);
  const CUtensorMap* const b_descs = reinterpret_cast<const CUtensorMap*>(b_descs_u64);
  const CUtensorMap* const sfa_descs = reinterpret_cast<const CUtensorMap*>(sfa_descs_u64);
  const CUtensorMap* const sfb_descs = reinterpret_cast<const CUtensorMap*>(sfb_descs_u64);
  const CUtensorMap* const a_desc = a_descs + group_idx;
  const CUtensorMap* const b_desc = b_descs + group_idx;
  const CUtensorMap* const sfa_desc = sfa_descs + group_idx;
  const CUtensorMap* const sfb_desc = sfb_descs + group_idx;

  // For cta_group::2, each CTA owns a full 128-row fragment within the 256-row cluster tile.
  const int m_offset = m_offset_cluster + ((CtaGroup == 2) ? (cluster_rank * CTA_TILE_M) : 0);
  const int sfa_tile_m = (CtaGroup == 2) ? (tile_m * 2 + cluster_rank) : tile_m;

  // Prepare smem descriptors for A/B tiles (per pipeline stage).
  umma::SmemDescriptor desc_a_base[PIPELINE_STAGES]{};
  umma::SmemDescriptor desc_b_base[PIPELINE_STAGES][UnrollN]{};
  // Prepare smem descriptors for scale tiles (Layout_K_INTER_Atom<uint8_t> tiled to 128x16).
  umma::SmemDescriptor desc_sfa[PIPELINE_STAGES]{};
  umma::SmemDescriptor desc_sfb[PIPELINE_STAGES][UnrollN]{};

  // For SWIZZLE_128B major-K UMMA descriptors, advancing by one logical MN row is
  // not a simple linear +row*row_bytes adjustment due to the position-dependent
  // swizzle. CUTLASS's canonical UMMA_K layout (see make_umma_desc<Major::K>)
  // implies the following u128-addressing strides for SW128:
  //   - within an 8-row swizzle atom: +8 u128 per row
  //   - across 8-row blocks:        +stride_u128 per block (SBO)
  // For our use case we only need the base pointer shift for MN offsets at K=0.
  auto sw128_major_k_row_offset_bytes = [](int row_offset_rows, uint32_t stride_u128) -> uint32_t {
    // row_offset_rows is in units of uint8 rows (MN index). The address arithmetic is in units of
    // uint128 (4 LSB stripped). Apply the same Swizzle<3,4,3> transform as CUTLASS uses for SW128.
    const uint32_t within = static_cast<uint32_t>(row_offset_rows & 7);
    const uint32_t block = static_cast<uint32_t>(row_offset_rows >> 3);
    const uint32_t logical_u128 = within * 8u + block * stride_u128;
    // Swizzle<3,4,3> is defined on *byte* offsets. We operate in u128 (16-byte) units here,
    // so the equivalent transform is Swizzle<3,0,3>: XOR bits[0:2] with bits[3:5] shifted down.
    const uint32_t yyy = (logical_u128 & 0x00000038u) >> 3;
    const uint32_t swizzled_u128 = logical_u128 ^ yyy;
    return swizzled_u128 * 16u;
  };

  // For the 256x128 cta_group::2 path we use direct per-rank tile offsets; keep descriptor
  // row remaps disabled.
  const uint64_t cta2_desc_a_row_offset = 0ull;
  const uint64_t cta2_desc_b_row_offset = 0ull;
  const uint64_t cta2_desc_sfa_row_offset = 0ull;

#pragma unroll
  for (int stage = 0; stage < PIPELINE_STAGES; ++stage) {
    const uint32_t sA_addr = tcgen05::cast_smem_ptr_to_uint(&sA[stage][0][0]);
    const uint32_t sSFA_addr = tcgen05::cast_smem_ptr_to_uint(&sSFA[stage][0][0]);

    desc_a_base[stage].version_ = 1;
    desc_a_base[stage].lbo_mode_ = 0;
    desc_a_base[stage].base_offset_ = 0;
    desc_a_base[stage].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_128B);
    desc_a_base[stage].start_address_ = static_cast<uint16_t>(sA_addr >> 4);
    desc_a_base[stage].leading_byte_offset_ = 1;
    // Major-K descriptor stride is in units of uint128 (4 LSB dropped).
    // For SW128 K-major, each logical row is 128B, so every 8-row block is 1024B => 64 u128.
    desc_a_base[stage].stride_byte_offset_ = 64;

#pragma unroll
    for (int u = 0; u < UnrollN; ++u) {
      uint32_t sB_addr = tcgen05::cast_smem_ptr_to_uint(&sB[stage][u][0][0]);
      uint32_t sSFB_addr = tcgen05::cast_smem_ptr_to_uint(&sSFB[stage][u][0][0]);
      if constexpr (UnrollN == 2) {
        // Packed-UnrollN=2 loads place both N tiles into [u=0] SMEM buffers.
        // Remap u=1 descriptors to the second half of that packed buffer.
        if constexpr (CtaGroup == 1) {
          sB_addr = tcgen05::cast_smem_ptr_to_uint(&sB[stage][0][0][0]);
          sSFB_addr = tcgen05::cast_smem_ptr_to_uint(&sSFB[stage][0][0][0]);
          if (u > 0) {
            sB_addr += sw128_major_k_row_offset_bytes(static_cast<int>(u * TILE_N), /*stride_u128=*/64u);
            sSFB_addr += static_cast<uint32_t>(u) * static_cast<uint32_t>(SFB_TILE_BYTES);
          }
        } else if constexpr (CtaGroup == 2) {
          if (cta2_partition_b_mode != 1) {
            sB_addr = tcgen05::cast_smem_ptr_to_uint(&sB[stage][0][0][0]);
            sSFB_addr = tcgen05::cast_smem_ptr_to_uint(&sSFB[stage][0][0][0]);
            if (u > 0) {
              sB_addr += sw128_major_k_row_offset_bytes(static_cast<int>(u * TILE_N), /*stride_u128=*/64u);
              sSFB_addr += static_cast<uint32_t>(u) * static_cast<uint32_t>(SFB_TILE_BYTES);
            }
          }
        }
      }

      desc_b_base[stage][u].version_ = 1;
      desc_b_base[stage][u].lbo_mode_ = 0;
      desc_b_base[stage][u].base_offset_ = 0;
      desc_b_base[stage][u].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_128B);
      if constexpr (CtaGroup == 2) {
        // Partition B along N/2 across the two CTAs by shifting the *SMEM descriptor base* (not the
        // global coordinate). This keeps the shared-memory tile fully initialized and avoids UB
        // from reading uninitialized rows.
        //
        // Mode:
        //   cta2_partition_b=2 (default): descriptor shift by N/2 within the 128-row shared tile
        //   cta2_partition_b=1: global N shift by N/2 (experimental; requires special padding)
        if (cta2_partition_b_mode == 2) {
          sB_addr += static_cast<uint32_t>(cluster_rank) *
                     sw128_major_k_row_offset_bytes(static_cast<int>(TILE_N / 2), /*stride_u128=*/64u);
        }
      }
      desc_b_base[stage][u].start_address_ = static_cast<uint16_t>(sB_addr >> 4);
      desc_b_base[stage][u].leading_byte_offset_ = 1;
      // Major-K descriptor stride is in units of uint128 (4 LSB dropped).
      // For SW128 K-major, each logical row is 128B, so every 8-row block is 1024B => 64 u128.
      desc_b_base[stage][u].stride_byte_offset_ = 64;

      desc_sfb[stage][u].version_ = 1;
      desc_sfb[stage][u].lbo_mode_ = 0;
      desc_sfb[stage][u].base_offset_ = 0;
      desc_sfb[stage][u].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_NONE);
      // The B/SFB tiles for each unrolled-N operand live at `&sB[stage][u]` / `&sSFB[stage][u]`.
      // Any packed TMA experiments must update both the load ops and these descriptors together.
      desc_b_base[stage][u].start_address_ = static_cast<uint16_t>(sB_addr >> 4);
      desc_sfb[stage][u].start_address_ = static_cast<uint16_t>(sSFB_addr >> 4);
      desc_sfb[stage][u].leading_byte_offset_ = 1;
      // UMMA/UTCCP Major-K descriptor semantics:
      // - `leading_byte_offset_` is the stride between consecutive rows within an 8-row atom (in 16B units).
      // - `stride_byte_offset_` is the stride between 8-row blocks (in 16B units).
      //
      // For our packed scale tiles, each row is 16B and 8 rows span 128B => stride_byte_offset = 8.
      // (This matches the standalone CUTLASS-layout probe and avoids `cudaErrorMisalignedAddress` in UTCCP.)
      desc_sfb[stage][u].stride_byte_offset_ = 8;
    }

    desc_sfa[stage].version_ = 1;
    desc_sfa[stage].lbo_mode_ = 0;
    desc_sfa[stage].base_offset_ = 0;
    desc_sfa[stage].layout_type_ = static_cast<uint8_t>(umma::LayoutType::SWIZZLE_NONE);
    desc_sfa[stage].start_address_ = static_cast<uint16_t>(sSFA_addr >> 4);
    desc_sfa[stage].leading_byte_offset_ = 1;
    // See desc_sfb above for rationale.
    desc_sfa[stage].stride_byte_offset_ = 8;
  }

  // Allocate TMEM once per CTA.
  __shared__ uint32_t tmem_base_c;
  __shared__ volatile int ws_scales_ready_tile;
  __shared__ volatile int ws_u1_done_tile;
  __shared__ volatile int ws_u0_ready_tile;
  __shared__ volatile int ws_u0_done_tile;
  __shared__ volatile int ws_sfb1_seg2_ready_tile;
  __shared__ volatile int ws_sfb1_seg3_ready_tile;
  __shared__ volatile int ws_seg0_done_tile;
  __shared__ volatile int ws_tma_req_tile;
  __shared__ volatile int ws_tma_req_stage;
  __shared__ volatile int ws_tma_req_k_byte;
  __shared__ volatile int ws_tma_req_sfa_row_offset;
  __shared__ volatile int ws_tma_req_sfb_row_offset;
  // NOTE: tcgen05.alloc/dealloc are warp-synchronous: issue from a single fully-active warp.
  if constexpr (CtaGroup == 2) {
    // Ensure the 2 participating CTAs reach alloc together (requirement for cta_group::2).
    cluster.sync();
  }
		  if (threadIdx.x < 32) {
		    // Columns must be power-of-2, 32..512.
		    tcgen05::tmem_alloc<CtaGroup>(&tmem_base_c, /*num_columns=*/TMEM_COLUMNS);
				    if constexpr (WS_UNROLL2_MMA) {
				      if (threadIdx.x == 0) {
			        ws_scales_ready_tile = -1;
			        ws_u1_done_tile = -1;
			        if constexpr (WS_SFB1_SEGMENT_HELPERS) {
		          ws_sfb1_seg2_ready_tile = -1;
		          ws_sfb1_seg3_ready_tile = -1;
		        }
		        if constexpr (WS_SPLIT_U0_SEGS) {
		          ws_u0_ready_tile = -1;
		          ws_u0_done_tile = -1;
		        }
			      }
			    }
	    if constexpr (WS_SEGMENT_PARALLEL || CTA2_MMA_SEGMENT_PARALLEL) {
	      if (threadIdx.x == 0) {
	        ws_seg0_done_tile = -1;
	      }
	    }
	    if constexpr (WS_TMA_PRODUCER) {
	      if (threadIdx.x == 0) {
	        ws_tma_req_tile = -1;
	      }
	    }
	  }
  __syncthreads();
  if constexpr (CtaGroup == 2) {
    cluster.sync();
  }

	  // TMEM address plan (word addressing).
	  //
	  // Important: `tcgen05.alloc.cta_group::2` returns a TMEM pointer shared by both CTAs.
	  //
	  // For cta_group::2 + UnrollN=2 (M_MMA=64 per CTA), CUTLASS' 2SM accumulator mapping is:
	  //   - per-tile: addr(rank, u) = base + rank*128 + u*64  (word columns; see probe)
	  //   - external t (e.g., UnrollN=2 128-column tiles): t adds +256 columns
	  // See `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu` for the measured deltas and dp behavior.
	  const uint32_t tmem_addr_base = tmem_base_c;
  // IMPORTANT:
  // TMEM pointers are encoded as {col:16, dp:8, idx:8}. The `idx` field identifies the allocation
	  // returned by `tcgen05.alloc` and must remain intact. Do not "partition by rank" by mutating idx.
  //
	  // For cta_group::2 + UnrollN=2 (UMMA_2SM, M_MMA=64 per CTA):
	  // - Accumulators follow the CUTLASS-probed mapping (rank*128 + u_int*64 + t*256 in COL, and
	  //   N128 half encoded in DP via dp += 64). See `tmem_sf_frg_probe.cu`.
	  // - Scale factors use CUTLASS' `tmem_sf_frg` mapping. Note that (SFA,SFB) TMEM pointers are
	  //   interpreted by block-scaled UMMA as scale-factor addresses (not accumulator addresses), so
	  //   their DP/COL coordinates can overlap the accumulator coordinate space without corrupting C.
	  uint32_t tmem_c_tiles[UnrollN];
	  uint32_t tmem_sfa_ptrs[4];
	  uint32_t tmem_sfb_ptrs[4 * UnrollN];
  constexpr uint32_t SFA_NUM_MN = 1u;
  constexpr uint32_t SFB_NUM_MN = static_cast<uint32_t>(UnrollN);
  constexpr uint32_t SFA_COL_EXTENT = SF_COLS_PER_TILE_PER_MN * SFA_NUM_MN;
  constexpr uint32_t SFB_COL_EXTENT = SF_COLS_PER_TILE_PER_MN * SFB_NUM_MN;

  const uint32_t tmem_c_rank = [&]() -> uint32_t {
    if constexpr (CtaGroup == 2) {
      // Both UnrollN=1 and UnrollN=2 use the same per-rank base for M_MMA=64:
      //   base(rank) = base + rank*128
      // UnrollN=2 adds an *external* tile dimension (t) at +256 columns (see below).
      return tcgen05::tmem_addr_add(
          tmem_addr_base,
          /*dp_add=*/0u,
          /*col_add=*/static_cast<uint32_t>(cluster_rank) * 128u +
              static_cast<uint32_t>(cta2_tmem_c_word_offset));
    } else {
      return tcgen05::tmem_addr_add(
			          tmem_addr_base,
			          /*dp_add=*/0u,
			          /*col_add=*/static_cast<uint32_t>(cta2_tmem_c_word_offset));
			    }
			  }();

			  tmem_c_tiles[0] = tmem_c_rank;
				  if constexpr (UnrollN == 2) {
				    if constexpr (CtaGroup == 2) {
				      // CUTLASS probe for UMMA_2SM M_MMA=64 shows the external tile dimension (t) is at +256
					      // columns. Keep this mapping for correctness; scale factors are stored in a disjoint dp
					      // bank (dp+64) so we can still address them while using all 512 columns for accumulators.
				      tmem_c_tiles[1] = tcgen05::tmem_addr_add(tmem_c_rank, /*dp_add=*/0u, /*col_add=*/256u);
				    } else {
				      // cta_group::1: place the second N tile at +128 columns.
				      tmem_c_tiles[1] = tcgen05::tmem_addr_add(tmem_c_rank, /*dp_add=*/0u, /*col_add=*/128u);
			    }
			  }

	  // Use CUTLASS' `tmem_sf_frg` mapping for cta_group::2 scale-factor fragments.
	  //
	  // This mapping is the ground truth for SM100 block-scaled UMMA (`mxf4nvf4.block_scale.block16`)
	  // and is required for correct SFA/SFB addressing across the 2-CTA group.
  // Enable CUTLASS `tmem_sf_frg` mapping for all cta_group::2 variants so UMMA consumes the same
  // TMEM scale layout it was designed for. The legacy cta2 mapping was useful for UTCCP bring-up,
  // but it can still be incorrect for UMMA even when the dumped scale bytes look correct.
  constexpr bool kUseCutlassTmemSfFrg =
      (CtaGroup == 2) && (UnrollN == 2) && (NVFP4_GROUP_GEMM_USE_CUTLASS_TMEM_SF_FRG != 0);

			      const uint32_t tmem_sf_base =
			      (CtaGroup == 2)
			          ? [&]() -> uint32_t {
			              if constexpr (kUseCutlassTmemSfFrg) {
			                // CUTLASS `tmem_sf_frg` mapping is defined in the same TMEM address space as
			                // the CUTLASS-probed accumulator layout (`tmem_frg_2sm`).
			                //
			                // Important nuance for our tcgen05 kernels:
			                // - UnrollN=1: CUTLASS accumulator colspan is 256, so we have a clean, disjoint
			                //   scale region at col+=256 within the same allocation. Do NOT use dp+=64 as a
			                //   "bank": dp is already consumed by the accumulator layout for n>=64, and
			                //   overlap can corrupt accumulators and/or yield NaNs.
			                // - UnrollN=2: accumulator colspan expands to 512, so we cannot use a pure col
			                //   offset within a 512-col allocation; we instead place scales in a disjoint dp
			                //   window (still experimental).
			                if constexpr (UnrollN == 1) {
			                  return tcgen05::tmem_addr_add(
			                      tmem_addr_base,
			                      /*dp_add=*/0u,
			                      /*col_add=*/256u + static_cast<uint32_t>(cta2_tmem_sf_word_offset));
			                }
			                // NOTE: SM100 TMEM uses 7 dp bits (0..127). Use dp+64 as the disjoint scale window.
			                constexpr uint32_t kSfDpAdd = (NVFP4_GROUP_GEMM_CTA2_SF_DP_BANK != 0) ? 64u : 0u;
			                return tcgen05::tmem_addr_add(
			                    tmem_addr_base,
			                    /*dp_add=*/kSfDpAdd,
			                    /*col_add=*/static_cast<uint32_t>(cta2_tmem_sf_word_offset));
			              }
			              if constexpr (UnrollN == 2) {
			                // Legacy cta_group::2 + UnrollN=2 bring-up:
			                // Use a disjoint DP bank for scale storage (same rationale as CUTLASS mapping).
			                // If DP banking is disabled, fall back to column-offset placement (may overlap accumulators).
			                // NOTE: SM100 TMEM uses 7 dp bits (0..127). Use dp+64 as the disjoint bank.
			                constexpr uint32_t kSfDpAdd = (NVFP4_GROUP_GEMM_CTA2_SF_DP_BANK != 0) ? 64u : 0u;
		                return tcgen05::tmem_addr_add(
		                    tmem_addr_base,
		                    /*dp_add=*/kSfDpAdd,
		                    /*col_add=*/static_cast<uint32_t>(cta2_tmem_sf_word_offset));
		              }
		              // Legacy cta_group::2 UnrollN=1 path: place scales in the second half of TMEM columns
		              // (col += 256) since accum uses only 256 cols.
	              return tcgen05::tmem_addr_add(
	                  tmem_addr_base,
	                  /*dp_add=*/0u,
	                  /*col_add=*/256u + static_cast<uint32_t>(cta2_tmem_sf_word_offset));
	            }()
	          : tcgen05::tmem_addr_add(
	                tmem_addr_base,
	                /*dp_add=*/0u,
	                /*col_add=*/(UnrollN == 2) ? 256u : static_cast<uint32_t>(cta2_tmem_sf_word_offset));
  uint32_t tmem_sf_base_eff = tmem_sf_base;
  if constexpr ((CtaGroup == 2) && (UnrollN == 2)) {
    if (debug_tmem_idx_add > 0 && debug_tmem_idx_add < 64) {
      // Experimental disjoint TMEM bank selector for scale fragments.
      // Keep this opt-in/off-by-default so existing behavior is unchanged.
      tmem_sf_base_eff =
          static_cast<uint32_t>(tmem_sf_base + (static_cast<uint32_t>(debug_tmem_idx_add) << 24));
    }
  }
  const uint32_t tmem_sf_rank_base =
      (CtaGroup == 2)
          ? [&]() -> uint32_t {
              if constexpr (UnrollN == 2) {
                // cta_group::2 + UnrollN=2: keep per-rank TMEM scale windows disjoint.
                return tcgen05::tmem_addr_add(
                    tmem_sf_base_eff,
                    /*dp_add=*/0u,
                    /*col_add=*/static_cast<uint32_t>(cluster_rank) *
                        static_cast<uint32_t>(cta2_tmem_sf_rank_word_offset));
              } else {
                // Legacy cta_group::2 UnrollN=1 path (known-correct anchor): rank partition in cols.
                return tcgen05::tmem_addr_add(
                    tmem_sf_base_eff,
                    /*dp_add=*/0u,
                    /*col_add=*/static_cast<uint32_t>(cluster_rank) *
                        static_cast<uint32_t>(cta2_tmem_sf_rank_word_offset));
              }
            }()
          : tmem_sf_base_eff;

	  // For cta_group::2 + UnrollN=2, CUTLASS' `tmem_sf_frg` mapping already incorporates rank and u
	  // into the TMEM column offsets (see probe). Do not additionally "window" TMEM by rank, otherwise
	  // rank offsets double-count and UMMA consumes the wrong scale-id banks.
  //
  // Also note (M_MMA=64, N_SM=2):
  // - CUTLASS `tmem_sf_frg` mapping: SFA spans 16 columns across both ranks (8 per rank), so SFB
  //   starts at +16 columns.
  // - Legacy rank-windowed mapping: we add padding so SFB still starts on a 16-column boundary.
  const uint32_t tmem_sfa_base_raw =
      kUseCutlassTmemSfFrg ? tmem_sf_base_eff : tmem_sf_rank_base;
	  // For CUTLASS' `tmem_sf_frg` mapping, SFA and SFB have distinct *logical* layouts and are addressed
	  // independently by block-scaled UMMA, but they still occupy TMEM storage and must not overlap.
	  // Place SFB immediately after SFA's column span (CUTLASS' mapping uses +16 columns for N_SM=2).
	  //
	  // For the legacy rank-windowed mapping, we also pack SFB after SFA within the same TMEM slice and
	  // keep the SFB base 16-column aligned to satisfy UTCCP requirements on SM100.
  const uint32_t sfa_cols_span = kUseCutlassTmemSfFrg
                                     ? ((CtaGroup == 2) ? (SF_COLS_PER_TILE_PER_MN_CUTLASS * 2u)
                                                        : SF_COLS_PER_TILE_PER_MN_CUTLASS)
                                     : SFA_COL_EXTENT;
	  const uint32_t tmem_sfb_base_raw =
	      tcgen05::tmem_addr_add(tmem_sfa_base_raw, /*dp_add=*/0u, /*col_add=*/sfa_cols_span);

  // IMPORTANT (block-scaled UMMA, cta_group::2 + UnrollN=2):
  //
  // CUTLASS encodes the UMMA "scale-id" (a_sf_id_/b_sf_id_) via the *top-2 bits* of the TMEM
  // pointer passed to `make_runtime_instr_desc_block_scaled()` / `tcgen05.mma`. However, UTCCP
  // stores can trap if those top-2 bits are non-zero. So keep the UTCCP destination pointers
  // (tmem_sfa_ptrs/tmem_sfb_ptrs) with top-2 bits cleared, and apply the scale-id bits only on
  // the pointers used by UMMA.
  const uint32_t tmem_sfa_base = tmem_sfa_base_raw;
  const uint32_t tmem_sfb_base = tmem_sfb_base_raw;
#pragma unroll
  const int sfa_rank = cluster_rank;
  // SFB rank follows cluster rank so each CTA consumes the SFB sub-fragment layout expected by
  // cta_group::2 UMMA. Mode1 already shifts source SFB rows per rank.
  const int sfb_rank = cluster_rank;
  for (int seg = 0; seg < 4; ++seg) {
    if constexpr (kUseCutlassTmemSfFrg) {
      // CUTLASS `tmem_sf_frg` mapping for cta_group::2 (M_MMA=64):
      //   col(rank, seg) = rank*SF_COLS_PER_TILE_PER_MN + seg*SF_COLS_PER_KBLOCK_PER_MN
      const uint32_t col_add =
          static_cast<uint32_t>(sfa_rank) * SF_COLS_PER_TILE_PER_MN_CUTLASS +
          static_cast<uint32_t>(seg) * SF_COLS_PER_KBLOCK_PER_MN_CUTLASS;
      // IMPORTANT (cta_group::2 + UTCCP64 correctness):
      // We always compute the CUTLASS-probed per-seg TMEM column offsets here (seg stride = 2 cols).
      // When UTCCP64 is used, the 64-row copy must populate *both* segs (seg0+seg1, seg2+seg3) into
      // their respective per-seg columns (col{0,2} and col{4,6}). Do not encode seg selection via DP.
      tmem_sfa_ptrs[seg] = tcgen05::tmem_addr_add(tmem_sfa_base, /*dp_add=*/0u, /*col_add=*/col_add);
	    } else {
	      if constexpr (CtaGroup == 2) {
	        // Legacy cta_group::2 scale layout (UTCCP32-friendly):
	        // Use a simple, dense 32x16B per-segment placement (4 word-columns per K64 segment).
	        // This keeps all destination columns 16B-aligned so we can safely use 32x128b.warpx4
	        // instead of relying on UTCCP64's packed/unaligned behavior.
	        constexpr uint32_t kSegStrideCols = 4u;  // 16B per row
	        const uint32_t col_add = static_cast<uint32_t>(seg) * kSegStrideCols;
	        tmem_sfa_ptrs[seg] = tcgen05::tmem_addr_add(tmem_sfa_base, /*dp_add=*/0u, /*col_add=*/col_add);
	      } else {
	        tmem_sfa_ptrs[seg] = tcgen05::tmem_addr_add(
	            tmem_sfa_base,
	            /*dp_add=*/0u,
	            /*col_add=*/static_cast<uint32_t>(seg) * (SF_COLS_PER_KBLOCK_PER_MN * SFA_NUM_MN));
	      }
	    }
	  }
#pragma unroll
  for (int u = 0; u < UnrollN; ++u) {
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      if constexpr (kUseCutlassTmemSfFrg) {
        // CUTLASS probe (UMMA_2SM, M_MMA=64, num_MMA_K=4, UE4M3 scales) for SFB `tmem_sf_frg`:
        //
        // - 4x1 alloc (ScaleFactorDuplicated4by1):
        //     col(rank, u_int, seg, t) = rank*16 + seg*4 + u_int*2 + t*32
        // - 2x2 alloc (ScaleFactorDuplicated2by2):
        //     col(rank, u_int, seg, t) = rank*8 + seg*2 + u_int*1 + t*16
        //
        // Where:
        // - `rank` is the CTA rank in the cta_group::2 cluster (0/1)
        // - `u_int` is CUTLASS' internal UMMA_2SM SFB subpartition (0/1) that is populated implicitly
        //   by the 16-byte-wide UTCCP copies when using the `u_int=0` base pointer.
        // - `t` is the external tile index (we map our UnrollN tiles to `t`).
        //
        // See: `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu`
        uint32_t col_add = 0u;
#if NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE == 0
        col_add =
            static_cast<uint32_t>(sfb_rank) * 16u +
            static_cast<uint32_t>(seg) * 4u +
            static_cast<uint32_t>(u) * 32u;
#elif NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE == 1
        col_add =
            static_cast<uint32_t>(sfb_rank) * 8u +
            static_cast<uint32_t>(seg) * 2u +
            static_cast<uint32_t>(u) * 16u;
#else
        static_assert(NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE == 0 || NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE == 1,
                      "NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE must be 0 (4x1) or 1 (2x2).");
#endif
        // IMPORTANT (cta_group::2 + UTCCP64 correctness):
        // Compute CUTLASS-probed per-(seg,tile) TMEM column offsets (seg stride is 4 cols for 4x1 alloc,
        // 2 cols for 2x2 alloc). UTCCP64 must populate both segs in each 64-row op into these per-seg
        // columns; do not encode seg selection via DP.
        tmem_sfb_ptrs[u * 4 + seg] =
            tcgen05::tmem_addr_add(tmem_sfb_base, /*dp_add=*/0u, /*col_add=*/col_add);
	      } else {
	        // Legacy SFB TMEM layout (non-CUTLASS `tmem_sf_frg` mapping).
	        //
	        // For cta_group::2 + UnrollN=2 we must keep u=0 and u=1 disjoint in TMEM. The previous
	        // bring-up path mirrored SFA and ignored `u`, which caused u1 to overwrite u0 and produced
	        // NaNs/verify failures.
	        if constexpr (CtaGroup == 2) {
	          constexpr uint32_t kSegStrideCols = 4u;   // 16B per row (UTCCP32-friendly)
	          constexpr uint32_t kTileStrideCols = 16u; // 4 segs * 4 cols
	          uint32_t col_add =
	              static_cast<uint32_t>(seg) * (kSegStrideCols * SFB_NUM_MN) +
	              static_cast<uint32_t>(u) * kSegStrideCols;
	          if (cta2_sfb_slot_mode == 1) {
	            col_add = static_cast<uint32_t>(u) * kTileStrideCols +
	                      static_cast<uint32_t>(seg) * kSegStrideCols;
	          }
	          tmem_sfb_ptrs[u * 4 + seg] = tcgen05::tmem_addr_add(tmem_sfb_base, /*dp_add=*/0u, /*col_add=*/col_add);
	        } else {
	          // cta_group::1 correctness-first mapping: interleave MN tiles by K64 segment in the TMEM column field.
	          uint32_t col_add = static_cast<uint32_t>(seg) * (SF_COLS_PER_KBLOCK_PER_MN * SFB_NUM_MN) +
	                             static_cast<uint32_t>(u) * SF_COLS_PER_KBLOCK_PER_MN;
	          tmem_sfb_ptrs[u * 4 + seg] = tcgen05::tmem_addr_add(tmem_sfb_base, /*dp_add=*/0u, /*col_add=*/col_add);
	        }
	      }
	    }
	  }
		  if constexpr (CtaGroup == 2) {
		    if (debug_print_ptrs != 0 && group_idx == 0 && tile_m == 0 && tile_n == 0 && threadIdx.x == 0) {
		      const uint32_t sSFA_addr_dbg = tcgen05::cast_smem_ptr_to_uint(&sSFA[0][0][0]);
		      const uint32_t sSFB_addr_dbg = tcgen05::cast_smem_ptr_to_uint(&sSFB[0][0][0][0]);
		      const uint64_t desc_sfa0_dbg = static_cast<uint64_t>(desc_sfa[0]);
		      const uint64_t desc_sfb0_dbg = static_cast<uint64_t>(desc_sfb[0][0]);
		      const uint32_t tmem_c1 = (UnrollN == 2) ? tmem_c_tiles[1] : 0u;
		      const uint32_t tsfa0_cp =
		          static_cast<uint32_t>(tmem_sfa_ptrs[0] + static_cast<uint32_t>(cta2_tsfa_word_offset));
		      const uint32_t tsfb0_cp =
		          static_cast<uint32_t>(tmem_sfb_ptrs[0] + static_cast<uint32_t>(cta2_tsfb_word_offset));
	      const uint32_t tsfb1 =
	          (UnrollN == 2)
	              ? static_cast<uint32_t>(tmem_sfb_ptrs[4] + static_cast<uint32_t>(cta2_tsfb_word_offset))
	              : 0u;
	      uint32_t tsfa0_mma = tsfa0_cp;
	      uint32_t tsfb0_mma = tsfb0_cp;
	      if constexpr (UnrollN == 2) {
	        tsfa0_mma = tcgen05::tmem_set_top2_bits(tsfa0_mma, static_cast<uint32_t>(cta2_sfa_sf_id));
	        tsfb0_mma = tcgen05::tmem_set_top2_bits(tsfb0_mma, static_cast<uint32_t>(cta2_sfb_sf_id));
		      }
		      const uint32_t sfa_id = (tsfa0_mma & 0xC0000000u) >> 30;
		      const uint32_t sfb_id = (tsfb0_mma & 0xC0000000u) >> 30;
		      printf("cta2 rank=%d sSFA=0x%08x(sfa%%128=%u) sSFB=0x%08x(sfb%%128=%u) desc_sfa0=0x%016llx desc_sfb0=0x%016llx "
		             "tmem_base=0x%08x tmem_c0=0x%08x tmem_c1=0x%08x tsfa0_cp=0x%08x tsfa0_mma=0x%08x(tsfa_id=%u) tsfb0_cp=0x%08x tsfb0_mma=0x%08x tsfb1_cp=0x%08x(tsfb_id=%u)\\n",
		             cluster_rank,
		             sSFA_addr_dbg, static_cast<unsigned>(sSFA_addr_dbg & 127u),
		             sSFB_addr_dbg, static_cast<unsigned>(sSFB_addr_dbg & 127u),
		             static_cast<unsigned long long>(desc_sfa0_dbg),
		             static_cast<unsigned long long>(desc_sfb0_dbg),
		             tmem_base_c, tmem_c_rank, tmem_c1, tsfa0_cp, tsfa0_mma, sfa_id, tsfb0_cp, tsfb0_mma, tsfb1, sfb_id);
		    }
		  } else {
    if (debug_print_ptrs != 0 && group_idx == 0 && tile_m == 0 && tile_n == 0 && threadIdx.x == 0) {
      const uint32_t tsfa0 = tmem_sfa_ptrs[0];
      const uint32_t tsfb0 = tmem_sfb_ptrs[0];
      const uint32_t tmem_c0 = tmem_c_tiles[0];
      const uint32_t tmem_c1 = (UnrollN == 2) ? tmem_c_tiles[1] : 0u;
	      printf("cta1 tmem_base=0x%08x tmem_c0=0x%08x tmem_c1=0x%08x tsfa0=0x%08x tsfb0=0x%08x\\n",
	             tmem_base_c, tmem_c0, tmem_c1, tsfa0, tsfb0);
	    }
	  }

  // Instruction descriptor (block scaled, MXF4 E2M1, scale UE4M3, K-major operands).
  umma::InstrDescriptorBlockScaled idesc{};
  idesc.sparse_id2_ = 0;
  idesc.sparse_flag_ = 0;
  idesc.b_sf_id_ = 0;
  idesc.a_format_ = 1;  // MXF4Format::E2M1
  idesc.b_format_ = 1;  // MXF4Format::E2M1
  idesc.a_negate_ = 0;
  idesc.b_negate_ = 0;
  idesc.a_major_ = static_cast<uint8_t>(umma::Major::K);
  idesc.b_major_ = static_cast<uint8_t>(umma::Major::K);
  idesc.n_dim_ = (TILE_N_MMA >> 3);
  idesc.scale_format_ = 0;  // ScaleFormat::UE4M3
  // For cta_group::2, we use the M=256 2-CTA cluster MMA variant (each CTA contributes 128 rows).
  // The instruction descriptor must encode the *full* M dimension for the 2-CTA group instruction.
  // For cta_group::1, keep the canonical full tile M dimension. CUTLASS encodes `m_dim_ = (M >> 4)`
  // with compile-time M (128 here). Some runtime-masked M encodings can trigger illegal instruction
  // faults on SM100 when using block-scaled UMMA.
  idesc.m_dim_ = (CtaGroup == 2) ? (CLUSTER_TILE_M >> 4) : (TILE_M >> 4);
  idesc.a_sf_id_ = 0;
  idesc.k_size_ = 0;  // MXF4 dense K64
  if constexpr (CtaGroup == 2) {
    // Bring-up debug knobs: allow overriding the descriptor M/N dims to validate 2CTA semantics.
    if (cta2_idesc_m_dim_override > 0) {
      idesc.m_dim_ = static_cast<uint16_t>(cta2_idesc_m_dim_override);
    }
    if (cta2_idesc_n_dim_override > 0) {
      idesc.n_dim_ = static_cast<uint16_t>(cta2_idesc_n_dim_override);
    }
  }

  // Precompute per-segment TMEM scale pointers and runtime instruction descriptor hi bits.
  // These are invariant across K tiles: each K tile overwrites the same TMEM scale slots.
  uint32_t tmem_sfa_seg[4];
  uint32_t tmem_sfb_seg[UnrollN][4];
  uint32_t idesc_hi_seg[UnrollN][4];
  // MMA issue participation experiments:
  // - Default schedules issue UMMA from a subset of lanes (thread0 / lane0-only), so only those
  //   lanes need the precomputed descriptor fragments in registers.
  // - When we intentionally issue UMMA from all threads (CTA2_MMA_ALL_THREADS), every lane must
  //   pass valid (uniform) operands; leaving these arrays uninitialized for non-lane0 threads can
  //   trigger `cudaErrorIllegalInstruction`.
  const bool compute_idesc_for_lane =
      CTA2_MMA_ALL_THREADS ? true : (((threadIdx.x & 31) == 0) ? true : false);
  if (compute_idesc_for_lane) {
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      // CUTLASS `tmem_sf_frg` + UTCCP64 (cta_group::2) nuance:
      // The 64x128b scale copy writes a full 16B row, which spans two 8B "halves" in TMEM
      // (col0 and col2 for the first copy; col4 and col6 for the second). The misaligned
      // half-row base pointers (seg=1,3) are therefore *not* valid scale-fragment bases for UMMA.
      // Use the aligned base (seg=0 for {0,1} and seg=2 for {2,3}) for the UMMA scale operands.
      const int seg_eff_sfa =
          (kUseCutlassTmemSfFrg && (NVFP4_GROUP_GEMM_CTA2_SFA_PAIR_SEGS != 0)) ? (seg & ~1) : seg;
      const uint32_t tmem_sfa_cp = (CtaGroup == 2)
                                       ? tcgen05::tmem_addr_add(
                                             tmem_sfa_ptrs[seg_eff_sfa],
                                             /*dp_add=*/0u,
                                             /*col_add=*/static_cast<uint32_t>(cta2_tsfa_word_offset))
                                       : tmem_sfa_ptrs[seg_eff_sfa];
      // Keep descriptor and operand TMEM scale pointers aligned to a single contract.
      // cta_group::2 UnrollN=2 keeps the optional SF-ID override path via top-2 bits.
      uint32_t tmem_sfa = tmem_sfa_cp;
      if constexpr (CtaGroup == 2 && UnrollN == 2) {
        tmem_sfa = tcgen05::tmem_set_top2_bits(tmem_sfa, static_cast<uint32_t>(cta2_sfa_sf_id));
      }
      tmem_sfa_seg[seg] = tmem_sfa;
#pragma unroll
      for (int u = 0; u < UnrollN; ++u) {
        // SFB `tmem_sf_frg` mapping depends on alloc mode:
        // - 4x1 alloc (default): per-seg columns are 16B-aligned; UTCCP32 writes each seg independently.
        // - 2x2 alloc: seg stride is 8B (misaligned), which forces UTCCP64 and requires seg pairing.
        constexpr bool kPairSfbSegs =
            (kUseCutlassTmemSfFrg && (NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE == 1) &&
             (NVFP4_GROUP_GEMM_CTA2_SFB_PAIR_SEGS != 0));
        const int seg_eff_sfb = kPairSfbSegs ? (seg & ~1) : seg;
        const uint32_t tmem_sfb_cp = (CtaGroup == 2)
                                         ? tcgen05::tmem_addr_add(
                                               tmem_sfb_ptrs[u * 4 + seg_eff_sfb],
                                               /*dp_add=*/0u,
                                               /*col_add=*/static_cast<uint32_t>(cta2_tsfb_word_offset))
                                         : tmem_sfb_ptrs[u * 4 + seg_eff_sfb];
        uint32_t tmem_sfb = tmem_sfb_cp;
        if constexpr (CtaGroup == 2 && UnrollN == 2) {
          tmem_sfb = tcgen05::tmem_set_top2_bits(tmem_sfb, static_cast<uint32_t>(cta2_sfb_sf_id));
        }
        tmem_sfb_seg[u][seg] = tmem_sfb;
        const uint64_t idesc_runtime =
            umma::make_runtime_instr_desc_block_scaled(idesc, tmem_sfa, tmem_sfb);
        idesc_hi_seg[u][seg] = static_cast<uint32_t>(idesc_runtime >> 32);
      }
    }
  }
  if (debug_print_ptrs != 0 && group_idx == 0 && tile_m == 0 && tile_n == 0 && threadIdx.x == 0) {
    printf("dbg_sf_raw rank=%d sfa_ptrs=[0x%08x,0x%08x,0x%08x,0x%08x] "
           "sfb_u0_ptrs=[0x%08x,0x%08x,0x%08x,0x%08x] sfb_u1_ptrs=[0x%08x,0x%08x,0x%08x,0x%08x]\\n",
           cluster_rank,
           tmem_sfa_ptrs[0], tmem_sfa_ptrs[1], tmem_sfa_ptrs[2], tmem_sfa_ptrs[3],
           tmem_sfb_ptrs[0], tmem_sfb_ptrs[1], tmem_sfb_ptrs[2], tmem_sfb_ptrs[3],
           (UnrollN == 2) ? tmem_sfb_ptrs[4] : 0u,
           (UnrollN == 2) ? tmem_sfb_ptrs[5] : 0u,
           (UnrollN == 2) ? tmem_sfb_ptrs[6] : 0u,
           (UnrollN == 2) ? tmem_sfb_ptrs[7] : 0u);
    printf("dbg_sf_mma rank=%d sfa_seg=[0x%08x,0x%08x,0x%08x,0x%08x] "
           "sfb_u0_seg=[0x%08x,0x%08x,0x%08x,0x%08x] sfb_u1_seg=[0x%08x,0x%08x,0x%08x,0x%08x]\\n",
           cluster_rank,
           tmem_sfa_seg[0], tmem_sfa_seg[1], tmem_sfa_seg[2], tmem_sfa_seg[3],
           tmem_sfb_seg[0][0], tmem_sfb_seg[0][1], tmem_sfb_seg[0][2], tmem_sfb_seg[0][3],
           (UnrollN == 2) ? tmem_sfb_seg[1][0] : 0u,
           (UnrollN == 2) ? tmem_sfb_seg[1][1] : 0u,
           (UnrollN == 2) ? tmem_sfb_seg[1][2] : 0u,
           (UnrollN == 2) ? tmem_sfb_seg[1][3] : 0u);
  }

  // Shared-memory -> TMEM scale copy helper.
  // UnrollN=1 and all SFA copies use 32x128b.warpx4.
  // UnrollN=2 correctness-first: copy SFB per-unrolled-tile with the same 32x128b.warpx4 primitive.
  // (We can revisit 64x128b warpx2 once the CUTLASS layout is fully exploited.)
  auto copy_scale_fragments = [&](uint64_t src_desc_base, const uint32_t* tmem_ptrs, bool is_sfb) -> void {
	    if constexpr (CtaGroup == 2) {
      // Debug-only trap isolation: allow skipping SFA/SFB UTCCP copies independently.
	      if constexpr (NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFA != 0) {
	        if (!is_sfb) {
	          return;
	        }
	      }
	      if constexpr (NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFB != 0) {
	        if (is_sfb) {
	          return;
	        }
	      }

	      if constexpr (NVFP4_GROUP_GEMM_CTA2_UTCCP_RANK0_ONLY != 0) {
	        if (cluster_rank != 0) {
	          return;
	        }
	      }
	    }

    // cta_group::2: optionally populate the second SFB `u` subpartition (CUTLASS `tmem_sf_frg`).
    //
    // CUTLASS' UMMA_2SM SFB fragment has an internal `u` dimension (size 2) in addition to any
    // external tile dimension (e.g., our UnrollN=2 mapping to CUTLASS' `t`).
    //
    // In practice, leaving `u=1` uninitialized can produce large correctness errors concentrated
    // in the n>=64 half of each N128 tile. When enabled, we duplicate the same SMEM scale tile
    // into the `u=1` TMEM columns (col += u_delta).
    const bool populate_sfb_u1 =
        is_sfb && (CtaGroup == 2) && (kUseCutlassTmemSfFrg) && (NVFP4_GROUP_GEMM_CTA2_SFB_POPULATE_U1 != 0);
    uint32_t tmem_ptrs_u1[4];
    bool do_u1_copy = populate_sfb_u1;
    if (populate_sfb_u1) {
      constexpr uint32_t kU1ColDelta =
#if NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE == 0
          2u;  // 4x1 alloc: u_int stride = 2 columns
#else
          1u;  // 2x2 alloc: u_int stride = 1 column
#endif
#pragma unroll
      for (int seg = 0; seg < 4; ++seg) {
        tmem_ptrs_u1[seg] = tcgen05::tmem_addr_add(tmem_ptrs[seg], /*dp_add=*/0u, /*col_add=*/kU1ColDelta);
      }
    }
#if NVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF
    // Experimental: copy the entire 128-row scale tile (4x 32-row segments) in one UTCCP op.
    // This is cta_group::1-only; cta_group::2 does not have a 128x128b UTCCP variant in our wrappers.
    //
    // IMPORTANT (UnrollN=2): SFB tiles use an interleaved TMEM column layout across MN tiles:
    //   addr(u, seg) = base + seg * (4 * NumMN) + u * 4, with NumMN=2.
    // A single contiguous 128x128b copy only matches contiguous 16-column layout, so it is
    // valid for SFA (NumMN=1) but not for per-u SFB copies when UnrollN=2.
    // Keep SFB on 32x128b in that case to preserve correctness.
    if constexpr (CtaGroup == 1) {
      if (!is_sfb || (UnrollN == 1)) {
        tcgen05::utccp_cp_cta1_128x128b(src_desc_base, tmem_ptrs[0]);
      } else {
        constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
        for (int seg = 0; seg < 4; ++seg) {
          const uint64_t src_desc = src_desc_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
          tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs[seg]);
        }
      }
    } else {
      constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
      for (int seg = 0; seg < 4; ++seg) {
        const uint64_t src_desc = src_desc_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
        tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs[seg]);
      }
    }
 #elif NVFP4_GROUP_GEMM_USE_UTCCP_64X128B || NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA || NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB
    const bool use64 = is_sfb
                           ? ((NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB != 0) ||
                              (NVFP4_GROUP_GEMM_USE_UTCCP_64X128B != 0))
                           : ((NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA != 0) ||
                              (NVFP4_GROUP_GEMM_USE_UTCCP_64X128B != 0));
    // On SM100a, we've observed `tcgen05.cp.cta_group::2.32x128b.warpx4` can trap with
    // `cudaErrorMisalignedAddress` when the TMEM destination columns are not 16B-aligned.
    //
    // Fall back to UTCCP64 in that case (or when explicitly requested via the compile-time knob).
    bool dst_cols_aligned = true;
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      const uint32_t col = tmem_ptrs[seg] & 0x0000FFFFu;
      dst_cols_aligned &= ((col & 3u) == 0u);
    }
    const bool use64_eff = use64 || ((CtaGroup == 2) && (!dst_cols_aligned));
    // On SM100a, we've observed `tcgen05.cp.cta_group::2.32x128b.warpx4` can trap with
    // `cudaErrorMisalignedAddress` for the CUTLASS-probed scale-fragment destinations
    // (col offsets {0,2,4,6}). The 64x128b.warpx2 primitive is safe because it targets only
    // the aligned bases (seg01 at col0, seg23 at col4).
    //
    // Use 64x128b copies when requested or required by destination alignment.
    if (use64_eff) {
      // Copy two 32-row stripes at a time using the warp-specialized UTCCP64 primitive.
      //
	      // IMPORTANT (SM100a + UTCCP64 semantics, validated by `tmem_sf_frg_probe.cu`):
	      // For packed 128x16 scale tiles, `tcgen05.cp.*.64x128b.*` covers *two* 32-row segments per op:
	      //   - First op (src start +0): copies seg01 (rows 0..63) into the aligned base (col0)
	      //   - Second op (src start +64): copies seg23 (rows 64..127) into the aligned base (col4)
	      // `SmemDescriptor.start_address_` is in u128 (16B) rows, so +64 corresponds to +64 rows.
	      constexpr int kDescStepRows = 64;  // u128 rows (16B).
      const uint64_t src_desc02 = src_desc_base;
      const uint64_t src_desc13 = src_desc_base + static_cast<uint64_t>(kDescStepRows);
      const uint32_t dst_base02 = tmem_ptrs[0];
      const uint32_t dst_base13 =
          (CtaGroup == 2) ? (kUseCutlassTmemSfFrg ? tmem_ptrs[2] : tmem_ptrs[1]) : tmem_ptrs[2];
#if NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 0
      // ::01_23 for both operands.
      if constexpr (CtaGroup == 2) {
        if (threadIdx.x < 64) {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
        }
      } else {
        tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
        tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
      }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 1
      // ::02_13 for both operands.
      if constexpr (CtaGroup == 2) {
        if (threadIdx.x < 64) {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
        }
      } else {
        tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
        tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
      }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 2
      // SFA uses ::01_23, SFB uses ::02_13.
      if (is_sfb) {
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
        }
      } else {
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
        }
      }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 3
      // SFA uses ::02_13, SFB uses ::01_23.
      if (is_sfb) {
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
        }
      } else {
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
        }
      }
#else
#error "Unsupported NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE"
#endif
      if (populate_sfb_u1) {
        // cta_group::2 correctness: when SFB uses UTCCP64, we still must populate the internal
        // u_int=1 partition (u-shifted TMEM columns), otherwise UMMA observes missing scales.
        const uint32_t dst_base02_u1 = tmem_ptrs_u1[0];
        const uint32_t dst_base13_u1 =
            (CtaGroup == 2) ? (kUseCutlassTmemSfFrg ? tmem_ptrs_u1[2] : tmem_ptrs_u1[1]) : tmem_ptrs_u1[2];
#if NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 0
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02_u1);
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13_u1);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02_u1);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13_u1);
        }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 1
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02_u1);
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13_u1);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02_u1);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13_u1);
        }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 2
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02_u1);
            tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13_u1);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02_u1);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13_u1);
        }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 3
        if constexpr (CtaGroup == 2) {
          if (threadIdx.x < 64) {
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02_u1);
            tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13_u1);
          }
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02_u1);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13_u1);
        }
#else
#error "Unsupported NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE"
#endif
      }
    } else {
      constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
        for (int seg = 0; seg < 4; ++seg) {
          const uint64_t src_desc = src_desc_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
          tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs[seg]);
          if (do_u1_copy) {
            tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs_u1[seg]);
          }
        }
      }
#else
    constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
    // UTCCP `.warpx4` is a warpgroup-style primitive. Keep this unpredicated so the full CTA
    // participates (matches the standalone probe and avoids misaligned-address traps on SM100a).
    if (debug_print_ptrs != 0 && threadIdx.x == 0 && group_idx == 0 && tile_m == 0 && tile_n == 0) {
      umma::SmemDescriptor desc_dbg{};
      desc_dbg.desc_ = src_desc_base;
      printf("dbg_utccp_%s desc_base=0x%016llx tmem0=0x%08x lbo=%u sbo=%u layout=%u start=0x%x\\n",
             is_sfb ? "sfb" : "sfa",
             static_cast<unsigned long long>(src_desc_base),
             static_cast<unsigned>(tmem_ptrs[0]),
             static_cast<unsigned>(desc_dbg.leading_byte_offset_),
             static_cast<unsigned>(desc_dbg.stride_byte_offset_),
             static_cast<unsigned>(desc_dbg.layout_type_),
             static_cast<unsigned>(desc_dbg.start_address_));
    }
    // On SM100a, `tcgen05.cp.cta_group::2.32x128b.warpx4` can trap with `cudaErrorMisalignedAddress`
    // when the TMEM destination base is not 16B-aligned (col % 4 != 0). CUTLASS' `tmem_sf_frg`
    // mapping for SFA uses per-seg column offsets {0,2,4,6}, so seg1/seg3 are 8B-aligned but
    // not 16B-aligned.
    //
    // Fall back to UTCCP64 (warpx2) in that case: it targets only the aligned segment bases
    // (seg01 at col0, seg23 at col4) and copies two 32-row segments per op.
    bool dst_cols_aligned = true;
#pragma unroll
    for (int seg = 0; seg < 4; ++seg) {
      const uint32_t col = tmem_ptrs[seg] & 0x0000FFFFu;
      dst_cols_aligned &= ((col & 3u) == 0u);
    }
    const bool need_utccp64 = (CtaGroup == 2) && (!dst_cols_aligned);
    if (need_utccp64) {
      // See `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu` for validated UTCCP64 placement.
      constexpr int kDescStepRows = 64;  // u128 rows (16B) between seg01 and seg23 within a 128-row tile.
      const uint64_t src_desc02 = src_desc_base;
      const uint64_t src_desc13 = src_desc_base + static_cast<uint64_t>(kDescStepRows);
      const uint32_t dst_base02 = tmem_ptrs[0];
      const uint32_t dst_base13 = kUseCutlassTmemSfFrg ? tmem_ptrs[2] : tmem_ptrs[1];
#if NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 0
      if (threadIdx.x < 64) {
        tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
        tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
      }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 1
      if (threadIdx.x < 64) {
        tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
        tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
      }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 2
      if (threadIdx.x < 64) {
        if (is_sfb) {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
        }
      }
#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 3
      if (threadIdx.x < 64) {
        if (is_sfb) {
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_01_23<CtaGroup>(src_desc13, dst_base13);
        } else {
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc02, dst_base02);
          tcgen05::utccp_cp_64x128b_warpx2_02_13<CtaGroup>(src_desc13, dst_base13);
        }
      }
#else
#error "Unsupported NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE"
#endif
      // UTCCP64 path does not support the u_int=1 duplication copy today; in practice we only
      // hit this fallback for SFA (no u1) and for misaligned SFB layouts where do_u1_copy=false.
    } else {
      for (int seg = 0; seg < 4; ++seg) {
        const uint64_t src_desc = src_desc_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
        tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs[seg]);
        if (do_u1_copy) {
          tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, tmem_ptrs_u1[seg]);
        }
      }
    }
#endif
  };


  // TMA helper: issue a single-stage load for the given K tile into the selected pipeline stage.
  // Only thread0 participates in the barrier (expected_count=1) and owns the returned token.
	  auto issue_tma_tile = [&](int stage, int k_byte, int sfa_row_offset, int sfb_row_offset) -> block_barrier::arrival_token {
	    block_barrier* bar = bars[stage];
	    block_barrier::arrival_token tok;
	    // Stage all async TMA work under a single barrier generation.
	    // When enabled, multicast A + SFA from the cluster leader (rank0) to all CTAs in the cluster.
    if constexpr (EnableTmaMulticast) {
      if (threadIdx.x == 0) {
        constexpr size_t kBytesA = sizeof(sA[0]);
        int unroll_n_valid = 0;
#pragma unroll
	        for (int u = 0; u < UnrollN; ++u) {
	          if ((tile_n + u) < n_tiles_group) {
	            ++unroll_n_valid;
	          }
	        }
	        const int unroll_n_tx = unroll_n_valid;
			        size_t kBytesB =
			            static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
			        if constexpr (CtaGroup == 2) {
			          if (cta2_partition_b_mode == 1) {
			            // Partitioned mode: each CTA rank fetches only N/2 rows for every active unrolled tile.
			            kBytesB =
			                static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N / 2) * static_cast<size_t>(K_TILE_BYTES);
			          }
			        }
	        if constexpr (UnrollN == 2) {
	          if constexpr (CtaGroup == 2) {
	            if (cta2_partition_b_mode != 1) {
	              // Unpartitioned UnrollN=2 path: one 256-row B transaction (2 tiles).
	              kBytesB = static_cast<size_t>(2) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
	            }
	          } else {
	            // cta_group::1 UnrollN=2 path: one 256-row B transaction (2 tiles).
	            kBytesB = static_cast<size_t>(2) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
	          }
	        }
	        size_t kBytesSFB = static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(SFB_TILE_BYTES);
			        if constexpr (UnrollN == 2) {
			          if constexpr (CtaGroup == 2) {
			            if (cta2_partition_b_mode != 1) {
		              // Unpartitioned UnrollN=2 path: one 256-row SFB transaction (2 tiles).
		              kBytesSFB = static_cast<size_t>(2) * static_cast<size_t>(SFB_TILE_BYTES);
		            }
		          } else {
	            kBytesSFB = static_cast<size_t>(2) * static_cast<size_t>(SFB_TILE_BYTES);
	          }
	        }
	        const size_t kBytesSF = static_cast<size_t>(SFA_TILE_BYTES) + kBytesSFB;
        tok = cuda::device::barrier_arrive_tx(*bar, 1, kBytesA + kBytesB + kBytesSF);
      } else {
        // In multicast mode, all threads join this barrier generation before
        // rank0 issues cp.async.bulk.tensor.cluster multicast.
        tok = bar->arrive();
      }
	    } else if (threadIdx.x == 0) {
	      constexpr size_t kBytesA = sizeof(sA[0]);
	      int unroll_n_valid = 0;
#pragma unroll
	      for (int u = 0; u < UnrollN; ++u) {
	        if ((tile_n + u) < n_tiles_group) {
	          ++unroll_n_valid;
	        }
	      }
	      const int unroll_n_tx = unroll_n_valid;
			      size_t kBytesB =
			          static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
			      if constexpr (CtaGroup == 2) {
			        if (cta2_partition_b_mode == 1) {
			          kBytesB =
			              static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N / 2) * static_cast<size_t>(K_TILE_BYTES);
			        }
			      }
	      if constexpr (UnrollN == 2) {
	        if constexpr (CtaGroup == 2) {
	          if (cta2_partition_b_mode != 1) {
	            kBytesB = static_cast<size_t>(2) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
	          }
	        } else {
	          kBytesB = static_cast<size_t>(2) * static_cast<size_t>(TILE_N) * static_cast<size_t>(K_TILE_BYTES);
	        }
	      }
				      size_t kBytesSFB = static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(SFB_TILE_BYTES);
			      if constexpr (UnrollN == 2) {
			        if constexpr (CtaGroup == 2) {
			          if (cta2_partition_b_mode != 1) {
		            kBytesSFB = static_cast<size_t>(2) * static_cast<size_t>(SFB_TILE_BYTES);
		          }
		        } else {
	          kBytesSFB = static_cast<size_t>(2) * static_cast<size_t>(SFB_TILE_BYTES);
	        }
	      }
	      const size_t kBytesSF = static_cast<size_t>(SFA_TILE_BYTES) + kBytesSFB;
	      tok = cuda::device::barrier_arrive_tx(*bar, 1, kBytesA + kBytesB + kBytesSF);

      // For cta_group::2, CUTLASS partitions B across the two CTAs along N. In mode 1 we shift
      // the global N coordinate by N/2 so each CTA loads only one half. In other modes we keep the
      // global N coordinate and shift the shared-memory descriptor base per rank.
      //
      // Debug knob: allow disabling this shift to validate whether B is truly partitioned along N/2.
      const bool partition_b_global_shift = (CtaGroup == 2) && (cta2_partition_b_mode == 1);
      const int b_n_offset_base = partition_b_global_shift ? (n_offset + cluster_rank_b * (TILE_N / 2)) : n_offset;
      // In cta_group::2 mode1, B is partitioned by N/2 but SFB keeps the canonical tile_n row origin.
      const int sfb_row_offset_base = sfb_row_offset;
	      // B/SFB vary with tile_n, so always load them per-CTA.
	      if constexpr (UnrollN == 2) {
	        if constexpr (CtaGroup == 2) {
	          if (cta2_partition_b_mode == 1) {
#pragma unroll
	            for (int u = 0; u < UnrollN; ++u) {
	              if ((tile_n + u) >= n_tiles_group) {
	                continue;
	              }
	              const int b_n_offset = b_n_offset_base + u * TILE_N;
		              const int sfb_row_offset_u = sfb_row_offset_base + u * SFB_ROWS;
		              cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte * 2, b_n_offset, *bar);
		              cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
		                  &sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_u, *bar);
	            }
		          } else {
		            // UnrollN=2 cta_group::2 unpartitioned path: single 256-row B/SFB loads.
		            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
		                &sB[stage][0][0][0], b_desc, k_byte * 2, b_n_offset_base, *bar);
		            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
		                &sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
		          }
		        } else {
		          // cta_group::1 UnrollN=2 path: single 256-row B/SFB loads.
		          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
	              &sB[stage][0][0][0], b_desc, k_byte * 2, b_n_offset_base, *bar);
	          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
	              &sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
	        }
	      } else {
#pragma unroll
	        for (int u = 0; u < UnrollN; ++u) {
	          if ((tile_n + u) >= n_tiles_group) {
	            continue;
	          }
	          const int b_n_offset = b_n_offset_base + u * TILE_N;
	          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte * 2, b_n_offset, *bar);
	          if constexpr (UnrollN == 1) {
	            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
	                &sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
	          }
	        }
	      }
	      if (!use_tma_multicast) {
	        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sA[stage][0][0], a_desc, k_byte * 2, m_offset, *bar);
	        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFA[stage][0][0], sfa_desc, /*x=*/0, sfa_row_offset, *bar);
      }

      // NOTE: cp.async.bulk.tensor completion is tracked via the mbarrier; no commit_group is required.
    }

    // TMA multicast requires all CTAs to have joined the barrier generation before the
    // leader issues the multicast op, otherwise some CTAs can miss completions.
    // See `ch10/tma_multicast_cluster.cu` for the required sync pattern.
    if constexpr (EnableTmaMulticast) {
      __syncthreads();
      cg::this_cluster().sync();
    }

    if constexpr (EnableTmaMulticast) {
      if (cluster_rank == 0 && threadIdx.x == 0) {
      // Coordinates are in the descriptor's dimension order. A is encoded as 16U4 elements
      // (2 fp4 values/byte), so x is in fp4 elements, not bytes.
      const int coords_a[2] = {k_byte * 2, m_offset};
      if constexpr (NVFP4_GROUP_GEMM_MULTICAST_A != 0) {
        cptx::cp_async_bulk_tensor(cptx::space_cluster,
                                   cptx::space_global,
                                   &sA[stage][0][0],
                                   a_desc,
                                   coords_a,
                                   cuda::device::barrier_native_handle(*bar),
                                   tma_multicast_mask);
      }

      // For packed SFA: (x=0, y=row_offset).
      const int coords_sfa[2] = {0, sfa_row_offset};
      if constexpr (NVFP4_GROUP_GEMM_MULTICAST_SFA != 0) {
        cptx::cp_async_bulk_tensor(cptx::space_cluster,
                                   cptx::space_global,
                                   &sSFA[stage][0][0],
                                   sfa_desc,
                                   coords_sfa,
                                   cuda::device::barrier_native_handle(*bar),
                                   tma_multicast_mask);
      }
    }
    }

    // Load B/SFB per-CTA after all CTAs have joined the barrier generation and the leader has
    // issued the multicast. This matches the reference sync pattern and avoids missed completions.
    if constexpr (EnableTmaMulticast) {
      if (threadIdx.x == 0) {
        // Optional debug: if a given operand isn't multicast, fall back to per-CTA TMA loads.
        if constexpr (NVFP4_GROUP_GEMM_MULTICAST_A == 0) {
          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sA[stage][0][0], a_desc, k_byte * 2, m_offset, *bar);
        }
        if constexpr (NVFP4_GROUP_GEMM_MULTICAST_SFA == 0) {
          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFA[stage][0][0], sfa_desc, /*x=*/0, sfa_row_offset, *bar);
        }

        // For cta_group::2, CUTLASS partitions B across the two CTAs along N. In mode 1 we shift
        // the global N coordinate by N/2 so each CTA loads only one half. In other modes we keep the
        // global N coordinate and shift the shared-memory descriptor base per rank.
        const bool partition_b_global_shift = (CtaGroup == 2) && (cta2_partition_b_mode == 1);
        const int b_n_offset_base = partition_b_global_shift ? (n_offset + cluster_rank_b * (TILE_N / 2)) : n_offset;
        // In cta_group::2 mode1, B is partitioned by N/2 but SFB keeps the canonical tile_n row origin.
        const int sfb_row_offset_base = sfb_row_offset;
	        // B/SFB vary with tile_n, so always load them per-CTA.
	        if constexpr (UnrollN == 2) {
	          if constexpr (CtaGroup == 2) {
	            if (cta2_partition_b_mode == 1) {
#pragma unroll
		              for (int u = 0; u < UnrollN; ++u) {
			                if ((tile_n + u) >= n_tiles_group) {
			                  continue;
			                }
			                const int b_n_offset = b_n_offset_base + u * TILE_N;
				                const int sfb_row_offset_u = sfb_row_offset_base + u * SFB_ROWS;
				                cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte * 2, b_n_offset, *bar);
				                cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
				                    &sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_u, *bar);
			              }
			        } else {
			          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
			              &sB[stage][0][0][0], b_desc, k_byte * 2, b_n_offset_base, *bar);
			          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
			              &sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
			        }
		        } else {
		          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
		              &sB[stage][0][0][0], b_desc, k_byte * 2, b_n_offset_base, *bar);
		          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
		              &sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
		        }
	        } else {
#pragma unroll
	          for (int u = 0; u < UnrollN; ++u) {
	            if ((tile_n + u) >= n_tiles_group) {
	              continue;
	            }
	            const int b_n_offset = b_n_offset_base + u * TILE_N;
	            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte * 2, b_n_offset, *bar);
	            if constexpr (UnrollN == 1) {
	              cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
	                  &sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
	            }
	          }
	        }
	      }
	    }
	    return tok;
	  };

  // TMA helper (ops-only): issue the async TMA copies for a tile assuming the barrier generation
  // has already been armed via `barrier_arrive_tx` by another thread (warp-specialized producer).
		  auto issue_tma_tile_ops_only = [&](int stage, int k_byte, int sfa_row_offset, int sfb_row_offset) -> void {
		    block_barrier* bar = bars[stage];

    const bool partition_b_global_shift = (CtaGroup == 2) && (cta2_partition_b_mode == 1);
    const int b_n_offset_base = partition_b_global_shift ? (n_offset + cluster_rank_b * (TILE_N / 2)) : n_offset;
    // In cta_group::2 mode1, B is partitioned by N/2 but SFB keeps the canonical tile_n row origin.
    const int sfb_row_offset_base = sfb_row_offset;

		    if constexpr (UnrollN == 2) {
		      if constexpr (CtaGroup == 2) {
		        if (cta2_partition_b_mode == 1) {
#pragma unroll
			          for (int u = 0; u < UnrollN; ++u) {
			            if ((tile_n + u) >= n_tiles_group) {
			              continue;
			            }
			            const int b_n_offset = b_n_offset_base + u * TILE_N;
				            const int sfb_row_offset_u = sfb_row_offset_base + u * SFB_ROWS;
				            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte * 2, b_n_offset, *bar);
				            cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
				                &sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_u, *bar);
				          }
			        } else {
			          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
			              &sB[stage][0][0][0], b_desc, k_byte * 2, b_n_offset_base, *bar);
			          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
			              &sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
			        }
		      } else {
		        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
		            &sB[stage][0][0][0], b_desc, k_byte * 2, b_n_offset_base, *bar);
	        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
	            &sSFB[stage][0][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
	      }
	    } else {
#pragma unroll
	      for (int u = 0; u < UnrollN; ++u) {
	        if ((tile_n + u) >= n_tiles_group) {
	          continue;
	        }
	        const int b_n_offset = b_n_offset_base + u * TILE_N;
	        cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sB[stage][u][0][0], b_desc, k_byte * 2, b_n_offset, *bar);
	        if constexpr (UnrollN == 1) {
	          cuda_device::cp_async_bulk_tensor_2d_global_to_shared(
	              &sSFB[stage][u][0][0], sfb_desc, /*x=*/0, sfb_row_offset_base, *bar);
	        }
	      }
	    }
	    // A/SFA are always loaded per-CTA in the non-multicast path.
	    cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sA[stage][0][0], a_desc, k_byte * 2, m_offset, *bar);
	    cuda_device::cp_async_bulk_tensor_2d_global_to_shared(&sSFA[stage][0][0], sfa_desc, /*x=*/0, sfa_row_offset, *bar);
	  };

  auto wait_tma_tile = [&](int stage, block_barrier::arrival_token& tok) {
    block_barrier* bar = bars[stage];
    if (threadIdx.x == 0) {
      // Wait for the outstanding TMA transactions associated with this barrier generation.
      bar->wait(std::move(tok));
    }
    __syncthreads();
    if constexpr (CtaGroup == 2) {
      cg::this_cluster().sync();
    }
  };

  // Warp-specialized TMA wait for the common case (cta_group::1, no cluster multicast):
  // thread0 performs the barrier wait, then we warp-sync to keep warp0 from racing ahead.
  auto wait_tma_tile_warp0 = [&](int stage, block_barrier::arrival_token& tok) {
    block_barrier* bar = bars[stage];
    if (threadIdx.x == 0) {
      bar->wait(std::move(tok));
    }
    __syncwarp();
  };

	  const int warp = static_cast<int>(threadIdx.x) >> 5;
	  const int lane = static_cast<int>(threadIdx.x) & 31;

  // SFB row addressing (global tensor coordinates):
  // - Canonical N-major: [n_tiles, k_tiles, 128, 16]
  //     row_offset = (tile_n * k_tiles_total + k_tile_idx) * 128
  // - UnrollN=2 cta_group::1 fast path uses K-major packing so (tile_n, tile_n+1) are contiguous
  //   for a fixed k_tile:
  //     [k_tiles, n_tiles_tma, 128, 16], row_offset = (k_tile_idx * n_tiles_tma + tile_n) * 128
  // - UnrollN=2 keeps K-major packing for SFB in both cta_group::1 and cta_group::2.
  const int n_tiles_tma = (UnrollN == 2) ? ((n_tiles_group + 1) & ~1) : n_tiles_group;
  auto sfb_row_offset_for = [&](int tile_n_start, int k_tile_idx) -> int {
    if constexpr (UnrollN == 2) {
      return (k_tile_idx * n_tiles_tma + tile_n_start) * SFB_ROWS;
    } else {
      return (tile_n_start * k_tiles_total + k_tile_idx) * SFB_ROWS;
    }
  };

  // SFA row addressing (global tensor coordinates): [m_tiles, k_tiles, 128, 16].
  auto sfa_row_offset_for = [&](int k_tile_idx) -> int {
    return (sfa_tile_m * k_tiles_total + k_tile_idx) * SFA_ROWS;
  };

	  // Full-CTA bring-up path (used for cta_group::2 and for optional TMA multicast mode).
	  auto run_full_cta_mainloop = [&]() {
	    if constexpr (PIPELINE_STAGES == 1) {
	      // Sequential K-tile loop (1 stage): load -> scales -> MMA.
      for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
	        const int k_byte = k_tile_idx * K_TILE_BYTES;
	        const int sfa_row_offset = sfa_row_offset_for(k_tile_idx);
	        const int sfb_row_offset = sfb_row_offset_for(tile_n, k_tile_idx);
	        auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
	        wait_tma_tile(/*stage=*/0, tok0);
        if (debug_print_ptrs != 0 && k_tile_idx == 0 && group_idx == 0 && tile_m == 0 && tile_n == 0 &&
            threadIdx.x == 0) {
          const uint8_t b00 = sB[0][0][0][0];
          const uint8_t b01 = sB[0][0][0][1];
          const uint8_t b10 = (UnrollN == 2) ? sB[0][1][0][0] : 0u;
          const uint8_t b11 = (UnrollN == 2) ? sB[0][1][0][1] : 0u;
          const uint8_t sf00 = sSFB[0][0][0][0];
          const uint8_t sf01 = sSFB[0][0][0][1];
          const uint8_t sf10 = (UnrollN == 2) ? sSFB[0][1][0][0] : 0u;
          const uint8_t sf11 = (UnrollN == 2) ? sSFB[0][1][0][1] : 0u;
          printf("dbg_tma rank=%d b_u0=[%u,%u] b_u1=[%u,%u] sfb_u0=[%u,%u] sfb_u1=[%u,%u]\\n",
                 cluster_rank, (unsigned)b00, (unsigned)b01, (unsigned)b10, (unsigned)b11,
                 (unsigned)sf00, (unsigned)sf01, (unsigned)sf10, (unsigned)sf11);
        }

        if constexpr (DEBUG_STAGE == 2) {
          break;
        }

        if constexpr (WS_SEGMENT_PARALLEL) {
          if (lane == 0) {
            const int seg = warp;
            if (seg < 4) {
              constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
              const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
              const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][0]);
              tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
                  desc_sfa_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP), tmem_sfa_seg[seg]);
              tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
                  desc_sfb_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP), tmem_sfb_seg[0][seg]);
            }
          }
        } else {
          // UTCCP scale copies:
          // - cta_group::1: issue from a single elected thread (CUTLASS-style).
          // - cta_group::2: `tcgen05.cp.*.warpx4` behaves like a warpgroup collective; issuing it from
          //   a single lane can trap. Execute the UTCCP instructions from the full CTA (128 threads).
          if constexpr (CtaGroup == 2) {
            const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
            copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
            for (int u = 0; u < UnrollN; ++u) {
              if ((tile_n + u) >= n_tiles_group) {
                continue;
              }
              const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][u]);
              const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
              copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
            }
          } else {
            if (threadIdx.x == 0) {
              const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
              copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
              for (int u = 0; u < UnrollN; ++u) {
                if ((tile_n + u) >= n_tiles_group) {
                  continue;
                }
                const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][u]);
                const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
                copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
              }
            }
          }
          __syncthreads();
          if constexpr (CtaGroup == 2) {
            cg::this_cluster().sync();
          }
        }
        // NOTE: CUTLASS typically does not issue an explicit `tcgen05.wait::st` fence after UTCCP
        // scale copies and relies on UMMA stalling as needed. In this bring-up kernel on B200,
        // we've observed NaN accumulators when UMMA appears to race TMEM stores, so we optionally
        // fence UTCCP stores before issuing MMA for cta_group::2.
        if constexpr (CtaGroup == 2 && (NVFP4_GROUP_GEMM_CTA2_WAIT_AFTER_UTCCP != 0)) {
          tcgen05::tmem_wait_st_sync();
          tcgen05::tmem_wait_ld_sync();
        }

        if constexpr (DEBUG_STAGE == 3) {
          // DEBUG_STAGE=3 exits immediately after issuing UTCCP scale copies (no MMA/epilogue).
          // Ensure outstanding TMEM stores complete before the kernel proceeds to TMEM dealloc,
          // otherwise SM100a can fault (observed as `cudaErrorMisalignedAddress`).
          tcgen05::tmem_wait_st_sync();
          tcgen05::tmem_wait_ld_sync();
          break;
        }

        if constexpr (WS_SEGMENT_PARALLEL) {
          if (lane == 0) {
            const int seg = warp;
            if (seg < 4) {
              if (k_tile_idx == 0 && seg != 0) {
                while (ws_seg0_done_tile < 0) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
                  __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
                }
              }
              const uint64_t desc_a =
                  static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              const uint64_t desc_b =
                  static_cast<uint64_t>(desc_b_base[0][0]) + cta2_desc_b_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
              tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                  desc_a,
                  desc_b,
                  tmem_c_tiles[0],
                  accumulate,
                  idesc_hi_seg[0][seg],
                  tmem_sfa_seg[seg],
                  tmem_sfb_seg[0][seg]);
              if (k_tile_idx == 0 && seg == 0) {
                ws_seg0_done_tile = 0;
                __threadfence_block();
              }
            }
          }
        } else if constexpr (CTA2_MMA_ALL_THREADS) {
          // cta_group::2 bring-up: issue UMMA from the full CTA (all 128 threads).
          // This is a correctness probe: if `tcgen05.mma.cta_group::2` is a warpgroup collective,
          // issuing it from a single lane yields undefined accumulator contents (NaNs/Inf).
#pragma unroll
          for (int u = 0; u < UnrollN; ++u) {
            if ((tile_n + u) >= n_tiles_group) {
              continue;
            }
#pragma unroll
            for (int seg = 0; seg < 4; ++seg) {
              const uint64_t desc_a =
                  static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              const uint64_t desc_b =
                  static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
              const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
              tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                  desc_a,
                  desc_b,
                  tmem_c_tiles[u],
                  accumulate,
                  idesc_hi_seg[u][seg],
                  tmem_sfa_seg[seg],
                  tmem_sfb_seg[u][seg]);
            }
          }
        } else if constexpr (CTA2_MMA_SEGMENT_PARALLEL) {
          // cta_group::2 bring-up: segment-parallel UMMA schedule.
          // One warp issues one K64 segment to avoid over-issuing tcgen05.mma across all warps.
          if (lane == 0) {
            const int seg = warp;
            if (seg < 4) {
              if (k_tile_idx == 0 && seg != 0) {
                while (ws_seg0_done_tile < 0) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
                  __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
                }
              }
#pragma unroll
              for (int u = 0; u < UnrollN; ++u) {
                if ((tile_n + u) >= n_tiles_group) {
                  continue;
                }
                const uint64_t desc_a =
                    static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                const uint64_t desc_b =
                    static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                    desc_a,
                    desc_b,
                    tmem_c_tiles[u],
                    accumulate,
                    idesc_hi_seg[u][seg],
                    tmem_sfa_seg[seg],
                    tmem_sfb_seg[u][seg]);
              }
              if (k_tile_idx == 0 && seg == 0) {
                ws_seg0_done_tile = 0;
                __threadfence_block();
              }
            }
          }
        } else {
		          if constexpr (NVFP4_GROUP_GEMM_MMA_LANE0_ALL_WARPS != 0) {
		            if (lane == 0) {
#pragma unroll
		              for (int u = 0; u < UnrollN; ++u) {
		                if ((tile_n + u) >= n_tiles_group) {
		                  continue;
		                }
#pragma unroll
	                for (int seg = 0; seg < 4; ++seg) {
	                  const uint64_t desc_a =
	                      static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
	                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	                  const uint64_t desc_b =
	                      static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
	                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	                  const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
	                  tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
	                      desc_a,
	                      desc_b,
	                      tmem_c_tiles[u],
	                      accumulate,
	                      idesc_hi_seg[u][seg],
	                      tmem_sfa_seg[seg],
	                      tmem_sfb_seg[u][seg]);
	                }
	              }
	            }
		          } else {
		            if (threadIdx.x == 0) {
#pragma unroll
	            for (int u = 0; u < UnrollN; ++u) {
	              if ((tile_n + u) >= n_tiles_group) {
	                continue;
	              }
#pragma unroll
              for (int seg = 0; seg < 4; ++seg) {
                const uint64_t desc_a =
                    static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                const uint64_t desc_b =
                    static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                    desc_a,
                    desc_b,
                    tmem_c_tiles[u],
                    accumulate,
                    idesc_hi_seg[u][seg],
                    tmem_sfa_seg[seg],
                    tmem_sfb_seg[u][seg]);
              }
            }
	            }
	          }
	        }

        if constexpr (DEBUG_STAGE == 4) {
          break;
        }
      }
      return;
    }

    // Prologue: preload the first K tile into stage0.
	    if (k_tiles_total > 0) {
	      const int k_tile_idx = 0;
	      const int k_byte = 0;
	      const int sfa_row_offset = sfa_row_offset_for(k_tile_idx);
	      const int sfb_row_offset = sfb_row_offset_for(tile_n, k_tile_idx);
	      auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
	      wait_tma_tile(/*stage=*/0, tok0);
      if (debug_print_ptrs != 0 && group_idx == 0 && tile_m == 0 && tile_n == 0 && threadIdx.x == 0) {
        const uint8_t b00 = sB[0][0][0][0];
        const uint8_t b01 = sB[0][0][0][1];
        const uint8_t b10 = (UnrollN == 2) ? sB[0][1][0][0] : 0u;
        const uint8_t b11 = (UnrollN == 2) ? sB[0][1][0][1] : 0u;
        const uint8_t sf00 = sSFB[0][0][0][0];
        const uint8_t sf01 = sSFB[0][0][0][1];
        const uint8_t sf10 = (UnrollN == 2) ? sSFB[0][1][0][0] : 0u;
        const uint8_t sf11 = (UnrollN == 2) ? sSFB[0][1][0][1] : 0u;
        printf("dbg_tma rank=%d b_u0=[%u,%u] b_u1=[%u,%u] sfb_u0=[%u,%u] sfb_u1=[%u,%u]\\n",
               cluster_rank, (unsigned)b00, (unsigned)b01, (unsigned)b10, (unsigned)b11,
               (unsigned)sf00, (unsigned)sf01, (unsigned)sf10, (unsigned)sf11);
      }
	    }

    // Mainloop: iterate over K tiles with a circular shared-memory pipeline.
    for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
      const int stage_cur = k_tile_idx % PIPELINE_STAGES;
      const int stage_next = (k_tile_idx + 1) % PIPELINE_STAGES;
      const int k_byte = k_tile_idx * K_TILE_BYTES;

      if constexpr (DEBUG_STAGE == 2) {
        break;
      }

      // Prefetch the next K tile into the next stage (overlaps with UTCCP+MMA of the current tile).
      const bool has_next = (k_tile_idx + 1) < k_tiles_total;
      block_barrier::arrival_token tok_next;
	      if (has_next) {
	        const int next_tile = k_tile_idx + 1;
	        const int next_k_byte = next_tile * K_TILE_BYTES;
	        const int sfa_row_offset_next = sfa_row_offset_for(next_tile);
	        const int sfb_row_offset_next = sfb_row_offset_for(tile_n, next_tile);
	        tok_next = issue_tma_tile(stage_next, next_k_byte, sfa_row_offset_next, sfb_row_offset_next);
	      }

	      if constexpr (WS_SEGMENT_PARALLEL) {
	        if (lane == 0) {
	          const int seg = warp;
	          if (seg < 4) {
	            constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
	            const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[stage_cur]) + cta2_desc_sfa_row_offset;
	            const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[stage_cur][0]);
	            tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
	                desc_sfa_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP), tmem_sfa_seg[seg]);
	            tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
	                desc_sfb_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP), tmem_sfb_seg[0][seg]);
	          }
	        }
      } else {
        // UTCCP scale copies:
        // - cta_group::1: issue from a single elected thread.
        // - cta_group::2: execute UTCCP from the full CTA (warpgroup collective semantics).
        if constexpr (CtaGroup == 2) {
          const uint64_t desc_sfa_base =
              static_cast<uint64_t>(desc_sfa[stage_cur]) + cta2_desc_sfa_row_offset;
          copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
          for (int u = 0; u < UnrollN; ++u) {
            if ((tile_n + u) >= n_tiles_group) {
              continue;
            }
            const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[stage_cur][u]);
            const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
            copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
          }
        } else {
          if (threadIdx.x == 0) {
            const uint64_t desc_sfa_base =
                static_cast<uint64_t>(desc_sfa[stage_cur]) + cta2_desc_sfa_row_offset;
            copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
            for (int u = 0; u < UnrollN; ++u) {
              if ((tile_n + u) >= n_tiles_group) {
                continue;
              }
              // Only skip u=1 in the warp0-only UnrollN=1 scheduling path where a helper warp owns u=1.
              // In the active full-CTA UnrollN=2 path, dropping u=1 here corrupts the second N128 tile.
              if constexpr (WS_UNROLL2_MMA && (NVFP4_GROUP_GEMM_WARP0_ONLY_MAINLOOP != 0) && (UnrollN == 1)) {
                if (ws_u1_active && u == 1) {
                  continue;
                }
              }
              const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[stage_cur][u]);
              const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
              copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
            }
          }
        }

        __syncthreads();
        if constexpr (CtaGroup == 2) {
          cg::this_cluster().sync();
	        }
	      }

      // UTCCP copies write TMEM asynchronously. For cta_group::2 bring-up, we fence those stores
      // before the subsequent MMA reads scale-factor fragments (can be disabled once verified).
      if constexpr (CtaGroup == 2 && (NVFP4_GROUP_GEMM_CTA2_WAIT_AFTER_UTCCP != 0)) {
        tcgen05::tmem_wait_st_sync();
        tcgen05::tmem_wait_ld_sync();
      }

      if constexpr (DEBUG_STAGE == 3) {
        // See note above: DEBUG_STAGE=3 must wait for UTCCP stores before early-exit.
        tcgen05::tmem_wait_st_sync();
        tcgen05::tmem_wait_ld_sync();
        break;
      }

	      if constexpr (WS_SEGMENT_PARALLEL) {
	        if (lane == 0) {
	          const int seg = warp;
	          if (seg < 4) {
	            if (k_tile_idx == 0 && seg != 0) {
	              while (ws_seg0_done_tile < 0) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
	                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
	              }
	            }
	            const uint64_t desc_a =
	                static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
	                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	            const uint64_t desc_b =
	                static_cast<uint64_t>(desc_b_base[stage_cur][0]) + cta2_desc_b_row_offset +
	                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	            const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
	            tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
	                desc_a,
	                desc_b,
	                tmem_c_tiles[0],
	                accumulate,
	                idesc_hi_seg[0][seg],
	                tmem_sfa_seg[seg],
	                tmem_sfb_seg[0][seg]);
	            if (k_tile_idx == 0 && seg == 0) {
	              ws_seg0_done_tile = 0;
	              __threadfence_block();
	            }
	          }
	        }
		      } else if constexpr (CTA2_MMA_ALL_THREADS) {
		        // cta_group::2 bring-up: issue UMMA from the full CTA (all 128 threads).
		        // This is a correctness probe: if `tcgen05.mma.cta_group::2` is a warpgroup collective,
		        // issuing it from a single lane yields undefined accumulator contents (NaNs/Inf).
#pragma unroll
		        for (int u = 0; u < UnrollN; ++u) {
		          if ((tile_n + u) >= n_tiles_group) {
		            continue;
		          }
#pragma unroll
		          for (int seg = 0; seg < 4; ++seg) {
		            const uint64_t desc_a =
		                static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
		                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
		            const uint64_t desc_b =
		                static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
		                static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
		            const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
		            tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
		                desc_a,
		                desc_b,
		                tmem_c_tiles[u],
		                accumulate,
		                idesc_hi_seg[u][seg],
		                tmem_sfa_seg[seg],
		                tmem_sfb_seg[u][seg]);
		          }
		        }
		      } else if constexpr (CTA2_MMA_SEGMENT_PARALLEL) {
		        if (lane == 0) {
		          const int seg = warp;
		          if (seg < 4) {
		            if (k_tile_idx == 0 && seg != 0) {
		              while (ws_seg0_done_tile < 0) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
	                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
	              }
	            }
#pragma unroll
	            for (int u = 0; u < UnrollN; ++u) {
	              if ((tile_n + u) >= n_tiles_group) {
	                continue;
	              }
	              const uint64_t desc_a =
	                  static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
	                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	              const uint64_t desc_b =
	                  static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
	                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	              const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
	              tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
	                  desc_a,
	                  desc_b,
	                  tmem_c_tiles[u],
	                  accumulate,
	                  idesc_hi_seg[u][seg],
	                  tmem_sfa_seg[seg],
	                  tmem_sfb_seg[u][seg]);
	            }
	            if (k_tile_idx == 0 && seg == 0) {
	              ws_seg0_done_tile = 0;
	              __threadfence_block();
	            }
	          }
	        }
	      } else {
			        if constexpr (NVFP4_GROUP_GEMM_MMA_LANE0_ALL_WARPS != 0) {
			          if (lane == 0) {
#pragma unroll
			            for (int u = 0; u < UnrollN; ++u) {
			              if ((tile_n + u) >= n_tiles_group) {
			                continue;
			              }
#pragma unroll
		              for (int seg = 0; seg < 4; ++seg) {
		                const uint64_t desc_a =
		                    static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
		                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
		                const uint64_t desc_b =
		                    static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
		                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));

		                // accumulate=0 for first segment of first tile, else accumulate=1.
		                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
		                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
		                    desc_a,
		                    desc_b,
		                    tmem_c_tiles[u],
		                    accumulate,
		                    idesc_hi_seg[u][seg],
		                    tmem_sfa_seg[seg],
		                    tmem_sfb_seg[u][seg]);
		              }
		            }
		          }
			        } else {
			          if (threadIdx.x == 0) {
#pragma unroll
			          for (int u = 0; u < UnrollN; ++u) {
			            if ((tile_n + u) >= n_tiles_group) {
		              continue;
		            }
#pragma unroll
	            for (int seg = 0; seg < 4; ++seg) {
	              const uint64_t desc_a =
	                  static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
	                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	              const uint64_t desc_b =
	                  static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
	                  static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));

	              // accumulate=0 for first segment of first tile, else accumulate=1.
	              const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
	              tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
	                  desc_a,
	                  desc_b,
	                  tmem_c_tiles[u],
	                  accumulate,
		                  idesc_hi_seg[u][seg],
		                  tmem_sfa_seg[seg],
		                  tmem_sfb_seg[u][seg]);
		            }
		          }
		          }
		        }
		      }

      if constexpr (DEBUG_STAGE == 4) {
        break;
      }

      // Ensure the next stage is resident in shared memory before the loop advances.
      if (has_next) {
        wait_tma_tile(stage_next, tok_next);
      } else {
        __syncthreads();
        if constexpr (CtaGroup == 2) {
          cg::this_cluster().sync();
        }
      }
    }
  };

			  if constexpr (CtaGroup == 1) {
			    if constexpr (!EnableTmaMulticast && (NVFP4_GROUP_GEMM_WARP0_ONLY_MAINLOOP != 0) && (UnrollN == 1)) {
	      // Fast path: warp0 runs the mainloop; other warps stay idle until epilogue.
	      // This avoids per-K-tile CTA-wide synchronization overhead in the common (non-multicast) mode.
	    if constexpr (PIPELINE_STAGES == 1) {
	      // Single-stage bring-up: keep the simple warp0-only mainloop.
	      if (warp == 0) {
        if constexpr (STAGE1_PREFETCH) {
          // Stage-1 pipelined mainloop: issue TMA for tile k+1 after issuing UMMA for tile k.
          // This assumes UMMA consumes shared memory at issue time (matching CUTLASS' num_ab_stage=1 intent).
	          if (k_tiles_total > 0) {
	            const int k_tile0 = 0;
	            const int k_byte0 = 0;
		            const int sfa_row_offset0 = sfa_row_offset_for(k_tile0);
		            const int sfb_row_offset0 = sfb_row_offset_for(tile_n, k_tile0);
		            block_barrier::arrival_token tok_cur = issue_tma_tile(/*stage=*/0, k_byte0, sfa_row_offset0, sfb_row_offset0);

            for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
              wait_tma_tile_warp0(/*stage=*/0, tok_cur);

              if constexpr (DEBUG_STAGE == 2) {
                break;
              }

              const int k_byte = k_tile_idx * K_TILE_BYTES;
              if (lane == 0) {
                const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
	                copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
	                for (int u = 0; u < UnrollN; ++u) {
	                  if ((tile_n + u) >= n_tiles_group) {
	                    continue;
	                  }
	                  const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][u]);
	                  const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
	                  copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
	                }
	                if constexpr (DEBUG_STAGE != 3) {
#pragma unroll
	                  for (int seg = 0; seg < 4; ++seg) {
	                    const uint64_t desc_a =
	                        static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
	                        static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	                    const uint32_t accumulate = (k_byte == 0 && seg == 0) ? 0u : 1u;
#pragma unroll
	                    for (int u = 0; u < UnrollN; ++u) {
	                      if ((tile_n + u) >= n_tiles_group) {
	                        continue;
	                      }
                      const uint64_t desc_b =
                          static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
                          static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                      tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                          desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg], tmem_sfa_seg[seg],
                          tmem_sfb_seg[u][seg]);
                    }
                  }
                }
              }

              if constexpr (DEBUG_STAGE == 3 || DEBUG_STAGE == 4) {
                break;
              }

              const bool has_next = (k_tile_idx + 1) < k_tiles_total;
	              if (has_next) {
	                const int next_tile = k_tile_idx + 1;
	                const int next_k_byte = next_tile * K_TILE_BYTES;
		                const int sfa_row_offset_next = sfa_row_offset_for(next_tile);
		                const int sfb_row_offset_next = sfb_row_offset_for(tile_n, next_tile);
		                tok_cur = issue_tma_tile(/*stage=*/0, next_k_byte, sfa_row_offset_next, sfb_row_offset_next);
	              } else {
	                __syncwarp();
	              }
            }
          }
        } else {
          for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
            // Ensure warp1 has finished using the shared-memory stage before we overwrite it
            // (PIPELINE_STAGES==1 reuses the same stage buffer every iteration).
            if constexpr (WS_UNROLL2_MMA) {
              if (lane == 0 && ws_u1_active && k_tile_idx > 0) {
                while (ws_u1_done_tile < (k_tile_idx - 1)) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
                  __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
                }
              }
            }
            if constexpr (WS_SPLIT_U0_SEGS) {
              if (lane == 0 && k_tile_idx > 0) {
                while (ws_u0_done_tile < (k_tile_idx - 1)) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
                  __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
                }
              }
            }
	            const int k_byte = k_tile_idx * K_TILE_BYTES;
		            const int sfa_row_offset = sfa_row_offset_for(k_tile_idx);
		            const int sfb_row_offset = sfb_row_offset_for(tile_n, k_tile_idx);
		            auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
	            wait_tma_tile_warp0(/*stage=*/0, tok0);

            if constexpr (DEBUG_STAGE == 2) {
              break;
            }

		            if (lane == 0) {
		              const uint64_t desc_sfa_base = static_cast<uint64_t>(desc_sfa[0]) + cta2_desc_sfa_row_offset;
		              // Scale-copy schedule:
		              // - Always copy SFA in warp0 (shared across u=0/u=1 MMAs).
		              // - For WS_UNROLL2_MMA, warp0 signals warp1 after SFA is ready; warp1 copies SFB(u=1).
		              copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
		              if constexpr (WS_UNROLL2_MMA) {
		                if (ws_u1_active) {
		                  // Signal warp1 that TMA is complete and SFA is resident in TMEM for this K tile.
		                  ws_scales_ready_tile = k_tile_idx;
		                }
		                // Warp0 copies u=0's SFB tile.
		                const uint64_t desc_sfb0_base = static_cast<uint64_t>(desc_sfb[0][0]);
		                const uint32_t* tmem_sfb_ptrs_u0 = tmem_sfb_ptrs + 0 * 4;
		                copy_scale_fragments(desc_sfb0_base, tmem_sfb_ptrs_u0, /*is_sfb=*/true);
		              } else {
		                for (int u = 0; u < UnrollN; ++u) {
		                  if ((tile_n + u) >= n_tiles_group) {
	                    continue;
                  }
                  const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[0][u]);
                  const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
	                  copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
	                }
	              }
		              if constexpr (DEBUG_STAGE != 3) {
		                if constexpr (WS_SPLIT_U0_SEGS) {
		                  // Split u=0 MMA issue: warp0 issues seg0, warp2 issues seg1..3.
	                  // seg0 must execute first when accumulate=0 (k_tile_idx==0), otherwise
                  // later segments would accumulate into uninitialized accumulators.
                  if ((tile_n + 0) < n_tiles_group) {
                    constexpr int seg = 0;
                    constexpr int u = 0;
                    const uint64_t desc_a =
                        static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                        static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                    const uint64_t desc_b =
                        static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
                        static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                    const uint32_t accumulate = (k_byte == 0) ? 0u : 1u;
                    tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                        desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg], tmem_sfa_seg[seg],
                        tmem_sfb_seg[u][seg]);
                  }
	                  // Signal warp2 that seg0 has been issued for this K tile.
	                  ws_u0_ready_tile = k_tile_idx;
	                } else {
#pragma unroll
	                  for (int seg = 0; seg < 4; ++seg) {
                    const uint64_t desc_a =
                        static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
                        static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
	                    const uint32_t accumulate = (k_byte == 0 && seg == 0) ? 0u : 1u;
#pragma unroll
	                    for (int u = 0; u < UnrollN; ++u) {
	                      if ((tile_n + u) >= n_tiles_group) {
	                        continue;
	                      }
                      if constexpr (WS_UNROLL2_MMA) {
                        if (ws_u1_active && u == 1) {
                          continue;
                        }
                      }
                      const uint64_t desc_b =
                          static_cast<uint64_t>(desc_b_base[0][u]) + cta2_desc_b_row_offset +
                          static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
                      tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
                          desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg], tmem_sfa_seg[seg],
                          tmem_sfb_seg[u][seg]);
                    }
                  }
                }
              }
              // See note above: avoid globally fencing TMEM stores after UTCCP scale copies.
            }

            if constexpr (DEBUG_STAGE == 3) {
              break;
            }

            if constexpr (DEBUG_STAGE == 4) {
              break;
            }
          }
        }
      }
	      if constexpr (WS_UNROLL2_MMA) {
		        if (warp == 1 && ws_u1_active && lane == 0) {
			          // Warp1 lane0 issues the u=1 MMAs after:
			          // - warp0 has completed TMA and copied SFA into TMEM for this K tile.
			          // Warp1 copies SFB(u=1) scales into TMEM, then issues UMMA.
		          uint32_t ws_tmem_sfa_seg[4];
		          uint32_t ws_tmem_sfb_seg[4];
		          uint32_t ws_idesc_hi_seg[4];
#pragma unroll
          for (int seg = 0; seg < 4; ++seg) {
            const uint32_t tmem_sfa = tmem_sfa_ptrs[seg];
            const uint32_t tmem_sfb = tmem_sfb_ptrs[1 * 4 + seg];
            ws_tmem_sfa_seg[seg] = tmem_sfa;
            ws_tmem_sfb_seg[seg] = tmem_sfb;
            const uint64_t idesc_runtime = umma::make_runtime_instr_desc_block_scaled(idesc, tmem_sfa, tmem_sfb);
            ws_idesc_hi_seg[seg] = static_cast<uint32_t>(idesc_runtime >> 32);
          }

					          for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
					            while (ws_scales_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
					              __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
					            }
					            // Copy SFB(u=1) scales for this tile from shared memory into TMEM.
					            const uint64_t desc_sfb1_base = static_cast<uint64_t>(desc_sfb[0][1]);
					            const uint32_t* tmem_sfb_ptrs_u1 = tmem_sfb_ptrs + 1 * 4;
					            if constexpr (WS_SFB1_SEGMENT_HELPERS) {
					              // Warp1 copies only seg0/seg1; helper warps copy seg2/seg3 in parallel.
					              constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
					              tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
					                  desc_sfb1_base + static_cast<uint64_t>(0 * SF_COPY_32x128B_DESC_STEP), tmem_sfb_ptrs_u1[0]);
					              tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
					                  desc_sfb1_base + static_cast<uint64_t>(1 * SF_COPY_32x128B_DESC_STEP), tmem_sfb_ptrs_u1[1]);
					              // Issue seg0/seg1 MMAs while helper warps copy seg2/seg3.
#pragma unroll
					              for (int seg = 0; seg < 2; ++seg) {
					                const uint64_t desc_a =
					                    static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
					                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                const uint64_t desc_b =
					                    static_cast<uint64_t>(desc_b_base[0][1]) + cta2_desc_b_row_offset +
					                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
					                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
					                    desc_a, desc_b, tmem_c_tiles[1], accumulate, ws_idesc_hi_seg[seg],
					                    ws_tmem_sfa_seg[seg], ws_tmem_sfb_seg[seg]);
					              }

					              // Wait for helper warps to finish copying seg2/seg3 scales for this tile.
					              while (ws_sfb1_seg2_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
					                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
					              }
					              while (ws_sfb1_seg3_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
					                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
					              }

#pragma unroll
					              for (int seg = 2; seg < 4; ++seg) {
					                const uint64_t desc_a =
					                    static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
					                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                const uint64_t desc_b =
					                    static_cast<uint64_t>(desc_b_base[0][1]) + cta2_desc_b_row_offset +
					                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                const uint32_t accumulate = 1u;
					                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
					                    desc_a, desc_b, tmem_c_tiles[1], accumulate, ws_idesc_hi_seg[seg],
					                    ws_tmem_sfa_seg[seg], ws_tmem_sfb_seg[seg]);
					              }
					            } else {
					              copy_scale_fragments(desc_sfb1_base, tmem_sfb_ptrs_u1, /*is_sfb=*/true);
#pragma unroll
					              for (int seg = 0; seg < 4; ++seg) {
					                const uint64_t desc_a =
					                    static_cast<uint64_t>(desc_a_base[0]) + cta2_desc_a_row_offset +
					                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                const uint64_t desc_b =
					                    static_cast<uint64_t>(desc_b_base[0][1]) + cta2_desc_b_row_offset +
					                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
					                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
					                    desc_a, desc_b, tmem_c_tiles[1], accumulate, ws_idesc_hi_seg[seg],
					                    ws_tmem_sfa_seg[seg], ws_tmem_sfb_seg[seg]);
					              }
					            }
				            ws_u1_done_tile = k_tile_idx;
				          }
				        }
			      }
			      if constexpr (WS_SFB1_SEGMENT_HELPERS) {
			        // Helper warps copy the remaining SFB(u=1) scale segments in parallel with warp1's MMAs.
			        if ((warp == 2 || warp == 3) && ws_u1_active && lane == 0) {
			          const int seg = (warp == 2) ? 2 : 3;
			          constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
				          for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
				            while (ws_scales_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
				              __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
				            }
				            const uint64_t desc_sfb1_base = static_cast<uint64_t>(desc_sfb[0][1]);
				            const uint64_t src_desc =
				                desc_sfb1_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
				            const uint32_t dst_addr = tmem_sfb_ptrs[1 * 4 + seg];
				            tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, dst_addr);
				            if (seg == 2) {
				              ws_sfb1_seg2_ready_tile = k_tile_idx;
				            } else {
				              ws_sfb1_seg3_ready_tile = k_tile_idx;
				            }
				            __threadfence_block();
				          }
				        }
				      }
			    } else {
			      if (warp == 0) {
			        // Warp0-only mainloop with a circular shared-memory pipeline.
	        // Prologue: preload the first K tile into stage0.
		        if (k_tiles_total > 0) {
		          const int k_tile_idx = 0;
			          const int k_byte = 0;
				          const int sfa_row_offset = sfa_row_offset_for(k_tile_idx);
				          const int sfb_row_offset = sfb_row_offset_for(tile_n, k_tile_idx);
				          auto tok0 = issue_tma_tile(/*stage=*/0, k_byte, sfa_row_offset, sfb_row_offset);
			          wait_tma_tile_warp0(/*stage=*/0, tok0);
		        }

	        // Mainloop: iterate over K tiles with the shared-memory pipeline.
		        for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
		          const int stage_cur = k_tile_idx % PIPELINE_STAGES;
		          const int stage_next = (k_tile_idx + 1) % PIPELINE_STAGES;
		          const int k_byte = k_tile_idx * K_TILE_BYTES;

	          if constexpr (DEBUG_STAGE == 2) {
	            break;
	          }

		          // Ensure warp1 has finished using the stage buffer for tile (k-1) before we issue the
		          // next prefetch (which may overwrite that same stage in the circular pipeline).
		          if constexpr (WS_UNROLL2_MMA) {
	            if (lane == 0 && ws_u1_active && k_tile_idx > 0) {
	              while (ws_u1_done_tile < (k_tile_idx - 1)) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
	                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
	              }
	            }
	          }
	          if constexpr (WS_SPLIT_U0_SEGS) {
	            if (lane == 0 && k_tile_idx > 0) {
	              while (ws_u0_done_tile < (k_tile_idx - 1)) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
	                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
	              }
	            }
	          }

		          // Prefetch the next K tile into the next stage.
		          const bool has_next = (k_tile_idx + 1) < k_tiles_total;
	          block_barrier::arrival_token tok_next;
	          if (has_next) {
                const int next_tile = k_tile_idx + 1;
                const int next_k_byte = next_tile * K_TILE_BYTES;
	                const int sfa_row_offset_next = sfa_row_offset_for(next_tile);
	                const int sfb_row_offset_next = sfb_row_offset_for(tile_n, next_tile);
            if constexpr (WS_TMA_PRODUCER) {
              if (lane == 0) {
                block_barrier* bar = bars[stage_next];
                constexpr size_t kBytesA = sizeof(sA[0]);
                int unroll_n_valid = 0;
#pragma unroll
                for (int u = 0; u < UnrollN; ++u) {
                  if ((tile_n + u) < n_tiles_group) {
                    ++unroll_n_valid;
                  }
                }
                const int unroll_n_tx = unroll_n_valid;
	                size_t kBytesB = static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(TILE_N) *
	                                 static_cast<size_t>(K_TILE_BYTES);
	                if constexpr (CtaGroup == 2 && UnrollN == 1) {
	                  if (cta2_partition_b_mode == 1) {
	                    kBytesB = static_cast<size_t>(unroll_n_tx) *
	                              static_cast<size_t>(TILE_N / 2) *
	                              static_cast<size_t>(K_TILE_BYTES);
	                  }
	                }
	                size_t kBytesSFB = static_cast<size_t>(unroll_n_tx) * static_cast<size_t>(SFB_TILE_BYTES);
	                const size_t kBytesSF = static_cast<size_t>(SFA_TILE_BYTES) + kBytesSFB;
                tok_next = cuda::device::barrier_arrive_tx(*bar, 1, kBytesA + kBytesB + kBytesSF);

                ws_tma_req_stage = stage_next;
                ws_tma_req_k_byte = next_k_byte;
                ws_tma_req_sfa_row_offset = sfa_row_offset_next;
                ws_tma_req_sfb_row_offset = sfb_row_offset_next;
                __threadfence_block();
                ws_tma_req_tile = next_tile;
              }
            } else {
              tok_next = issue_tma_tile(stage_next, next_k_byte, sfa_row_offset_next, sfb_row_offset_next);
            }
	          }

						          if (lane == 0) {
						            const uint64_t desc_sfa_base =
						                static_cast<uint64_t>(desc_sfa[stage_cur]) + cta2_desc_sfa_row_offset;
						            // Scale-copy schedule:
						            // - Always copy SFA in warp0 (shared across u=0/u=1 MMAs).
						            // - For WS_UNROLL2_MMA, warp0 signals warp1 after SFA is ready; warp1 copies SFB(u=1).
							            copy_scale_fragments(desc_sfa_base, tmem_sfa_ptrs, /*is_sfb=*/false);
								            if constexpr (WS_UNROLL2_MMA) {
								              if (ws_u1_active) {
								                // Signal warp1 that TMA is complete and SFA is resident in TMEM for this K tile.
								                ws_scales_ready_tile = k_tile_idx;
								              }
								              // Warp0 copies u=0's SFB tile.
								              const uint64_t desc_sfb0_base = static_cast<uint64_t>(desc_sfb[stage_cur][0]);
								              const uint32_t* tmem_sfb_ptrs_u0 = tmem_sfb_ptrs + 0 * 4;
								              copy_scale_fragments(desc_sfb0_base, tmem_sfb_ptrs_u0, /*is_sfb=*/true);
						            } else {
						              for (int u = 0; u < UnrollN; ++u) {
						                if ((tile_n + u) >= n_tiles_group) {
				                  continue;
				                }
				                const uint64_t desc_sfb_base = static_cast<uint64_t>(desc_sfb[stage_cur][u]);
				                const uint32_t* tmem_sfb_ptrs_u = tmem_sfb_ptrs + u * 4;
						                copy_scale_fragments(desc_sfb_base, tmem_sfb_ptrs_u, /*is_sfb=*/true);
						              }
							            }
							            if constexpr (DEBUG_STAGE != 3) {
							              if constexpr (WS_SPLIT_U0_SEGS) {
						                if ((tile_n + 0) < n_tiles_group) {
					                  constexpr int seg = 0;
					                  constexpr int u = 0;
					                  const uint64_t desc_a =
					                      static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
					                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                  const uint64_t desc_b =
					                      static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
					                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                  const uint32_t accumulate = (k_byte == 0) ? 0u : 1u;
					                  tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
					                      desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg], tmem_sfa_seg[seg],
					                      tmem_sfb_seg[u][seg]);
					                }
						                ws_u0_ready_tile = k_tile_idx;
						              } else {
#pragma unroll
						                for (int seg = 0; seg < 4; ++seg) {
				                  const uint64_t desc_a =
				                      static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
				                      static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                  const uint32_t accumulate = (k_byte == 0 && seg == 0) ? 0u : 1u;
#pragma unroll
						                  for (int u = 0; u < UnrollN; ++u) {
						                    if ((tile_n + u) >= n_tiles_group) {
						                      continue;
						                    }
					                    if constexpr (WS_UNROLL2_MMA) {
					                      if (ws_u1_active && u == 1) {
					                        continue;
					                      }
					                    }
					                    const uint64_t desc_b =
					                        static_cast<uint64_t>(desc_b_base[stage_cur][u]) + cta2_desc_b_row_offset +
					                        static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
					                    tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
					                        desc_a, desc_b, tmem_c_tiles[u], accumulate, idesc_hi_seg[u][seg],
					                        tmem_sfa_seg[seg], tmem_sfb_seg[u][seg]);
					                  }
				                }
					              }
				            }
				            // See note above: avoid globally fencing TMEM stores after UTCCP scale copies.
				          }

			          if constexpr (DEBUG_STAGE == 3) {
			            break;
			          }

		          if constexpr (DEBUG_STAGE == 4) {
		            break;
		          }

	          // Ensure the next stage is resident in shared memory before the loop advances.
	          if (has_next) {
	            wait_tma_tile_warp0(stage_next, tok_next);
	          } else {
	            __syncwarp();
	          }
	        }
	      }
          if constexpr (WS_TMA_PRODUCER) {
            // Signal the producer warp to exit its request loop before we join the CTA.
            if (warp == 0 && lane == 0) {
              __threadfence_block();
              ws_tma_req_tile = -2;
            }
          }
		      if constexpr (WS_UNROLL2_MMA) {
			        if (warp == 1 && ws_u1_active && lane == 0) {
			          // Warp1 lane0 issues u=1 MMAs once:
			          // - warp0 has copied SFA + SFB(u=1) for this K tile.
		          uint32_t ws_tmem_sfa_seg[4];
		          uint32_t ws_tmem_sfb_seg[4];
		          uint32_t ws_idesc_hi_seg[4];
#pragma unroll
	          for (int seg = 0; seg < 4; ++seg) {
	            const uint32_t tmem_sfa = tmem_sfa_ptrs[seg];
	            const uint32_t tmem_sfb = tmem_sfb_ptrs[1 * 4 + seg];
	            ws_tmem_sfa_seg[seg] = tmem_sfa;
	            ws_tmem_sfb_seg[seg] = tmem_sfb;
	            const uint64_t idesc_runtime = umma::make_runtime_instr_desc_block_scaled(idesc, tmem_sfa, tmem_sfb);
	            ws_idesc_hi_seg[seg] = static_cast<uint32_t>(idesc_runtime >> 32);
	          }

				          for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
				            while (ws_scales_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
				              __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
				            }
				            const int stage_cur = k_tile_idx % PIPELINE_STAGES;
				            // Copy SFB(u=1) scales for this tile from shared memory into TMEM.
				            const uint64_t desc_sfb1_base = static_cast<uint64_t>(desc_sfb[stage_cur][1]);
				            const uint32_t* tmem_sfb_ptrs_u1 = tmem_sfb_ptrs + 1 * 4;
				            if constexpr (WS_SFB1_SEGMENT_HELPERS) {
				              // Warp1 copies only seg0/seg1; helper warps copy seg2/seg3 in parallel.
				              constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
				              tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
				                  desc_sfb1_base + static_cast<uint64_t>(0 * SF_COPY_32x128B_DESC_STEP), tmem_sfb_ptrs_u1[0]);
				              tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(
				                  desc_sfb1_base + static_cast<uint64_t>(1 * SF_COPY_32x128B_DESC_STEP), tmem_sfb_ptrs_u1[1]);
				              // Issue seg0/seg1 MMAs while helper warps copy seg2/seg3.
#pragma unroll
				              for (int seg = 0; seg < 2; ++seg) {
				                const uint64_t desc_a =
				                    static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
				                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
				                const uint64_t desc_b =
				                    static_cast<uint64_t>(desc_b_base[stage_cur][1]) + cta2_desc_b_row_offset +
				                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
				                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
				                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
				                    desc_a, desc_b, tmem_c_tiles[1], accumulate, ws_idesc_hi_seg[seg],
				                    ws_tmem_sfa_seg[seg], ws_tmem_sfb_seg[seg]);
				              }

				              while (ws_sfb1_seg2_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
				                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
				              }
				              while (ws_sfb1_seg3_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
				                __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
				              }

#pragma unroll
				              for (int seg = 2; seg < 4; ++seg) {
				                const uint64_t desc_a =
				                    static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
				                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
				                const uint64_t desc_b =
				                    static_cast<uint64_t>(desc_b_base[stage_cur][1]) + cta2_desc_b_row_offset +
				                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
				                constexpr uint32_t accumulate = 1u;
				                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
				                    desc_a, desc_b, tmem_c_tiles[1], accumulate, ws_idesc_hi_seg[seg],
				                    ws_tmem_sfa_seg[seg], ws_tmem_sfb_seg[seg]);
				              }
				            } else {
				              copy_scale_fragments(desc_sfb1_base, tmem_sfb_ptrs_u1, /*is_sfb=*/true);
#pragma unroll
				              for (int seg = 0; seg < 4; ++seg) {
				                const uint64_t desc_a =
				                    static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
				                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
				                const uint64_t desc_b =
				                    static_cast<uint64_t>(desc_b_base[stage_cur][1]) + cta2_desc_b_row_offset +
				                    static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
				                const uint32_t accumulate = (k_tile_idx == 0 && seg == 0) ? 0u : 1u;
				                tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
				                    desc_a, desc_b, tmem_c_tiles[1], accumulate, ws_idesc_hi_seg[seg],
				                    ws_tmem_sfa_seg[seg], ws_tmem_sfb_seg[seg]);
				              }
				            }
				            ws_u1_done_tile = k_tile_idx;
				          }
				        }
			    }
			  }
			  if constexpr (WS_SFB1_SEGMENT_HELPERS) {
			    // Helper warps copy SFB(u=1) seg2/seg3 for each K tile once warp0 has finished the TMA+SFA step.
			    // No additional fences are required: UMMA will naturally stall on TMEM scale reads if needed.
			    if ((warp == 2 || warp == 3) && ws_u1_active && lane == 0) {
			      const int seg = (warp == 2) ? 2 : 3;
			      constexpr int SF_COPY_32x128B_DESC_STEP = 32;  // 32 * 16B = 512B per segment.
				      for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
				        while (ws_scales_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
				          __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
				        }
				        const int stage_cur = k_tile_idx % PIPELINE_STAGES;
				        const uint64_t desc_sfb1_base = static_cast<uint64_t>(desc_sfb[stage_cur][1]);
				        const uint64_t src_desc = desc_sfb1_base + static_cast<uint64_t>(seg * SF_COPY_32x128B_DESC_STEP);
				        const uint32_t dst_addr = tmem_sfb_ptrs[1 * 4 + seg];
				        tcgen05::utccp_cp_32x128b_warpx4<CtaGroup>(src_desc, dst_addr);
				        if (seg == 2) {
				          ws_sfb1_seg2_ready_tile = k_tile_idx;
				        } else {
				          ws_sfb1_seg3_ready_tile = k_tile_idx;
				        }
				        __threadfence_block();
				      }
				    }
				  }

				  if constexpr (WS_TMA_PRODUCER) {
				    if (warp == 2 && lane == 0) {
				      // Warp2 lane0 acts as a dedicated TMA producer for the warp0-only pipeline path.
		      // Warp0 arms the barrier generation (arrive_tx) and posts requests via shared state.
		      int last_issued = -1;
		      while (true) {
		        const int req_tile = ws_tma_req_tile;
		        if (req_tile == -2) {
		          break;
		        }
		        if (req_tile <= last_issued) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
		          __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
		          continue;
		        }
		        __threadfence_block();
		        const int req_stage = ws_tma_req_stage;
		        const int req_k_byte = ws_tma_req_k_byte;
		        const int req_sfa_row_offset = ws_tma_req_sfa_row_offset;
		        const int req_sfb_row_offset = ws_tma_req_sfb_row_offset;
		        issue_tma_tile_ops_only(req_stage, req_k_byte, req_sfa_row_offset, req_sfb_row_offset);
		        last_issued = req_tile;
		      }
		    }
		  }

		  if constexpr (WS_SPLIT_U0_SEGS) {
		    if (warp == 2 && lane == 0) {
		      // Warp2 lane0 issues the remaining u=0 MMAs (seg=1..3) after warp0 has issued seg0.
		      // This reduces single-thread UMMA issue pressure for the u=0 tile.
		      uint32_t ws_tmem_sfa_seg[4];
		      uint32_t ws_tmem_sfb_seg[4];
		      uint32_t ws_idesc_hi_seg[4];
#pragma unroll
		      for (int seg = 0; seg < 4; ++seg) {
		        const uint32_t tmem_sfa = tmem_sfa_ptrs[seg];
		        const uint32_t tmem_sfb = tmem_sfb_ptrs[0 * 4 + seg];
		        ws_tmem_sfa_seg[seg] = tmem_sfa;
		        ws_tmem_sfb_seg[seg] = tmem_sfb;
		        const uint64_t idesc_runtime = umma::make_runtime_instr_desc_block_scaled(idesc, tmem_sfa, tmem_sfb);
		        ws_idesc_hi_seg[seg] = static_cast<uint32_t>(idesc_runtime >> 32);
		      }

		      for (int k_tile_idx = 0; k_tile_idx < k_tiles_total; ++k_tile_idx) {
		        while (ws_u0_ready_tile < k_tile_idx) {
#if NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES
		          __nanosleep(NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES);
#endif
		        }
		        const int stage_cur = (PIPELINE_STAGES == 1) ? 0 : (k_tile_idx % PIPELINE_STAGES);
#pragma unroll
		        for (int seg = 1; seg < 4; ++seg) {
		          const uint64_t desc_a =
		              static_cast<uint64_t>(desc_a_base[stage_cur]) + cta2_desc_a_row_offset +
		              static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
		          const uint64_t desc_b =
		              static_cast<uint64_t>(desc_b_base[stage_cur][0]) + cta2_desc_b_row_offset +
		              static_cast<uint64_t>(seg * (K_SEG_BYTES >> 4));
		          constexpr uint32_t accumulate = 1u;
		          tcgen05::mma_mxf4nvf4_block16<CtaGroup>(
		              desc_a, desc_b, tmem_c_tiles[0], accumulate, ws_idesc_hi_seg[seg], ws_tmem_sfa_seg[seg],
		              ws_tmem_sfb_seg[seg]);
		        }
		        ws_u0_done_tile = k_tile_idx;
		      }
		    }
		  }

		    // Join all warps before any later TMEM loads in debug/epilogue.
		    __syncthreads();
		    } else {
		      run_full_cta_mainloop();
		    }
	  } else {
	    run_full_cta_mainloop();
	  }

	  if constexpr (CtaGroup == 2 || CTA1_COMMIT_BARRIER) {
	    uint32_t bar_addr = tcgen05::cast_smem_ptr_to_uint(&umma_done_barrier);
	    if constexpr (CtaGroup == 2) {
	      // Ensure both CTAs in the cluster target CTA0's barrier (CUTLASS Sm100MmaPeerBitMask).
	      bar_addr &= tcgen05::Sm100MmaPeerBitMask;
	    }
	    // For cta_group::1, ensure all warp-specialized MMA issue has completed before we commit.
	    // (Without this, thread0 could commit early while other warps are still issuing MMAs.)
	    if constexpr (CTA1_COMMIT_BARRIER) {
	      __syncthreads();
	    }
	    // One lane per CTA issues the commit.
	    // For cta_group::2, the barrier flips only once both CTAs have arrived, which prevents
	    // missing a short-lived intermediate parity.
	    if (threadIdx.x == 0) {
	      if constexpr (CtaGroup == 2) {
	        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
	                     :
	                     : "r"(bar_addr)
	                     : "memory");
	      } else {
	        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
	                     :
	                     : "r"(bar_addr)
	                     : "memory");
	      }
	    }
    // Wait for commit completion before any TMEM loads.
    // For cta_group::2 we wait on the shared (CTA0) barrier once, then cluster-sync.
    if (threadIdx.x == 0 && (CtaGroup != 2 || cluster_rank == 0)) {
	      uint32_t done = 0;
	      constexpr uint32_t kPhase = 0;
	      while (done == 0) {
	        asm volatile("{\n\t"
	                     ".reg .pred P1;\n\t"
	                     "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
	                     "selp.b32 %0, 1, 0, P1;\n\t"
	                     "}\n"
	                     : "=r"(done)
	                     : "r"(bar_addr), "r"(kPhase)
	                     : "memory");
	      }
	    }
	    __syncthreads();
	    if constexpr (CtaGroup == 2) {
	      cg::this_cluster().sync();
	    }
	  }

  // Ensure UMMA writes (TMEM stores) are visible to TMEM loads before any debug/epilogue reads.
  // For cta_group::2, UMMA is explicitly asynchronous and requires a commit+mbarrier wait above.
  // For cta_group::1, `tcgen05.ld.sync` will naturally stall on outstanding writes, so avoid an
  // extra global wait here (latency sensitive).
  if constexpr (CtaGroup == 2) {
    tcgen05::tmem_wait_st_sync();
    tcgen05::tmem_wait_ld_sync();
  }

  // Optional debug: dump a slice of TMEM (dp x col) into the output tile.
  // This is for offline mapping analysis only; it bypasses the normal epilogue.
  if (debug_tmem_dump != 0) {
    // Only dump from the first tile/group to avoid races/overwrites.
    if (group_idx == 0 && tile_m == 0 && tile_n == 0 &&
        (debug_tmem_only_rank < 0 || cluster_rank == debug_tmem_only_rank)) {
      // Special debug modes (>=10): dump TMEM scale tiles (raw bytes) into the output.
      // This helps validate whether UTCCP wrote the expected FP8 (UE4M3) scale bytes into TMEM.
      //
      // Dump format: output is FP16, but values are integer-coded byte values (0..255).
      // Layout: rows correspond to DP lanes 0..31, columns correspond to 4 segments x 16 bytes/row:
      //   col = seg*16 + byte_idx (byte_idx in 0..15). Each segment contributes 32 rows.
      //
      // debug_tmem_dump meanings:
      //   10/11/12: dump UTCCP destination bases (SFA / SFB u0 / SFB u1)
      //   13/14/15: dump MMA-consumed bases      (SFA / SFB u0 / SFB u1)
      //   20..25: same as above, but force base dp=0 (detect dp-banking behavior)
	      if (debug_tmem_dump >= 10) {
	        const int lane = static_cast<int>(threadIdx.x) & 31;
	        if ((threadIdx.x >> 5) == 0) {
	          constexpr int DUMP_DP = 32;   // one warp covers dp 0..31
	          constexpr int WORDS_PER_ROW = 4;  // 16 bytes/row = 4x 32-bit words
	          const int dp = lane;
	          const int gm = m_offset + dp;
	          if (dp < DUMP_DP && gm < m_size) {
	          const bool clear_dp_base = (debug_tmem_dump >= 20);
	          const int mode_class = clear_dp_base ? (debug_tmem_dump - 20) : (debug_tmem_dump - 10);
	          if (mode_class >= 0 && mode_class <= 5) {
	            const bool dump_mma_view = (mode_class >= 3);
	            const int operand_sel = mode_class % 3;  // 0=SFA, 1=SFB u0, 2=SFB u1
	            const bool dump_sfa = (operand_sel == 0);
	            const int dump_u = (operand_sel == 2) ? 1 : 0;
	            if (!dump_sfa && dump_u >= UnrollN) {
	              // UnrollN=1 cannot dump u=1.
	            } else {
	              for (int seg = 0; seg < 4; ++seg) {
	                // Under cta_group::2 + CUTLASS tmem_sf_frg mapping, SFA and (sometimes) SFB scale
	                // fragments use per-seg base columns {0,2,4,6}. On SM100a, 32x128b UTCCP can trap for
	                // misaligned bases (col % 4 != 0), so we fall back to UTCCP64 which writes:
	                //   seg01 -> aligned base col0 (spanning cols 0 and 2)
	                //   seg23 -> aligned base col4 (spanning cols 4 and 6)
	                // and encodes the second segment in the pair via a DP+64 window.
	                //
	                // This debug dump mirrors the validated placement in `tmem_sf_frg_probe.cu` so the
	                // host tool can compare dumped bytes vs the packed [128,16] scale tile.
	                uint32_t tmem_ptrs[4];
	                if (dump_mma_view) {
	                  if (dump_sfa) {
	                    tmem_ptrs[0] = tmem_sfa_seg[0];
	                    tmem_ptrs[1] = tmem_sfa_seg[1];
	                    tmem_ptrs[2] = tmem_sfa_seg[2];
	                    tmem_ptrs[3] = tmem_sfa_seg[3];
	                  } else {
	                    tmem_ptrs[0] = tmem_sfb_seg[dump_u][0];
	                    tmem_ptrs[1] = tmem_sfb_seg[dump_u][1];
	                    tmem_ptrs[2] = tmem_sfb_seg[dump_u][2];
	                    tmem_ptrs[3] = tmem_sfb_seg[dump_u][3];
	                  }
	                } else {
	                  if (dump_sfa) {
	                    tmem_ptrs[0] = tmem_sfa_ptrs[0];
	                    tmem_ptrs[1] = tmem_sfa_ptrs[1];
	                    tmem_ptrs[2] = tmem_sfa_ptrs[2];
	                    tmem_ptrs[3] = tmem_sfa_ptrs[3];
	                  } else {
	                    tmem_ptrs[0] = tmem_sfb_ptrs[dump_u * 4 + 0];
	                    tmem_ptrs[1] = tmem_sfb_ptrs[dump_u * 4 + 1];
	                    tmem_ptrs[2] = tmem_sfb_ptrs[dump_u * 4 + 2];
	                    tmem_ptrs[3] = tmem_sfb_ptrs[dump_u * 4 + 3];
	                  }
	                }

	                // Detect whether UTCCP64 placement is active for this operand.
	                bool dst_cols_aligned = true;
	#pragma unroll
	                for (int s = 0; s < 4; ++s) {
	                  dst_cols_aligned &= ((tmem_ptrs[s] & 0x3u) == 0u);
	                }
	                bool use64 = false;
	#if NVFP4_GROUP_GEMM_USE_UTCCP_64X128B || NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA || NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB
	                // If UTCCP64 is explicitly requested for this operand at compile time, honor it.
	                use64 = dump_sfa ? ((NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA != 0) ||
	                                    (NVFP4_GROUP_GEMM_USE_UTCCP_64X128B != 0))
	                                 : ((NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB != 0) ||
	                                    (NVFP4_GROUP_GEMM_USE_UTCCP_64X128B != 0));
	#endif
	                const bool use64_eff = use64 || ((CtaGroup == 2) && (!dst_cols_aligned));

	                if (use64_eff) {
	                  const int seg_pair = seg & ~1;
	                  uint32_t base0 = tmem_ptrs[seg_pair + 0];
	                  uint32_t base1 = tmem_ptrs[seg_pair + 1];
	                  if (clear_dp_base) {
	                    base0 &= 0xFF00FFFFu;
	                    base1 &= 0xFF00FFFFu;
	                  }
	                  // UTCCP64 segment stride in DP depends on the schedule variant.
	                  // Validated on B200 via `tmem_sf_frg_probe.cu`:
	                  // - ::01_23 uses dp_delta=64 for the second 32-row stripe.
	                  // - ::02_13 uses dp_delta=32 for the second 32-row stripe.
	                  uint32_t dp_delta = 64u;
	#if NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 0
	                  dp_delta = 64u;
	#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 1
	                  dp_delta = 32u;
	#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 2
	                  dp_delta = dump_sfa ? 64u : 32u;
	#elif NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE == 3
	                  dp_delta = dump_sfa ? 32u : 64u;
	#else
	#error "Unsupported NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE"
	#endif
	                  const uint32_t dp_read = static_cast<uint32_t>(dp + ((seg & 1) ? dp_delta : 0u));

	                  // Each 16B row is split into two 8B halves across the paired base columns:
	                  //   base0 col_sub{0,1} -> bytes[0..7]
	                  //   base1 col_sub{0,1} -> bytes[8..15]
	                  const uint32_t bits0 = tcgen05::tmem_ld_32dp32b_x1(tcgen05::tmem_addr_add(base0, dp_read, 0u));
	                  const uint32_t bits1 = tcgen05::tmem_ld_32dp32b_x1(tcgen05::tmem_addr_add(base0, dp_read, 1u));
	                  const uint32_t bits2 = tcgen05::tmem_ld_32dp32b_x1(tcgen05::tmem_addr_add(base1, dp_read, 0u));
	                  const uint32_t bits3 = tcgen05::tmem_ld_32dp32b_x1(tcgen05::tmem_addr_add(base1, dp_read, 1u));
	                  const uint32_t bits_arr[WORDS_PER_ROW] = {bits0, bits1, bits2, bits3};
	#pragma unroll
	                  for (int w = 0; w < WORDS_PER_ROW; ++w) {
	                    const uint32_t bits = bits_arr[w];
	#pragma unroll
	                    for (int b = 0; b < 4; ++b) {
	                      const uint32_t byte_val = (bits >> (8 * b)) & 0xFFu;
	                      const int out_col = seg * 16 + w * 4 + b;
	                      const int gn = n_offset + out_col;
	                      if (gn < n_size) {
	                        c_out[static_cast<size_t>(gm) * static_cast<size_t>(n_size) + static_cast<size_t>(gn)] =
	                            __float2half_rn(static_cast<float>(byte_val));
	                      }
	                    }
	                  }
	                } else {
	                  // UTCCP32 path: each seg base holds a contiguous 16B row at col offsets 0..3.
	                  uint32_t base = tmem_ptrs[seg];
	                  if (clear_dp_base) {
	                    base &= 0xFF00FFFFu;
	                  }
	                  for (int w = 0; w < WORDS_PER_ROW; ++w) {
	                    const uint32_t addr =
	                        tcgen05::tmem_addr_add(base, static_cast<uint32_t>(dp), static_cast<uint32_t>(w));
	                    const uint32_t bits = tcgen05::tmem_ld_32dp32b_x1(addr);
	                    // Unpack 4 bytes from the 32-bit word.
	                    for (int b = 0; b < 4; ++b) {
	                      const uint32_t byte_val = (bits >> (8 * b)) & 0xFFu;
	                      const int out_col = seg * 16 + w * 4 + b;
	                      const int gn = n_offset + out_col;
	                      if (gn < n_size) {
	                        c_out[static_cast<size_t>(gm) * static_cast<size_t>(n_size) + static_cast<size_t>(gn)] =
	                            __float2half_rn(static_cast<float>(byte_val));
	                      }
	                    }
	                  }
	                }
	              }
	            }
	          }
	          }
	        }
	      } else {
        // Default debug modes (1..3): dump accumulator slice as FP16 into the output tile.
        // TMEM loads operate on 32 DP lanes at a single column. Keep `col` uniform within the warp
        // and use the lane id to select DP lanes to avoid misaligned/invalid addressing.
        // Dump a single 128-DP window (one CTA's accumulator height). Use `debug_tmem_dump`
        // to select which base address we probe for alternate layouts.
        constexpr int DUMP_DP = 128;
        // Keep the dump window conservative: dumping beyond the first 128 columns can trip invalid
        // TMEM addressing in some experimental layouts. Expand only once the mapping is validated.
        constexpr int DUMP_COL = 128;
        // Allow probing multiple TMEM subpartitions by adjusting the high idx bits.
        // TMEM pointers are encoded as {col:16, dp:8, idx:8}.
        const uint32_t idx_add_u = (debug_tmem_idx_add > 0) ? static_cast<uint32_t>(debug_tmem_idx_add) : 0u;
        uint32_t tmem_dump_base = static_cast<uint32_t>(tmem_c_rank + (idx_add_u << 24));
        if constexpr (UnrollN == 2) {
          // Debug base selection:
          //  1: tile0 base (tmem_c_rank)
          //  2: tile1 base (col+128)
          //  3: candidate alternate tile1 base (dp+128 aliases dp on SM100; kept for historical bring-up)
          if (debug_tmem_dump == 2) {
            tmem_dump_base = static_cast<uint32_t>(tmem_c_tiles[1] + (idx_add_u << 24));
          } else if (debug_tmem_dump == 3) {
            const uint32_t tmem_dp128 = tcgen05::tmem_addr_add(tmem_c_rank, /*dp_add=*/128u, /*col_add=*/0u);
            tmem_dump_base = static_cast<uint32_t>(tmem_dp128 + (idx_add_u << 24));
          }
        }
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warps_per_cta = static_cast<int>(blockDim.x) >> 5;
        for (int dp_base = warp * 32; dp_base < DUMP_DP; dp_base += warps_per_cta * 32) {
          const int dp = dp_base + lane;
          if (dp >= DUMP_DP) {
            continue;
          }
          for (int col = 0; col < DUMP_COL; ++col) {
            const uint32_t addr =
                tcgen05::tmem_addr_add(tmem_dump_base, static_cast<uint32_t>(dp), static_cast<uint32_t>(col));
            const uint32_t bits = tcgen05::tmem_ld_32dp32b_x1(addr);
            const float f = __uint_as_float(bits);
            const half h = __float2half_rn(f);
            const int gm = m_offset + dp;
            const int gn = n_offset + col;
            if (gm < m_size && gn < n_size) {
              c_out[static_cast<size_t>(gm) * static_cast<size_t>(n_size) + static_cast<size_t>(gn)] = h;
            }
          }
        }
      }
    }

    __syncthreads();
    if (cluster_dim_x > 1) {
      cluster.sync();
    }
		    if (threadIdx.x < 32) {
		      tcgen05::tmem_dealloc<CtaGroup>(tmem_base_c, /*num_columns=*/TMEM_COLUMNS);
		      tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
		    }
		    return;
		  }

		  if constexpr (DEBUG_STAGE == 2) {
			    __syncthreads();
			    if (threadIdx.x < 32) {
			      tcgen05::tmem_dealloc<CtaGroup>(tmem_base_c, /*num_columns=*/TMEM_COLUMNS);
			      tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
			    }
			    return;
			  }

		  if constexpr (DEBUG_STAGE == 3 || DEBUG_STAGE == 4) {
			    __syncthreads();
			    if (threadIdx.x < 32) {
			      tcgen05::tmem_dealloc<CtaGroup>(tmem_base_c, /*num_columns=*/TMEM_COLUMNS);
			      tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
			    }
			    return;
			  }

	  if constexpr (DEBUG_STAGE == 5) {
	    // Full mainloop (TMA + scales + UTCCP + MMA) but skip the epilogue.
	    __syncthreads();
		    if constexpr (CtaGroup == 2) {
		      cluster.sync();
		    }
		    if (threadIdx.x < 32) {
		      tcgen05::tmem_dealloc<CtaGroup>(tmem_base_c, /*num_columns=*/TMEM_COLUMNS);
		      tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
		    }
		    return;
		  }

  // Epilogue: load accumulator tile from TMEM and store to global memory.
  // UMMA accumulators are FP32 in TMEM; convert to FP16 on store.
  const int warps_per_block = (static_cast<int>(blockDim.x) >> 5) > 0 ? (static_cast<int>(blockDim.x) >> 5) : 1;
  // TMEM accumulator mapping:
  // - cta_group::1 (M_MMA=128): the mapping is "simple": (m_local, n) -> dp=m_local, col=n.
  // - cta_group::2 (M_MMA=64): CUTLASS' UMMA_2SM mapping for block-scaled UMMA packs the N128 half into DP:
  //     dp = m_local + (n & 64), col = n
  //   where `n` is the *within-tile* column offset (0..127). See:
  //     `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu`
  //
  // Bring-up knob `cta2_epilogue_addr_mode`:
  //   0: dp += (n & 64), col = (n & 63)            (u=0 mapping from probe; default)
  //   1: dp += (n & 64), col = (n & 63) + 64       (u=1 mapping from probe)
  //   2: dp += (n & 64), col = n                   (u selected by n[6] via col high bit)
  //   3: dp = m_local,    col = n                  (no DP packing; experimental)
  //   4: dp = m_local,    col = (n & 63)           (no DP packing + col mask; experimental)
  //   5: rank-select for cta2 partitioned-N mode: rank0->mode0, rank1->mode1
  //   6: rank-select for cta2 partitioned-N mode: rank0->mode1, rank1->mode0
  //   7: legacy fixed mode0 for all UnrollN=2 u tiles
  //   8: explicit UnrollN=2 u remap (u0->mode0, u1->mode1)
  //   9: swapped UnrollN=2 u remap (u0->mode1, u1->mode0)
  const int cta_rows = CTA_TILE_M;
  const int cta_row_base = 0;
  auto cta2_map_epilogue_dp_col = [&](uint32_t dp_base, uint32_t n_col0, int u_tile) -> uint2 {
    uint32_t dp = dp_base;
    uint32_t col = n_col0;
    if constexpr (CtaGroup == 2) {
      const uint32_t n64 = n_col0 & 64u;
      const uint32_t nlow = n_col0 & 63u;
      int mode = cta2_epilogue_addr_mode;
      if constexpr (UnrollN == 2) {
        // cta_group::2 + UnrollN=2 unpartitioned path mapping controls:
        // - mode 0 (default): u0->mode0, u1->mode1
        // - mode 7: legacy fixed mode0 (no u remap)
        // - mode 8: explicit u0->mode0, u1->mode1 (same as mode0)
        // - mode 9: swapped u0->mode1, u1->mode0
        if (mode == 8) {
          mode = (u_tile == 0) ? 0 : 1;
        } else if (mode == 9) {
          mode = (u_tile == 0) ? 1 : 0;
        } else if (cta2_partition_b_mode != 1) {
          if (mode == 0) {
            mode = (u_tile == 0) ? 0 : 1;
          } else if (mode == 7) {
            mode = 0;
          }
        }
      }
      if (mode == 5 && cta2_partition_b_mode == 1) {
        mode = (cluster_rank_b == 0) ? 0 : 1;
      } else if (mode == 6 && cta2_partition_b_mode == 1) {
        mode = (cluster_rank_b == 0) ? 1 : 0;
      }
      if (mode == 0) {
        dp += n64;
        col = nlow;
      } else if (mode == 1) {
        dp += n64;
        col = nlow + 64u;
      } else if (mode == 2) {
        dp += n64;
        col = n_col0;
      } else if (mode == 3) {
        dp = dp_base;
        col = n_col0;
      } else if (mode == 4) {
        dp = dp_base;
        col = nlow;
      } else {
        // Default to the probe's u=0 mapping (mode 0).
        dp += n64;
        col = nlow;
      }
    }
    return make_uint2(dp, col);
  };
  for (int row_chunk = warp; row_chunk * 32 < cta_rows; row_chunk += warps_per_block) {
    const int row_start = m_offset + cta_row_base + row_chunk * 32;
    // `tcgen05.ld.32dp...` is a warp-level operation: all lanes must participate.
    // If the entire warp maps to out-of-bounds rows, we can skip the loads safely.
    if (row_start >= m_size) {
      continue;
    }
    const int m_local = row_chunk * 32 + lane;
    const int gm = row_start + lane;
    const bool row_in_bounds = (gm < m_size);
    const bool row_full = (row_start + 31) < m_size;
    const size_t base = row_in_bounds ? (static_cast<size_t>(gm) * static_cast<size_t>(n_size)) : 0u;
    const uint32_t dp_lane_base = static_cast<uint32_t>(m_local);
    // Vectorized TMEM loads of FP32 accumulators. Use x8.b32 and explicitly convert FP32->FP16
    // to avoid interpreting FP32 bits as packed FP16 (which would corrupt results).
#pragma unroll
    for (int u = 0; u < UnrollN; ++u) {
      if ((tile_n + u) >= n_tiles_group) {
        continue;
      }
      const uint32_t tmem_c_tile_u = tmem_c_tiles[u];
      const uint32_t tmem_col_offset_u = 0u;
      // cta_group::2 partitioned-B mode:
      // Each CTA rank computes and owns only N/2 columns of the N128 tile (rank0 -> [0:64), rank1 -> [64:128)).
      // The B tensormap box height is set to 64 on the host for this mode, so writing a full N128 epilogue
      // would store garbage/uninitialized columns. Restrict stores to the owned N slice.
      const bool cta2_partition_b_global_shift = (CtaGroup == 2) && (cta2_partition_b_mode == 1);
      const int cta2_n_slice = cta2_partition_b_global_shift ? (TILE_N / 2) : TILE_N;
      const int cta2_n_global_offset = cta2_partition_b_global_shift ? (cluster_rank_b * (TILE_N / 2)) : 0;
      const int n_offset_u = n_offset + u * TILE_N + cta2_n_global_offset;
      const bool n_tile_full = (n_offset_u + cta2_n_slice) <= n_size;
      if (row_full && n_tile_full) {
        const size_t out_base = base + static_cast<size_t>(n_offset_u);
#if NVFP4_GROUP_GEMM_EPILOGUE_LD_X32
        for (int n_base = 0; n_base < cta2_n_slice; n_base += 32) {
              // Feed global-in-tile N to the TMEM address mapper so cta_group::2 can fold n[6]
              // into DP (rank1 N[64:128) -> dp+64 path) when required by the UMMA layout.
              uint32_t col_lane = static_cast<uint32_t>(n_base + cta2_n_global_offset);
          uint32_t dp_lane = dp_lane_base;
          if constexpr (CtaGroup == 2) {
            const uint2 mapped = cta2_map_epilogue_dp_col(dp_lane_base, col_lane, u);
            dp_lane = mapped.x;
            col_lane = mapped.y;
          }
          const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tile_u, dp_lane, tmem_col_offset_u + col_lane);
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
          uint32_t v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31;
          tcgen05::tmem_ld_32dp32b_x32(
              addr,
              v0,
              v1,
              v2,
              v3,
              v4,
              v5,
              v6,
              v7,
              v8,
              v9,
              v10,
              v11,
              v12,
              v13,
              v14,
              v15,
              v16,
              v17,
              v18,
              v19,
              v20,
              v21,
              v22,
              v23,
              v24,
              v25,
              v26,
              v27,
              v28,
              v29,
              v30,
              v31);
          __half2* out_h2 = reinterpret_cast<__half2*>(c_out + out_base + static_cast<size_t>(n_base));
          const float f0 = __uint_as_float(v0);
          const float f1 = __uint_as_float(v1);
          const float f2 = __uint_as_float(v2);
          const float f3 = __uint_as_float(v3);
          const float f4 = __uint_as_float(v4);
          const float f5 = __uint_as_float(v5);
          const float f6 = __uint_as_float(v6);
          const float f7 = __uint_as_float(v7);
          const float f8 = __uint_as_float(v8);
          const float f9 = __uint_as_float(v9);
          const float f10 = __uint_as_float(v10);
          const float f11 = __uint_as_float(v11);
          const float f12 = __uint_as_float(v12);
          const float f13 = __uint_as_float(v13);
          const float f14 = __uint_as_float(v14);
          const float f15 = __uint_as_float(v15);
          const float f16 = __uint_as_float(v16);
          const float f17 = __uint_as_float(v17);
          const float f18 = __uint_as_float(v18);
          const float f19 = __uint_as_float(v19);
          const float f20 = __uint_as_float(v20);
          const float f21 = __uint_as_float(v21);
          const float f22 = __uint_as_float(v22);
          const float f23 = __uint_as_float(v23);
          const float f24 = __uint_as_float(v24);
          const float f25 = __uint_as_float(v25);
          const float f26 = __uint_as_float(v26);
          const float f27 = __uint_as_float(v27);
          const float f28 = __uint_as_float(v28);
          const float f29 = __uint_as_float(v29);
          const float f30 = __uint_as_float(v30);
          const float f31 = __uint_as_float(v31);
          out_h2[0] = __floats2half2_rn(f0, f1);
          out_h2[1] = __floats2half2_rn(f2, f3);
          out_h2[2] = __floats2half2_rn(f4, f5);
          out_h2[3] = __floats2half2_rn(f6, f7);
          out_h2[4] = __floats2half2_rn(f8, f9);
          out_h2[5] = __floats2half2_rn(f10, f11);
          out_h2[6] = __floats2half2_rn(f12, f13);
          out_h2[7] = __floats2half2_rn(f14, f15);
          out_h2[8] = __floats2half2_rn(f16, f17);
          out_h2[9] = __floats2half2_rn(f18, f19);
          out_h2[10] = __floats2half2_rn(f20, f21);
          out_h2[11] = __floats2half2_rn(f22, f23);
          out_h2[12] = __floats2half2_rn(f24, f25);
          out_h2[13] = __floats2half2_rn(f26, f27);
          out_h2[14] = __floats2half2_rn(f28, f29);
          out_h2[15] = __floats2half2_rn(f30, f31);
        }
#elif NVFP4_GROUP_GEMM_EPILOGUE_LD_X16
        for (int n_base = 0; n_base < cta2_n_slice; n_base += 16) {
              uint32_t col_lane = static_cast<uint32_t>(n_base + cta2_n_global_offset);
          uint32_t dp_lane = dp_lane_base;
          if constexpr (CtaGroup == 2) {
            const uint2 mapped = cta2_map_epilogue_dp_col(dp_lane_base, col_lane, u);
            dp_lane = mapped.x;
            col_lane = mapped.y;
          }
          const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tile_u, dp_lane, tmem_col_offset_u + col_lane);
          uint32_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
          tcgen05::tmem_ld_32dp32b_x16(addr, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
          __half2* out_h2 = reinterpret_cast<__half2*>(c_out + out_base + static_cast<size_t>(n_base));
          const float f0 = __uint_as_float(v0);
          const float f1 = __uint_as_float(v1);
          const float f2 = __uint_as_float(v2);
          const float f3 = __uint_as_float(v3);
          const float f4 = __uint_as_float(v4);
          const float f5 = __uint_as_float(v5);
          const float f6 = __uint_as_float(v6);
          const float f7 = __uint_as_float(v7);
          const float f8 = __uint_as_float(v8);
          const float f9 = __uint_as_float(v9);
          const float f10 = __uint_as_float(v10);
          const float f11 = __uint_as_float(v11);
          const float f12 = __uint_as_float(v12);
          const float f13 = __uint_as_float(v13);
          const float f14 = __uint_as_float(v14);
          const float f15 = __uint_as_float(v15);
          out_h2[0] = __floats2half2_rn(f0, f1);
          out_h2[1] = __floats2half2_rn(f2, f3);
          out_h2[2] = __floats2half2_rn(f4, f5);
          out_h2[3] = __floats2half2_rn(f6, f7);
          out_h2[4] = __floats2half2_rn(f8, f9);
          out_h2[5] = __floats2half2_rn(f10, f11);
          out_h2[6] = __floats2half2_rn(f12, f13);
          out_h2[7] = __floats2half2_rn(f14, f15);
        }
#else
        // Workaround: `tcgen05.ld ... x8` has been observed to fault for some UnrollN=2 TMEM ranges.
        // Use x16 for UnrollN=2 to keep epilogue stable; keep x8 for UnrollN=1.
        if constexpr (UnrollN == 2) {
          for (int n_base = 0; n_base < cta2_n_slice; n_base += 16) {
                uint32_t col_lane = static_cast<uint32_t>(n_base + cta2_n_global_offset);
            uint32_t dp_lane = dp_lane_base;
            if constexpr (CtaGroup == 2) {
              const uint2 mapped = cta2_map_epilogue_dp_col(dp_lane_base, col_lane, u);
              dp_lane = mapped.x;
              col_lane = mapped.y;
            }
            const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tile_u, dp_lane, tmem_col_offset_u + col_lane);
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
            tcgen05::tmem_ld_32dp32b_x16(addr, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
            __half2* out_h2 = reinterpret_cast<__half2*>(c_out + out_base + static_cast<size_t>(n_base));
            const float f0 = __uint_as_float(v0);
            const float f1 = __uint_as_float(v1);
            const float f2 = __uint_as_float(v2);
            const float f3 = __uint_as_float(v3);
            const float f4 = __uint_as_float(v4);
            const float f5 = __uint_as_float(v5);
            const float f6 = __uint_as_float(v6);
            const float f7 = __uint_as_float(v7);
            const float f8 = __uint_as_float(v8);
            const float f9 = __uint_as_float(v9);
            const float f10 = __uint_as_float(v10);
            const float f11 = __uint_as_float(v11);
            const float f12 = __uint_as_float(v12);
            const float f13 = __uint_as_float(v13);
            const float f14 = __uint_as_float(v14);
            const float f15 = __uint_as_float(v15);
            out_h2[0] = __floats2half2_rn(f0, f1);
            out_h2[1] = __floats2half2_rn(f2, f3);
            out_h2[2] = __floats2half2_rn(f4, f5);
            out_h2[3] = __floats2half2_rn(f6, f7);
            out_h2[4] = __floats2half2_rn(f8, f9);
            out_h2[5] = __floats2half2_rn(f10, f11);
            out_h2[6] = __floats2half2_rn(f12, f13);
            out_h2[7] = __floats2half2_rn(f14, f15);
          }
        } else {
          for (int n_base = 0; n_base < cta2_n_slice; n_base += 8) {
            uint32_t col_lane = static_cast<uint32_t>(n_base + cta2_n_global_offset);
            uint32_t dp_lane = dp_lane_base;
            if constexpr (CtaGroup == 2) {
              const uint2 mapped = cta2_map_epilogue_dp_col(dp_lane_base, col_lane, u);
              dp_lane = mapped.x;
              col_lane = mapped.y;
            }
            const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tile_u, dp_lane, tmem_col_offset_u + col_lane);
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
            tcgen05::tmem_ld_32dp32b_x8(addr, v0, v1, v2, v3, v4, v5, v6, v7);
            __half2* out_h2 = reinterpret_cast<__half2*>(c_out + out_base + static_cast<size_t>(n_base));
            const float f0 = __uint_as_float(v0);
            const float f1 = __uint_as_float(v1);
            const float f2 = __uint_as_float(v2);
            const float f3 = __uint_as_float(v3);
            const float f4 = __uint_as_float(v4);
            const float f5 = __uint_as_float(v5);
            const float f6 = __uint_as_float(v6);
            const float f7 = __uint_as_float(v7);
            out_h2[0] = __floats2half2_rn(f0, f1);
            out_h2[1] = __floats2half2_rn(f2, f3);
            out_h2[2] = __floats2half2_rn(f4, f5);
            out_h2[3] = __floats2half2_rn(f6, f7);
          }
        }
#endif
      } else {
        if constexpr (UnrollN == 2) {
          for (int n_base = 0; n_base < cta2_n_slice; n_base += 16) {
            uint32_t col_lane = static_cast<uint32_t>(n_base + cta2_n_global_offset);
            uint32_t dp_lane = dp_lane_base;
            if constexpr (CtaGroup == 2) {
              const uint2 mapped = cta2_map_epilogue_dp_col(dp_lane_base, col_lane, u);
              dp_lane = mapped.x;
              col_lane = mapped.y;
            }
            const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tile_u, dp_lane, tmem_col_offset_u + col_lane);
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15;
            tcgen05::tmem_ld_32dp32b_x16(addr, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);

            if (!row_in_bounds) {
              continue;
            }

            const int gn_base = n_offset_u + n_base;
            if (n_tile_full || gn_base + 15 < n_size) {
              __half2* out_h2 = reinterpret_cast<__half2*>(c_out + base + static_cast<size_t>(gn_base));
              const float f0 = __uint_as_float(v0);
              const float f1 = __uint_as_float(v1);
              const float f2 = __uint_as_float(v2);
              const float f3 = __uint_as_float(v3);
              const float f4 = __uint_as_float(v4);
              const float f5 = __uint_as_float(v5);
              const float f6 = __uint_as_float(v6);
              const float f7 = __uint_as_float(v7);
              const float f8 = __uint_as_float(v8);
              const float f9 = __uint_as_float(v9);
              const float f10 = __uint_as_float(v10);
              const float f11 = __uint_as_float(v11);
              const float f12 = __uint_as_float(v12);
              const float f13 = __uint_as_float(v13);
              const float f14 = __uint_as_float(v14);
              const float f15 = __uint_as_float(v15);
              out_h2[0] = __floats2half2_rn(f0, f1);
              out_h2[1] = __floats2half2_rn(f2, f3);
              out_h2[2] = __floats2half2_rn(f4, f5);
              out_h2[3] = __floats2half2_rn(f6, f7);
              out_h2[4] = __floats2half2_rn(f8, f9);
              out_h2[5] = __floats2half2_rn(f10, f11);
              out_h2[6] = __floats2half2_rn(f12, f13);
              out_h2[7] = __floats2half2_rn(f14, f15);
            } else {
              uint16_t* out_u16 = reinterpret_cast<uint16_t*>(c_out + base);
              const uint32_t vs[16] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
#pragma unroll
              for (int i = 0; i < 16; ++i) {
                const int gn = gn_base + i;
                if (gn < n_size) {
                  const float f = __uint_as_float(vs[i]);
                  const half h = __float2half_rn(f);
                  out_u16[static_cast<size_t>(gn)] = reinterpret_cast<const uint16_t&>(h);
                }
              }
            }
          }
        } else {
          for (int n_base = 0; n_base < cta2_n_slice; n_base += 8) {
            uint32_t col_lane = static_cast<uint32_t>(n_base + cta2_n_global_offset);
            uint32_t dp_lane = dp_lane_base;
            if constexpr (CtaGroup == 2) {
              const uint2 mapped = cta2_map_epilogue_dp_col(dp_lane_base, col_lane, u);
              dp_lane = mapped.x;
              col_lane = mapped.y;
            }
            const uint32_t addr = tcgen05::tmem_addr_add(tmem_c_tile_u, dp_lane, tmem_col_offset_u + col_lane);
            uint32_t v0, v1, v2, v3, v4, v5, v6, v7;
            tcgen05::tmem_ld_32dp32b_x8(addr, v0, v1, v2, v3, v4, v5, v6, v7);

            if (!row_in_bounds) {
              continue;
            }

            const int gn_base = n_offset_u + n_base;
            if (n_tile_full || gn_base + 7 < n_size) {
              __half2* out_h2 = reinterpret_cast<__half2*>(c_out + base + static_cast<size_t>(gn_base));
              const float f0 = __uint_as_float(v0);
              const float f1 = __uint_as_float(v1);
              const float f2 = __uint_as_float(v2);
              const float f3 = __uint_as_float(v3);
              const float f4 = __uint_as_float(v4);
              const float f5 = __uint_as_float(v5);
              const float f6 = __uint_as_float(v6);
              const float f7 = __uint_as_float(v7);
              out_h2[0] = __floats2half2_rn(f0, f1);
              out_h2[1] = __floats2half2_rn(f2, f3);
              out_h2[2] = __floats2half2_rn(f4, f5);
              out_h2[3] = __floats2half2_rn(f6, f7);
            } else {
              uint16_t* out_u16 = reinterpret_cast<uint16_t*>(c_out + base);
              const uint32_t vs[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
#pragma unroll
              for (int i = 0; i < 8; ++i) {
                const int gn = gn_base + i;
                if (gn < n_size) {
                  const float f = __uint_as_float(vs[i]);
                  const half h = __float2half_rn(f);
                  out_u16[static_cast<size_t>(gn)] = reinterpret_cast<const uint16_t&>(h);
                }
              }
            }
          }
        }
      }
    }
  }

  __syncthreads();
	  if constexpr (CtaGroup == 2) {
	    cluster.sync();
	  }
		  if (threadIdx.x < 32) {
		    tcgen05::tmem_dealloc<CtaGroup>(tmem_base_c, /*num_columns=*/TMEM_COLUMNS);
		    tcgen05::tmem_relinquish_alloc_permit<CtaGroup>();
		  }
}

}  // namespace

namespace {

// -----------------------------------------------------------------------------
// Debug: minimal cta_group::2 UTCCP smoke test (isolates misaligned-address faults).
//
// This mirrors the standalone probe in `labs/nvfp4_group_gemm/tmem_sf_frg_probe.cu`:
// - clusterDim.x = 2, gridDim.x = 2, blockDim.x = 128
// - Allocate full TMEM slice (512 columns)
// - Fill a 128x16 SWIZZLE_NONE scale tile in shared memory
// - Issue UTCCP cta_group::2 copies into rank-local TMEM columns
// - Dump a small TMEM window to global memory (optional but cheap)
// -----------------------------------------------------------------------------

constexpr int kUtccpSmokeDumpDp = 128;
constexpr int kUtccpSmokeDumpCol = 64;

__global__ void nvfp4_group_gemm_utccp_cta2_smoke_kernel(
    uint32_t* __restrict__ out_words, int pattern_kind, int use64, int schedule, int stage) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(tcgen05::block_rank_in_cluster());
  if (stage <= 0) {
    return;
  }

  __shared__ uint32_t tmem_base;
  // NOTE: tcgen05.alloc/dealloc are warp-synchronous: issue from a single fully-active warp.
  // For cta_group::2, ensure both CTAs reach alloc together.
  cluster.sync();
  if (threadIdx.x < 32) {
    // Allocate the full TMEM slice to avoid allocator slice interleaving between ranks.
    tcgen05::tmem_alloc_cta2(&tmem_base, /*num_columns=*/512);
  }
  __syncthreads();
  // Match the standalone probe: align both CTAs before touching TMEM.
  cluster.sync();
  if (out_words != nullptr && threadIdx.x == 0) {
    // Stash the raw base pointer so the host can decode {col,dp,idx,sf_id} fields.
    out_words[rank] = tmem_base;
  }
  if (stage == 1) {
    cluster.sync();
    if (threadIdx.x < 32) {
      tcgen05::tmem_dealloc_cta2(tmem_base, /*num_columns=*/512);
      tcgen05::tmem_relinquish_alloc_permit_cta2();
    }
    return;
  }

  __shared__ alignas(16) uint8_t sScale[128][16];
  const int tid = static_cast<int>(threadIdx.x);
  for (int idx = tid; idx < 128 * 16; idx += static_cast<int>(blockDim.x)) {
    const int r = idx / 16;
    const int c = idx - r * 16;
    const uint8_t v = (pattern_kind == 0) ? static_cast<uint8_t>(r) : static_cast<uint8_t>(c);
    sScale[r][c] = v;
  }
  __syncthreads();
  // Match the standalone probe: align both CTAs before UTCCP (cta_group::2 behaves like a collective).
  cluster.sync();

  // Canonical SWIZZLE_NONE UMMA/UTCCP SMEM descriptor used for packed scale tiles.
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
  const uint32_t tmem_rank_base =
      tcgen05::tmem_addr_add(tmem_base, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(rank) * 8u);
  const uint32_t dst01 = tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/0u);
  const uint32_t dst23 = tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/4u);

  if (stage == 2) {
    __syncthreads();
    cluster.sync();
    if (threadIdx.x < 32) {
      tcgen05::tmem_dealloc_cta2(tmem_base, /*num_columns=*/512);
      tcgen05::tmem_relinquish_alloc_permit_cta2();
    }
    return;
  }

  if (use64) {
    // `warpx2` means the instruction must be executed by 2 participating warps (64 threads).
    if (threadIdx.x < 64) {
      const uint64_t src01 = desc_base;
      const uint64_t src23 = desc_base + 64ull;
      if (schedule == 0) {
        tcgen05::utccp_cp_cta2_64x128b_warpx2_01_23(src01, dst01);
        tcgen05::utccp_cp_cta2_64x128b_warpx2_01_23(src23, dst23);
      } else {
        tcgen05::utccp_cp_cta2_64x128b_warpx2_02_13(src01, dst01);
        tcgen05::utccp_cp_cta2_64x128b_warpx2_02_13(src23, dst23);
      }
    }
  } else {
    // `warpx4` means the instruction must be executed by 4 participating warps (128 threads).
    if (threadIdx.x < 128) {
      constexpr int kDescStep32 = 32;  // 32 * 16B = 512B per segment.
      for (int seg = 0; seg < 4; ++seg) {
        const uint64_t src = desc_base + static_cast<uint64_t>(seg * kDescStep32);
        const uint32_t dst =
            tcgen05::tmem_addr_add(tmem_rank_base, /*dp_add=*/0u, /*col_add=*/static_cast<uint32_t>(seg) * 2u);
        tcgen05::utccp_cp_cta2_32x128b_warpx4(src, dst);
      }
    }
  }
  __syncthreads();
  if (stage >= 4) {
    cluster.sync();
    tcgen05::tmem_wait_st_sync();
    tcgen05::tmem_wait_ld_sync();
  }

  if (out_words != nullptr && stage >= 5) {
    const int dump_elems = kUtccpSmokeDumpDp * kUtccpSmokeDumpCol;
    const int base = rank * dump_elems;
    for (int idx = tid; idx < dump_elems; idx += static_cast<int>(blockDim.x)) {
      const int dp = idx / kUtccpSmokeDumpCol;
      const int col = idx - dp * kUtccpSmokeDumpCol;
      const uint32_t addr =
          tcgen05::tmem_addr_add(tmem_base, static_cast<uint32_t>(dp), static_cast<uint32_t>(col));
      out_words[base + idx] = tcgen05::tmem_ld_32dp32b_x1(addr);
    }
  }

  __syncthreads();
  cluster.sync();
  if (threadIdx.x < 32) {
    tcgen05::tmem_dealloc_cta2(tmem_base, /*num_columns=*/512);
    tcgen05::tmem_relinquish_alloc_permit_cta2();
  }
#else
  (void)out_words;
  (void)pattern_kind;
  (void)use64;
  (void)schedule;
  (void)stage;
#endif
}

}  // namespace

void nvfp4_group_gemm_forward_grouped_tcgen05_cuda(
    torch::Tensor a_ptrs,
    torch::Tensor b_ptrs,
    torch::Tensor sfa_ptrs,
    torch::Tensor sfb_ptrs,
    torch::Tensor c_ptrs,
    torch::Tensor m_sizes,
    torch::Tensor n_sizes,
    torch::Tensor k_halves,
    torch::Tensor k_scales,
    torch::Tensor a_descs,
    torch::Tensor b_descs,
    torch::Tensor sfa_descs,
    torch::Tensor sfb_descs,
    torch::Tensor cta_group_idx_map,
    torch::Tensor cta_tile_m_map,
    torch::Tensor cta_tile_n_map,
    int max_m_size,
    int max_n_size) {
  TORCH_CHECK(a_ptrs.is_cuda(), "a_ptrs must be CUDA tensor");
  TORCH_CHECK(b_ptrs.is_cuda(), "b_ptrs must be CUDA tensor");
  TORCH_CHECK(sfa_ptrs.is_cuda(), "sfa_ptrs must be CUDA tensor");
  TORCH_CHECK(sfb_ptrs.is_cuda(), "sfb_ptrs must be CUDA tensor");
  TORCH_CHECK(c_ptrs.is_cuda(), "c_ptrs must be CUDA tensor");
  TORCH_CHECK(m_sizes.is_cuda(), "m_sizes must be CUDA tensor");
  TORCH_CHECK(n_sizes.is_cuda(), "n_sizes must be CUDA tensor");
  TORCH_CHECK(k_halves.is_cuda(), "k_halves must be CUDA tensor");
  TORCH_CHECK(k_scales.is_cuda(), "k_scales must be CUDA tensor");
  TORCH_CHECK(a_descs.is_cuda(), "a_descs must be CUDA tensor");
  TORCH_CHECK(b_descs.is_cuda(), "b_descs must be CUDA tensor");
  TORCH_CHECK(sfa_descs.is_cuda(), "sfa_descs must be CUDA tensor");
  TORCH_CHECK(sfb_descs.is_cuda(), "sfb_descs must be CUDA tensor");
  TORCH_CHECK(cta_group_idx_map.is_cuda(), "cta_group_idx_map must be CUDA tensor");
  TORCH_CHECK(cta_tile_m_map.is_cuda(), "cta_tile_m_map must be CUDA tensor");
  TORCH_CHECK(cta_tile_n_map.is_cuda(), "cta_tile_n_map must be CUDA tensor");

  TORCH_CHECK(a_ptrs.scalar_type() == torch::kInt64, "a_ptrs must be torch.int64");
  TORCH_CHECK(b_ptrs.scalar_type() == torch::kInt64, "b_ptrs must be torch.int64");
  TORCH_CHECK(sfa_ptrs.scalar_type() == torch::kInt64, "sfa_ptrs must be torch.int64");
  TORCH_CHECK(sfb_ptrs.scalar_type() == torch::kInt64, "sfb_ptrs must be torch.int64");
  TORCH_CHECK(c_ptrs.scalar_type() == torch::kInt64, "c_ptrs must be torch.int64");
  TORCH_CHECK(m_sizes.scalar_type() == torch::kInt, "m_sizes must be torch.int32");
  TORCH_CHECK(n_sizes.scalar_type() == torch::kInt, "n_sizes must be torch.int32");
  TORCH_CHECK(k_halves.scalar_type() == torch::kInt, "k_halves must be torch.int32");
  TORCH_CHECK(k_scales.scalar_type() == torch::kInt, "k_scales must be torch.int32");
  TORCH_CHECK(a_descs.scalar_type() == torch::kInt64, "a_descs must be torch.int64");
  TORCH_CHECK(b_descs.scalar_type() == torch::kInt64, "b_descs must be torch.int64");
  TORCH_CHECK(sfa_descs.scalar_type() == torch::kInt64, "sfa_descs must be torch.int64");
  TORCH_CHECK(sfb_descs.scalar_type() == torch::kInt64, "sfb_descs must be torch.int64");
  TORCH_CHECK(cta_group_idx_map.scalar_type() == torch::kInt, "cta_group_idx_map must be torch.int32");
  TORCH_CHECK(cta_tile_m_map.scalar_type() == torch::kInt, "cta_tile_m_map must be torch.int32");
  TORCH_CHECK(cta_tile_n_map.scalar_type() == torch::kInt, "cta_tile_n_map must be torch.int32");

  TORCH_CHECK(a_descs.dim() == 2 && a_descs.size(1) == 16, "a_descs must be [groups,16] int64");
  TORCH_CHECK(b_descs.dim() == 2 && b_descs.size(1) == 16, "b_descs must be [groups,16] int64");
  TORCH_CHECK(sfa_descs.dim() == 2 && sfa_descs.size(1) == 16, "sfa_descs must be [groups,16] int64");
  TORCH_CHECK(sfb_descs.dim() == 2 && sfb_descs.size(1) == 16, "sfb_descs must be [groups,16] int64");

  const int groups = static_cast<int>(m_sizes.numel());
  TORCH_CHECK(groups > 0, "grouped call requires at least one group");
  TORCH_CHECK(a_ptrs.numel() == groups, "a_ptrs length mismatch");
  TORCH_CHECK(b_ptrs.numel() == groups, "b_ptrs length mismatch");
  TORCH_CHECK(sfa_ptrs.numel() == groups, "sfa_ptrs length mismatch");
  TORCH_CHECK(sfb_ptrs.numel() == groups, "sfb_ptrs length mismatch");
  TORCH_CHECK(c_ptrs.numel() == groups, "c_ptrs length mismatch");
  TORCH_CHECK(n_sizes.numel() == groups, "n_sizes length mismatch");
  TORCH_CHECK(k_halves.numel() == groups, "k_halves length mismatch");
  TORCH_CHECK(k_scales.numel() == groups, "k_scales length mismatch");
  TORCH_CHECK(a_descs.size(0) == groups, "a_descs groups mismatch");
  TORCH_CHECK(b_descs.size(0) == groups, "b_descs groups mismatch");
  TORCH_CHECK(sfa_descs.size(0) == groups, "sfa_descs groups mismatch");
  TORCH_CHECK(sfb_descs.size(0) == groups, "sfb_descs groups mismatch");

  TORCH_CHECK(max_m_size > 0, "max_m_size must be > 0");
  TORCH_CHECK(max_n_size > 0, "max_n_size must be > 0");
  auto parse_env_int = [](const char* name, int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      return default_value;
    }
    return std::atoi(value);
  };
  auto parse_env_int_optional = [](const char* name, int* out) -> bool {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    *out = std::atoi(value);
    return true;
  };
  const dim3 block(128, 1, 1);
  const int n_tiles = ceil_div_int(max_n_size, 128);
  const int m_tiles_1sm = ceil_div_int(max_m_size, 128);
  // The experimental cta_group::2 kernel operates on 128-row cluster tiles (2x 64-row CTAs).
  const int m_tiles_2sm = ceil_div_int(max_m_size, 128);
  const int cta2_desc_a_row_offset_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_A_ROW_OFFSET", 0);
  const int cta2_desc_b_row_offset_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_B_ROW_OFFSET", 0);
  const int cta2_desc_sfa_row_offset_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_SFA_ROW_OFFSET", 0);
  const int cta2_epilogue_row_base_rows =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_EPILOGUE_ROW_BASE", 64);
  const int cta2_epilogue_addr_mode =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_EPILOGUE_ADDR_MODE", 0);
  const int cta2_sfb_slot_mode =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_SLOT_MODE", 1);
  const int cta2_tmem_c_word_offset =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_TMEM_C_WORD_OFFSET", 0);
  int cta2_tmem_sf_word_offset = 128;
  const bool cta2_tmem_sf_word_offset_set =
      parse_env_int_optional("AISP_NVFP4_GROUP_GEMM_CTA2_TMEM_SF_WORD_OFFSET", &cta2_tmem_sf_word_offset);
  int cta2_tmem_sf_rank_word_offset = 32;
  const bool cta2_tmem_sf_rank_word_offset_set =
      // Per-rank TMEM scale window (word columns). UnrollN=1 needs 32 cols (SFA16 + SFB16).
      // UnrollN=2 requires a larger window; if unset we expand it below.
      parse_env_int_optional("AISP_NVFP4_GROUP_GEMM_CTA2_TMEM_SF_RANK_WORD_OFFSET", &cta2_tmem_sf_rank_word_offset);
  const int cta2_tsfa_word_offset =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_TSFA_WORD_OFFSET", 0);
  const int cta2_tsfb_word_offset =
      // Bring-up knob for adjusting the SFB TMEM pointer passed to UMMA (in 32-bit word columns).
      // With the current per-rank TMEM windowing, the correct default is 0.
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_TSFB_WORD_OFFSET", 0);
  const int cta2_sfa_sf_id =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_SFA_SF_ID", NVFP4_GROUP_GEMM_CTA2_SFA_SF_ID);
  const int cta2_sfb_sf_id =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_SF_ID", NVFP4_GROUP_GEMM_CTA2_SFB_SF_ID);
  TORCH_CHECK(cta2_sfa_sf_id >= 0 && cta2_sfa_sf_id <= 3,
              "AISP_NVFP4_GROUP_GEMM_CTA2_SFA_SF_ID must be 0..3, got ",
              cta2_sfa_sf_id);
  TORCH_CHECK(cta2_sfb_sf_id >= 0 && cta2_sfb_sf_id <= 3,
              "AISP_NVFP4_GROUP_GEMM_CTA2_SFB_SF_ID must be 0..3, got ",
              cta2_sfb_sf_id);
  const int allow_nonzero_sf_id =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_ALLOW_NONZERO_SF_ID", 0);
  // Safety guard: on B200 (SM100a), we've observed non-zero scale-id bits (addr[31:30]) for
  // block-scaled UMMA can trap with `cudaErrorIllegalInstruction` in some bring-up layouts.
  // Default to fail-fast to keep iteration stable, but allow opt-in sweeps when debugging
  // CUTLASS-aligned scale-id semantics.
  if (allow_nonzero_sf_id == 0) {
    TORCH_CHECK(cta2_sfa_sf_id == 0 && cta2_sfb_sf_id == 0,
                "AISP_NVFP4_GROUP_GEMM_CTA2_{SFA,SFB}_SF_ID must be 0 on B200 by default "
                "(set AISP_NVFP4_GROUP_GEMM_ALLOW_NONZERO_SF_ID=1 to override)");
  }
  const int debug_tmem_dump =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_DEBUG_TMEM_DUMP", 0);
  const int debug_tmem_only_rank =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_DEBUG_TMEM_ONLY_RANK", -1);
  const int debug_tmem_idx_add =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_DEBUG_TMEM_IDX_ADD", 0);
  const int cta2_partition_b =
      // For cta_group::2, default to partitioning B via a global N shift by N/2 (mode=1).
      // Each CTA loads an N/2 slice of B into shared.
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_PARTITION_B", 1);
  const int debug_print_ptrs =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_DEBUG_PRINT_PTRS", 0);
  const int cta2_idesc_m_dim_override =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_IDESC_M_DIM_OVERRIDE", 0);
  const int cta2_idesc_n_dim_override =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_CTA2_IDESC_N_DIM_OVERRIDE", 0);
  const int enable_tma_multicast_env =
      parse_env_int("AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST", 0);
  const int unroll_n = parse_env_int("AISP_NVFP4_GROUP_GEMM_UNROLL_N", 1);
  TORCH_CHECK(unroll_n == 1 || unroll_n == 2,
              "AISP_NVFP4_GROUP_GEMM_UNROLL_N must be 1 or 2, got ",
              unroll_n);
  if (!cta2_tmem_sf_rank_word_offset_set && unroll_n == 2) {
    // UnrollN=2 uses two SFB MN tiles:
    //   per-rank footprint = SFA(16 cols) + SFB(32 cols) = 48 cols.
    // Use 64 for alignment/headroom and to avoid rank overlap.
    cta2_tmem_sf_rank_word_offset = 64;
  }

  const char* cluster_env = std::getenv("AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X");
  int cluster_dim_x = 1;
  if (cluster_env != nullptr && cluster_env[0] != '\0') {
    const int parsed = std::atoi(cluster_env);
    if (parsed > 1) {
      cluster_dim_x = parsed;
    }
  }

  const uint64_t* a_ptrs_dev = reinterpret_cast<const uint64_t*>(a_ptrs.data_ptr<int64_t>());
  const uint64_t* b_ptrs_dev = reinterpret_cast<const uint64_t*>(b_ptrs.data_ptr<int64_t>());
  const uint64_t* sfa_ptrs_dev = reinterpret_cast<const uint64_t*>(sfa_ptrs.data_ptr<int64_t>());
  const uint64_t* sfb_ptrs_dev = reinterpret_cast<const uint64_t*>(sfb_ptrs.data_ptr<int64_t>());
  const uint64_t* c_ptrs_dev = reinterpret_cast<const uint64_t*>(c_ptrs.data_ptr<int64_t>());
  const uint64_t* a_descs_dev = reinterpret_cast<const uint64_t*>(a_descs.data_ptr<int64_t>());
  const uint64_t* b_descs_dev = reinterpret_cast<const uint64_t*>(b_descs.data_ptr<int64_t>());
  const uint64_t* sfa_descs_dev = reinterpret_cast<const uint64_t*>(sfa_descs.data_ptr<int64_t>());
  const uint64_t* sfb_descs_dev = reinterpret_cast<const uint64_t*>(sfb_descs.data_ptr<int64_t>());
  const int32_t* cta_group_idx_dev =
      (cta_group_idx_map.numel() > 0) ? cta_group_idx_map.data_ptr<int32_t>() : nullptr;
  const int32_t* cta_tile_m_dev =
      (cta_tile_m_map.numel() > 0) ? cta_tile_m_map.data_ptr<int32_t>() : nullptr;
  const int32_t* cta_tile_n_dev =
      (cta_tile_n_map.numel() > 0) ? cta_tile_n_map.data_ptr<int32_t>() : nullptr;

  // Opt-in to large shared memory per block for the double-buffered pipeline.
  int max_shared_optin = 0;
  AT_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, at::cuda::current_device()));

  if (cluster_dim_x > 1) {
    int cluster_launch_supported = 0;
    AT_CUDA_CHECK(cudaDeviceGetAttribute(
        &cluster_launch_supported, cudaDevAttrClusterLaunch, at::cuda::current_device()));
    TORCH_CHECK(cluster_launch_supported == 1,
                "Cluster launch requested via AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X=",
                cluster_dim_x,
                ", but cudaDevAttrClusterLaunch is not supported on this device.");

    cudaLaunchConfig_t cfg{};
    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_dim_x;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;

    const char* cta2_env = std::getenv("AISP_NVFP4_GROUP_GEMM_ENABLE_EXPERIMENTAL_CTA2");
    const bool enable_experimental_cta2 = (cta2_env != nullptr && cta2_env[0] != '\0' && std::atoi(cta2_env) != 0);
    const bool use_cta_group2 = (cluster_dim_x == 2) && enable_experimental_cta2;
    // cta_group::2 + UnrollN=2 uses DP+64 for scales; keep scale cols off the accumulator col window.
    // Default to col+64 unless explicitly overridden.
    const int cta2_tmem_sf_word_offset_eff =
        cta2_tmem_sf_word_offset_set
            ? cta2_tmem_sf_word_offset
            : ((use_cta_group2 && unroll_n == 2) ? 64 : 0);
    // Use the normalized runtime value after unroll-dependent default adjustment above.
    const int cta2_tmem_sf_rank_word_offset_eff = cta2_tmem_sf_rank_word_offset;
    int enable_tma_multicast = (enable_tma_multicast_env != 0) ? 1 : 0;
    if (use_cta_group2) {
      // cta_group::2 bring-up currently does not support TMA multicast.
      enable_tma_multicast = 0;
    }

    // Packed-CTA cluster launch (cta_group::1 only):
    // Use the host-precomputed CTA maps so we launch exactly the required CTAs per group,
    // avoiding the "max-based" grid that over-launches blocks for groups with smaller M.
    // This keeps cluster-mode overhead comparable to the non-cluster packed launch.
    const int total_ctas = static_cast<int>(cta_group_idx_map.numel());
    const bool use_packed_cluster = (!use_cta_group2) && (total_ctas > 0);
    if (use_packed_cluster) {
      TORCH_CHECK(cta_tile_m_map.numel() == total_ctas, "cta_tile_m_map length mismatch for packed cluster launch");
      TORCH_CHECK(cta_tile_n_map.numel() == total_ctas, "cta_tile_n_map length mismatch for packed cluster launch");
      if (enable_tma_multicast != 0) {
        TORCH_CHECK(cluster_dim_x <= 16,
                    "AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST=1 requires cluster_dim_x <= 16 "
                    "(mask is 16-bit). Got cluster_dim_x=",
                    cluster_dim_x);
        TORCH_CHECK((total_ctas % cluster_dim_x) == 0,
                    "Packed cluster launch with multicast requires total_ctas divisible by cluster_dim_x to avoid "
                    "partial clusters (would deadlock cluster sync). total_ctas=",
                    total_ctas,
                    " cluster_dim_x=",
                    cluster_dim_x);
      }

      cfg.gridDim = dim3(static_cast<unsigned int>(total_ctas), 1u, 1u);
      cfg.blockDim = block;
      cfg.dynamicSmemBytes = 0;
      cfg.stream = at::cuda::getCurrentCUDAStream();
      cfg.attrs = attrs;
      cfg.numAttrs = 1;

      if (unroll_n == 2) {
#if NVFP4_GROUP_GEMM_TMEM_COLUMNS >= 512
        if (enable_tma_multicast != 0) {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              cta_group_idx_dev,
              cta_tile_m_dev,
              cta_tile_n_dev,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        } else {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              cta_group_idx_dev,
              cta_tile_m_dev,
              cta_tile_n_dev,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        }
#else
        TORCH_CHECK(false, "Packed cluster-mode UnrollN=2 requires NVFP4_GROUP_GEMM_TMEM_COLUMNS >= 512.");
#endif
      } else {
        if (enable_tma_multicast != 0) {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              cta_group_idx_dev,
              cta_tile_m_dev,
              cta_tile_n_dev,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        } else {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              cta_group_idx_dev,
              cta_tile_m_dev,
              cta_tile_n_dev,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        }
      }
      AT_CUDA_CHECK(cudaGetLastError());
      return;
    }
    if (enable_tma_multicast != 0) {
      TORCH_CHECK(cluster_dim_x <= 16,
                  "AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST=1 requires cluster_dim_x <= 16 "
                  "(mask is 16-bit). Got cluster_dim_x=",
                  cluster_dim_x);
      const int n_ctas_n = ceil_div_int(n_tiles, unroll_n);
      TORCH_CHECK((n_ctas_n % cluster_dim_x) == 0,
                  "AISP_NVFP4_GROUP_GEMM_ENABLE_TMA_MULTICAST=1 requires the N-CTA dimension divisible by "
                  "cluster_dim_x to avoid partial clusters (would deadlock cluster sync). "
                  "n_ctas_n=",
                  n_ctas_n,
                  " n_tiles=",
                  n_tiles,
                  " unroll_n=",
                  unroll_n,
                  " cluster_dim_x=",
                  cluster_dim_x);
    }
    const int n_ctas_n = ceil_div_int(n_tiles, unroll_n);
    const int clustered_grid_x =
        use_cta_group2 ? (n_ctas_n * cluster_dim_x) : (ceil_div_int(n_ctas_n, cluster_dim_x) * cluster_dim_x);
    const int grid_y = use_cta_group2 ? m_tiles_2sm : m_tiles_1sm;
    cfg.gridDim = dim3(static_cast<unsigned int>(clustered_grid_x),
                       static_cast<unsigned int>(grid_y),
                       static_cast<unsigned int>(groups));
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    if (use_cta_group2) {
      if (unroll_n == 2) {
#if NVFP4_GROUP_GEMM_TMEM_COLUMNS == 512
        cudaFuncAttributes func_attr{};
        AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<2, 2, 64, false>));
        const int max_dynamic_smem =
            (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                : 0;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_tcgen05_kernel<2, 2, 64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            max_dynamic_smem));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_tcgen05_kernel<2, 2, 64, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_tcgen05_kernel<2, 2, 64, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            nvfp4_group_gemm_tcgen05_kernel<2, 2, 64, false>,
            a_ptrs_dev,
            b_ptrs_dev,
            sfa_ptrs_dev,
            sfb_ptrs_dev,
            c_ptrs_dev,
            m_sizes.data_ptr<int32_t>(),
            n_sizes.data_ptr<int32_t>(),
            k_halves.data_ptr<int32_t>(),
            k_scales.data_ptr<int32_t>(),
            a_descs_dev,
            b_descs_dev,
            sfa_descs_dev,
            sfb_descs_dev,
            /*cta_group_idx_map=*/nullptr,
            /*cta_tile_m_map=*/nullptr,
            /*cta_tile_n_map=*/nullptr,
            cta2_desc_a_row_offset_rows,
            cta2_desc_b_row_offset_rows,
            cta2_desc_sfa_row_offset_rows,
            cta2_epilogue_row_base_rows,
            cta2_epilogue_addr_mode,
            cta2_sfb_slot_mode,
            cta2_tmem_c_word_offset,
            cta2_tmem_sf_word_offset_eff,
	            cta2_tmem_sf_rank_word_offset_eff,
	            cta2_tsfa_word_offset,
	            cta2_tsfb_word_offset,
	            cta2_sfa_sf_id,
	            cta2_sfb_sf_id,
	            debug_tmem_dump,
	            debug_tmem_only_rank,
	            debug_tmem_idx_add,
            cta2_partition_b,
            debug_print_ptrs,
            cta2_idesc_m_dim_override,
            cta2_idesc_n_dim_override,
            cluster_dim_x));
#else
        TORCH_CHECK(false, "Experimental cta_group::2 + UnrollN=2 requires NVFP4_GROUP_GEMM_TMEM_COLUMNS=512.");
#endif
      } else {
        cudaFuncAttributes func_attr{};
        AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<2, 1, 64, false>));
        const int max_dynamic_smem =
            (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                : 0;
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_tcgen05_kernel<2, 1, 64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            max_dynamic_smem));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_tcgen05_kernel<2, 1, 64, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        AT_CUDA_CHECK(cudaFuncSetAttribute(
            nvfp4_group_gemm_tcgen05_kernel<2, 1, 64, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

        AT_CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            nvfp4_group_gemm_tcgen05_kernel<2, 1, 64, false>,
            a_ptrs_dev,
            b_ptrs_dev,
            sfa_ptrs_dev,
            sfb_ptrs_dev,
            c_ptrs_dev,
            m_sizes.data_ptr<int32_t>(),
            n_sizes.data_ptr<int32_t>(),
            k_halves.data_ptr<int32_t>(),
            k_scales.data_ptr<int32_t>(),
            a_descs_dev,
            b_descs_dev,
            sfa_descs_dev,
            sfb_descs_dev,
            /*cta_group_idx_map=*/nullptr,
            /*cta_tile_m_map=*/nullptr,
            /*cta_tile_n_map=*/nullptr,
            cta2_desc_a_row_offset_rows,
            cta2_desc_b_row_offset_rows,
            cta2_desc_sfa_row_offset_rows,
            cta2_epilogue_row_base_rows,
            cta2_epilogue_addr_mode,
            cta2_sfb_slot_mode,
            cta2_tmem_c_word_offset,
            cta2_tmem_sf_word_offset_eff,
	            cta2_tmem_sf_rank_word_offset_eff,
	            cta2_tsfa_word_offset,
	            cta2_tsfb_word_offset,
	            cta2_sfa_sf_id,
	            cta2_sfb_sf_id,
	            debug_tmem_dump,
	            debug_tmem_only_rank,
	            debug_tmem_idx_add,
            cta2_partition_b,
            debug_print_ptrs,
            cta2_idesc_m_dim_override,
            cta2_idesc_n_dim_override,
            cluster_dim_x));
      }
    } else {
      if (unroll_n == 2) {
#if NVFP4_GROUP_GEMM_TMEM_COLUMNS >= 512
        if (enable_tma_multicast != 0) {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, true>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              /*cta_group_idx_map=*/nullptr,
              /*cta_tile_m_map=*/nullptr,
              /*cta_tile_n_map=*/nullptr,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        } else {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              /*cta_group_idx_map=*/nullptr,
              /*cta_tile_m_map=*/nullptr,
              /*cta_tile_n_map=*/nullptr,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        }
#else
        TORCH_CHECK(false, "Cluster-mode UnrollN=2 requires NVFP4_GROUP_GEMM_TMEM_COLUMNS >= 512.");
#endif
      } else {
        if (enable_tma_multicast != 0) {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, true>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              /*cta_group_idx_map=*/nullptr,
              /*cta_tile_m_map=*/nullptr,
              /*cta_tile_n_map=*/nullptr,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        } else {
          cudaFuncAttributes func_attr{};
          AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>));
          const int max_dynamic_smem =
              (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
                  ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
                  : 0;
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
              max_dynamic_smem));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared));
          AT_CUDA_CHECK(cudaFuncSetAttribute(
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

          AT_CUDA_CHECK(cudaLaunchKernelEx(
              &cfg,
              nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>,
              a_ptrs_dev,
              b_ptrs_dev,
              sfa_ptrs_dev,
              sfb_ptrs_dev,
              c_ptrs_dev,
              m_sizes.data_ptr<int32_t>(),
              n_sizes.data_ptr<int32_t>(),
              k_halves.data_ptr<int32_t>(),
              k_scales.data_ptr<int32_t>(),
              a_descs_dev,
              b_descs_dev,
              sfa_descs_dev,
              sfb_descs_dev,
              /*cta_group_idx_map=*/nullptr,
              /*cta_tile_m_map=*/nullptr,
              /*cta_tile_n_map=*/nullptr,
              cta2_desc_a_row_offset_rows,
              cta2_desc_b_row_offset_rows,
              cta2_desc_sfa_row_offset_rows,
              cta2_epilogue_row_base_rows,
              cta2_epilogue_addr_mode,
              cta2_sfb_slot_mode,
              cta2_tmem_c_word_offset,
              cta2_tmem_sf_word_offset,
	              cta2_tmem_sf_rank_word_offset,
	              cta2_tsfa_word_offset,
	              cta2_tsfb_word_offset,
	              cta2_sfa_sf_id,
	              cta2_sfb_sf_id,
	              debug_tmem_dump,
	              debug_tmem_only_rank,
	              debug_tmem_idx_add,
              cta2_partition_b,
              debug_print_ptrs,
              cta2_idesc_m_dim_override,
              cta2_idesc_n_dim_override,
              cluster_dim_x));
        }
      }
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return;
  }

  // Default (no cluster): launch exactly the CTAs required by each group, matching GPU MODE's
  // packed-CTA mapping to avoid extra early-return blocks for small M/N groups.
  TORCH_CHECK(cta_group_idx_map.numel() > 0, "cta_group_idx_map must be non-empty for non-cluster launch");
  TORCH_CHECK(cta_tile_m_map.numel() == cta_group_idx_map.numel(), "cta_tile_m_map length mismatch");
  TORCH_CHECK(cta_tile_n_map.numel() == cta_group_idx_map.numel(), "cta_tile_n_map length mismatch");
  const int total_ctas = static_cast<int>(cta_group_idx_map.numel());
  const dim3 grid(1u, 1u, static_cast<unsigned int>(total_ctas));

#if NVFP4_GROUP_GEMM_TMEM_COLUMNS == 512
  if (unroll_n == 2) {
    {
      cudaFuncAttributes func_attr{};
      AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>));
      const int max_dynamic_smem =
          (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
              ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
              : 0;
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          max_dynamic_smem));
    }
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    nvfp4_group_gemm_tcgen05_kernel<1, 2, 128, false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        a_ptrs_dev,
        b_ptrs_dev,
        sfa_ptrs_dev,
        sfb_ptrs_dev,
        c_ptrs_dev,
        m_sizes.data_ptr<int32_t>(),
        n_sizes.data_ptr<int32_t>(),
        k_halves.data_ptr<int32_t>(),
        k_scales.data_ptr<int32_t>(),
        a_descs_dev,
        b_descs_dev,
        sfa_descs_dev,
        sfb_descs_dev,
        cta_group_idx_dev,
        cta_tile_m_dev,
        cta_tile_n_dev,
        cta2_desc_a_row_offset_rows,
        cta2_desc_b_row_offset_rows,
        cta2_desc_sfa_row_offset_rows,
        cta2_epilogue_row_base_rows,
        cta2_epilogue_addr_mode,
        cta2_sfb_slot_mode,
        cta2_tmem_c_word_offset,
        cta2_tmem_sf_word_offset,
	        cta2_tmem_sf_rank_word_offset,
	        cta2_tsfa_word_offset,
	        cta2_tsfb_word_offset,
	        cta2_sfa_sf_id,
	        cta2_sfb_sf_id,
	        debug_tmem_dump,
	        debug_tmem_only_rank,
	        debug_tmem_idx_add,
        cta2_partition_b,
        debug_print_ptrs,
        cta2_idesc_m_dim_override,
        cta2_idesc_n_dim_override,
        /*cluster_dim_x=*/1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#endif

  {
    cudaFuncAttributes func_attr{};
    AT_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>));
    const int max_dynamic_smem =
        (max_shared_optin > static_cast<int>(func_attr.sharedSizeBytes))
            ? (max_shared_optin - static_cast<int>(func_attr.sharedSizeBytes))
            : 0;
    AT_CUDA_CHECK(cudaFuncSetAttribute(
        nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_dynamic_smem));
  }
  AT_CUDA_CHECK(cudaFuncSetAttribute(
      nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false>, cudaFuncAttributePreferredSharedMemoryCarveout,
      cudaSharedmemCarveoutMaxShared));
  nvfp4_group_gemm_tcgen05_kernel<1, 1, 128, false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      a_ptrs_dev,
      b_ptrs_dev,
      sfa_ptrs_dev,
      sfb_ptrs_dev,
      c_ptrs_dev,
      m_sizes.data_ptr<int32_t>(),
      n_sizes.data_ptr<int32_t>(),
      k_halves.data_ptr<int32_t>(),
      k_scales.data_ptr<int32_t>(),
      a_descs_dev,
      b_descs_dev,
      sfa_descs_dev,
      sfb_descs_dev,
      cta_group_idx_dev,
      cta_tile_m_dev,
      cta_tile_n_dev,
      cta2_desc_a_row_offset_rows,
      cta2_desc_b_row_offset_rows,
      cta2_desc_sfa_row_offset_rows,
      cta2_epilogue_row_base_rows,
      cta2_epilogue_addr_mode,
      cta2_sfb_slot_mode,
      cta2_tmem_c_word_offset,
      cta2_tmem_sf_word_offset,
	      cta2_tmem_sf_rank_word_offset,
	      cta2_tsfa_word_offset,
	      cta2_tsfb_word_offset,
	      cta2_sfa_sf_id,
	      cta2_sfb_sf_id,
	      debug_tmem_dump,
	      debug_tmem_only_rank,
	      debug_tmem_idx_add,
      cta2_partition_b,
      debug_print_ptrs,
      cta2_idesc_m_dim_override,
      cta2_idesc_n_dim_override,
      /*cluster_dim_x=*/1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void nvfp4_group_gemm_utccp_cta2_smoke_cuda(
    torch::Tensor out_words, int pattern_kind, int use64, int schedule, int stage) {
  TORCH_CHECK(out_words.is_cuda(), "out_words must be a CUDA tensor");
  TORCH_CHECK(out_words.scalar_type() == torch::kInt, "out_words must be torch.int32");
  TORCH_CHECK(out_words.is_contiguous(), "out_words must be contiguous");

  uint32_t* out_ptr = nullptr;
  if (out_words.numel() > 0) {
    const int dump_elems = 2 * kUtccpSmokeDumpDp * kUtccpSmokeDumpCol;
    TORCH_CHECK(out_words.numel() >= dump_elems,
                "out_words must have at least ",
                dump_elems,
                " elements (2 ranks * ",
                kUtccpSmokeDumpDp,
                " dp * ",
                kUtccpSmokeDumpCol,
                " cols) or be empty to disable dumping");
    out_ptr = reinterpret_cast<uint32_t*>(out_words.data_ptr<int32_t>());
  }

  // Cluster launches require explicitly allowing non-portable cluster sizes on some drivers.
  AT_CUDA_CHECK(cudaFuncSetAttribute(
      nvfp4_group_gemm_utccp_cta2_smoke_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(2, 1, 1);           // one cluster (2 CTAs)
  cfg.blockDim = dim3(128, 1, 1);        // 4 warps
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  AT_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg,
      nvfp4_group_gemm_utccp_cta2_smoke_kernel,
      out_ptr,
      pattern_kind,
      use64,
      schedule,
      stage));
  AT_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nvfp4_group_gemm_forward_grouped_cuda", &nvfp4_group_gemm_forward_grouped_cuda);
  m.def("nvfp4_group_gemm_build_ab_tma_descs_cuda", &build_ab_tma_descs_cuda);
  m.def("nvfp4_group_gemm_build_scale_tma_descs_cuda", &build_scale_tma_descs_cuda);
  m.def("nvfp4_group_gemm_forward_grouped_tcgen05_cuda", &nvfp4_group_gemm_forward_grouped_tcgen05_cuda);
  m.def("nvfp4_group_gemm_utccp_cta2_smoke_cuda", &nvfp4_group_gemm_utccp_cta2_smoke_cuda);
}
