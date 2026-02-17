#include <cstdint>
#include <cstdio>

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

}  // namespace

int main() {
  using namespace cute;

  // Probe CUTLASS TMEM layouts for our cta_group::2 + UnrollN=2 case.
  //
  // Key question: for N_SM=2 block-scaled UMMA, how do the TMEM addresses vary with:
  // - rank (CTA rank in the 2SM MMA group)
  // - u (MN tile index when UnrollN=2)
  // - seg (K64 "MMA-K block" index within a K256 tile)
  //
  // In CUTLASS' `tmem_sf_frg`, the K256 tile is represented as `num_MMA_K=4` (four K64 MMAs).
  // The (VS=16, NSF=4) tuple represents the K64 scale-factor shape within a single MMA.
  using CFrag = UMMA::tmem_frg_2sm<float>;
  using SFAFrag =
      UMMA::tmem_sf_frg<cutlass::float_ue4m3_t, 16, 2, true, UMMA::TmemAllocMode::ScaleFactorDuplicated4by1>;
  using SFBFrag =
      UMMA::tmem_sf_frg<cutlass::float_ue4m3_t, 16, 2, false, UMMA::TmemAllocMode::ScaleFactorDuplicated4by1>;

  auto c = make_tensor<CFrag>(make_shape(make_shape(Int<128>{}, Int<128>{}), Int<1>{}, Int<2>{}, Int<2>{}));
  auto sfa =
      make_tensor<SFAFrag>(make_shape(make_shape(Int<128>{}, make_shape(Int<16>{}, Int<4>{})), Int<1>{}, Int<4>{}, Int<2>{}));
  auto sfb =
      make_tensor<SFBFrag>(make_shape(make_shape(Int<128>{}, make_shape(Int<16>{}, Int<4>{})), Int<2>{}, Int<4>{}, Int<2>{}));

  auto c_layout = c.layout();
  auto sfa_layout = sfa.layout();
  auto sfb_layout = sfb.layout();

  // CUTLASS uses this to place non-accumulator TMEM after the accumulator fragment:
  //   tmem_non_accumulator_base = tmem_base + find_tmem_tensor_col_offset(accumulators)
  // where find_tmem_tensor_col_offset() is:
  //   cosize(recast<uint32_t>(tensor).layout()) & 0xFFFF
  const uint32_t c_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(c).layout()) & 0xFFFF);
  const uint32_t sfa_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfa).layout()) & 0xFFFF);
  const uint32_t sfb_cols = static_cast<uint32_t>(cosize(recast<uint32_t>(sfb).layout()) & 0xFFFF);
  std::printf("tmem_colspan (masked): c=%u sfa=%u sfb=%u\n", c_cols, sfa_cols, sfb_cols);

  std::printf("C fragment encoded addresses (rank,u)\n");
  for (int rank = 0; rank < 2; ++rank) {
    for (int u = 0; u < 2; ++u) {
      uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, 0), 0, u, rank)));
      uint32_t addr = rotr32(off, 0);  // float32 -> OffsetShift=0
      std::printf("rank=%d u=%d off=%u addr=0x%08x\n", rank, u, off, addr);
    }
  }
  std::printf("C fragment dp/col samples (rank=0,u=0,n=0,m=0..3)\n");
  for (int m = 0; m < 4; ++m) {
    uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(m, 0), 0, 0, 0)));
    uint32_t addr = rotr32(off, 0);
    uint32_t col = addr & 0xFFFFu;
    uint32_t dp = (addr >> 16) & 0xFFu;
    uint32_t idx = (addr >> 24) & 0xFFu;
    std::printf("m=%d off=%u addr=0x%08x col=%u dp=%u idx=%u\n", m, off, addr, col, dp, idx);
  }
  std::printf("C fragment col span (m=0, n=0..127) per (rank,u)\n");
  for (int rank = 0; rank < 2; ++rank) {
    for (int u = 0; u < 2; ++u) {
      uint32_t col_min = 0xFFFFFFFFu;
      uint32_t col_max = 0;
      for (int n = 0; n < 128; ++n) {
        uint32_t off = static_cast<uint32_t>(c_layout(make_coord(make_coord(0, n), 0, u, rank)));
        uint32_t addr = rotr32(off, 0);
        uint32_t col = addr & 0xFFFFu;
        col_min = (col < col_min) ? col : col_min;
        col_max = (col > col_max) ? col : col_max;
      }
      std::printf("rank=%d u=%d col_min=%u col_max=%u\n", rank, u, col_min, col_max);
    }
  }

  // Probe SFA: vary `seg` as the K64 block index (num_MMA_K).
  // Use the base coordinate within the (VS, NSF) tuple: (vs=0, nsf=0).
  std::printf("SFA fragment encoded addresses (rank,seg)\n");
  for (int rank = 0; rank < 2; ++rank) {
    for (int seg = 0; seg < 4; ++seg) {
      uint32_t off =
          static_cast<uint32_t>(sfa_layout(make_coord(make_coord(0, make_coord(0, 0)), 0, seg, rank)));
      uint32_t addr = rotr32(off, 2);  // ue4m3 (8-bit) -> OffsetShift=2
      std::printf("rank=%d seg=%d off=%u addr=0x%08x\n", rank, seg, off, addr);
    }
  }
  std::printf("SFA fragment dp/col samples (rank=0,seg=0,vs=0,nsf=0,m=0..3)\n");
  for (int m = 0; m < 4; ++m) {
    uint32_t off =
        static_cast<uint32_t>(sfa_layout(make_coord(make_coord(m, make_coord(0, 0)), 0, 0, 0)));
    uint32_t addr = rotr32(off, 2);
    uint32_t col = addr & 0xFFFFu;
    uint32_t dp = (addr >> 16) & 0xFFu;
    uint32_t idx = (addr >> 24) & 0xFFu;
    std::printf("m=%d off=%u addr=0x%08x col=%u dp=%u idx=%u\n", m, off, addr, col, dp, idx);
  }

  // Probe SFB: vary u (MN tile) and seg (K64 block index).
  // Use the base coordinate within the (VS, NSF) tuple: (vs=0, nsf=0).
  std::printf("SFB fragment encoded addresses (rank,u,seg)\n");
  for (int rank = 0; rank < 2; ++rank) {
    for (int u = 0; u < 2; ++u) {
      for (int seg = 0; seg < 4; ++seg) {
        uint32_t off =
            static_cast<uint32_t>(sfb_layout(make_coord(make_coord(0, make_coord(0, 0)), u, seg, rank)));
        uint32_t addr = rotr32(off, 2);  // ue4m3 (8-bit) -> OffsetShift=2
        std::printf("rank=%d u=%d seg=%d off=%u addr=0x%08x\n", rank, u, seg, off, addr);
      }
    }
  }
  std::printf("SFB fragment dp/col samples (rank=0,u=0,seg=0,vs=0,nsf=0,m=0..3)\n");
  for (int m = 0; m < 4; ++m) {
    uint32_t off =
        static_cast<uint32_t>(sfb_layout(make_coord(make_coord(m, make_coord(0, 0)), 0, 0, 0)));
    uint32_t addr = rotr32(off, 2);
    uint32_t col = addr & 0xFFFFu;
    uint32_t dp = (addr >> 16) & 0xFFu;
    uint32_t idx = (addr >> 24) & 0xFFu;
    std::printf("m=%d off=%u addr=0x%08x col=%u dp=%u idx=%u\n", m, off, addr, col, dp, idx);
  }

  return 0;
}
