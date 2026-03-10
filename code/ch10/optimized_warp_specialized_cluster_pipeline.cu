// Chapter 10 optimized sample: warp-specialized cluster pipeline with DSMEM.
//
// WHAT:
//   - A leader CTA stages A/B tiles through a block-scoped cuda::pipeline
//   - Cluster peers read the leader's shared-memory tiles through DSMEM
//   - Dedicated compute/store warps operate on disjoint row bands per block
//
// WHY THIS IS FASTER:
//   - The leader overlaps staging with the pipeline rather than plain loads
//   - Each tile is fetched once per cluster, not once per block
//   - Explicit CTA/cluster handoffs keep the collectives legal and the roles simple

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 13)

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include "../core/common/headers/cuda_verify.cuh"
#include "../core/common/nvtx_utils.cuh"

namespace cg = cooperative_groups;

namespace {

constexpr int TILE_SIZE = 128;
constexpr int TILE_ELEMS = TILE_SIZE * TILE_SIZE;
constexpr int CLUSTER_BLOCKS = 4;
constexpr int WARPS_PER_BLOCK = 3;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int DEFAULT_TILES = 8;
constexpr int WARMUP_ITERS = 5;
constexpr int BENCH_ITERS = 20;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t _status = (call);                                            \
        if (_status != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(_status));       \
            std::abort();                                                        \
        }                                                                        \
    } while (0)

__device__ void compute_rows_from_ds(const float* __restrict__ A_src,
                                     const float* __restrict__ B_src,
                                     float* __restrict__ C_dst,
                                     int row_begin,
                                     int row_end,
                                     int lane_id) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
        for (int col = 0; col < TILE_SIZE; ++col) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                acc += A_src[row * TILE_SIZE + k] * B_src[k * TILE_SIZE + col];
            }
            C_dst[row * TILE_SIZE + col] = acc;
        }
    }
}

__global__ void dsmem_probe_kernel(float* out) {
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float buffer[32];
    const int cluster_rank = cluster.block_rank();

    if (cluster_rank == 0 && threadIdx.x == 0) {
        buffer[0] = 123.0f;
    }
    cluster.sync();
    if (cluster_rank == 1 && threadIdx.x == 0) {
        float* remote = cluster.map_shared_rank(buffer, 0);
        out[0] = remote[0];
    }
    cluster.sync();
}

struct ProbeResult {
    bool ok;
    const char* stage;
    cudaError_t error;
};

ProbeResult probe_dsmem_support() {
    float* d_out = nullptr;
    cudaError_t alloc_status = cudaMalloc(&d_out, sizeof(float));
    if (alloc_status != cudaSuccess) {
        return {false, "cudaMalloc", alloc_status};
    }

    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(2);
    cfg.blockDim = dim3(32);
    cfg.dynamicSmemBytes = 0;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    void* args[] = {&d_out};
    cudaError_t launch_status = cudaLaunchKernelExC(&cfg, (void*)dsmem_probe_kernel, args);
    if (launch_status != cudaSuccess) {
        cudaFree(d_out);
        return {false, "cudaLaunchKernelExC", launch_status};
    }

    cudaError_t sync_status = cudaDeviceSynchronize();
    cudaFree(d_out);
    if (sync_status != cudaSuccess) {
        return {false, "cudaDeviceSynchronize", sync_status};
    }
    return {true, nullptr, cudaSuccess};
}

extern "C" __global__ void optimized_warp_specialized_cluster_pipeline_kernel(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int num_tiles) {
    cg::thread_block cta = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();

    extern __shared__ float shared_mem[];
    float* A_tile_local = shared_mem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;

    using pipe_state_t = cuda::pipeline_shared_state<cuda::thread_scope_block, 1>;
    __shared__ alignas(pipe_state_t) unsigned char pipe_storage[sizeof(pipe_state_t)];
    auto* pipe_state = reinterpret_cast<pipe_state_t*>(pipe_storage);
    if (threadIdx.x == 0) {
        new (pipe_state) pipe_state_t();
    }
    cta.sync();
    auto pipe = cuda::make_pipeline(cta, pipe_state);
    auto warp = cg::tiled_partition<32>(cta);

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int cluster_rank = cluster.block_rank();
    const dim3 cluster_dims = cluster.dim_blocks();
    const int blocks_in_cluster =
        cluster_dims.x * cluster_dims.y * cluster_dims.z;
    const size_t tile_bytes = static_cast<size_t>(TILE_ELEMS) * sizeof(float);

    for (int tile = blockIdx.x / cluster_dims.x; tile < num_tiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;

        // The block-scoped collectives stay uniform within the leader CTA.
        if (cluster_rank == 0) {
            pipe.producer_acquire();
            if (warp_id == 0) {
                cuda::memcpy_async(
                    warp,
                    A_tile_local,
                    A_global + offset,
                    cuda::aligned_size_t<16>(tile_bytes),
                    pipe);
                cuda::memcpy_async(
                    warp,
                    B_tile_local,
                    B_global + offset,
                    cuda::aligned_size_t<16>(tile_bytes),
                    pipe);
            }
            pipe.producer_commit();
            pipe.consumer_wait();
            pipe.consumer_release();
        }

        cluster.sync();

        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);

        const int rows_per_block =
            (TILE_SIZE + blocks_in_cluster - 1) / blocks_in_cluster;
        const int row_begin = min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end = min(row_begin + rows_per_block, TILE_SIZE);

        if (warp_id == 1) {
            compute_rows_from_ds(A_src, B_src, C_tile_local, row_begin, row_end, lane_id);
        }

        cta.sync();

        if (warp_id == 2) {
            for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
        }

        cluster.sync();
    }
}

void initialize_inputs(std::vector<float>& h_A, std::vector<float>& h_B) {
    for (size_t i = 0; i < h_A.size(); ++i) {
        h_A[i] = static_cast<float>((i % TILE_SIZE) * 0.5f);
        h_B[i] = static_cast<float>((i % TILE_SIZE) * 0.25f + 1.0f);
    }
}

double verify_output(const std::vector<float>& h_A,
                     const std::vector<float>& h_B,
                     const std::vector<float>& h_C,
                     int num_tiles) {
    double max_err = 0.0;
    for (int tile = 0; tile < num_tiles; ++tile) {
        const size_t tile_offset = static_cast<size_t>(tile) * TILE_ELEMS;
        for (int row = 0; row < TILE_SIZE; ++row) {
            for (int col = 0; col < TILE_SIZE; ++col) {
                float ref = 0.0f;
                for (int k = 0; k < TILE_SIZE; ++k) {
                    ref += h_A[tile_offset + row * TILE_SIZE + k] *
                           h_B[tile_offset + k * TILE_SIZE + col];
                }
                max_err = std::max(
                    max_err,
                    static_cast<double>(
                        std::abs(ref - h_C[tile_offset + row * TILE_SIZE + col])));
            }
        }
    }
    return max_err;
}

int run_optimized(int num_tiles) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int cluster_launch = 0;
#ifdef cudaDevAttrClusterLaunch
    CUDA_CHECK(cudaDeviceGetAttribute(&cluster_launch, cudaDevAttrClusterLaunch, 0));
#endif
    const bool supports_clusters = (cluster_launch > 0) || (prop.major >= 9);
    if (!supports_clusters) {
        std::fprintf(stderr,
                     "SKIPPED: Warp specialization cluster pipelines require thread block cluster hardware support on %s (SM %d.%d).\n",
                     prop.name,
                     prop.major,
                     prop.minor);
        return 3;
    }

    ProbeResult probe = probe_dsmem_support();
    if (!probe.ok) {
        std::fprintf(stderr,
                     "SKIPPED: Distributed shared memory unavailable on %s (SM %d.%d). Stage=%s error=%s\n",
                     prop.name,
                     prop.major,
                     prop.minor,
                     probe.stage,
                     cudaGetErrorString(probe.error));
        return 3;
    }

    const size_t bytes = static_cast<size_t>(num_tiles) * TILE_ELEMS * sizeof(float);
    std::vector<float> h_A(bytes / sizeof(float));
    std::vector<float> h_B(bytes / sizeof(float));
    std::vector<float> h_C(bytes / sizeof(float), 0.0f);
    initialize_inputs(h_A, h_B);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const size_t shared_bytes = 3ull * TILE_ELEMS * sizeof(float);
    int max_dynamic_smem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_dynamic_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        0));
    if (shared_bytes > static_cast<size_t>(max_dynamic_smem)) {
        std::fprintf(stderr,
                     "SKIPPED: optimized_warp_specialized_cluster_pipeline requires %zu bytes of dynamic shared memory, but the device limit is %d bytes.\n",
                     shared_bytes,
                     max_dynamic_smem);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        return 3;
    }

    CUDA_CHECK(cudaFuncSetAttribute(
        optimized_warp_specialized_cluster_pipeline_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));
    CUDA_CHECK(cudaFuncSetAttribute(
        optimized_warp_specialized_cluster_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));

    const int clusters_in_grid = std::max(1, std::min(num_tiles, prop.multiProcessorCount));
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(clusters_in_grid * CLUSTER_BLOCKS);
    cfg.blockDim = dim3(THREADS_PER_BLOCK);
    cfg.dynamicSmemBytes = shared_bytes;
    cfg.stream = 0;

    cudaLaunchAttribute attrs[1]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CLUSTER_BLOCKS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        NVTX_RANGE("warmup");
        CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            optimized_warp_specialized_cluster_pipeline_kernel,
            d_A,
            d_B,
            d_C,
            num_tiles));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; ++i) {
        NVTX_RANGE("compute_kernel:optimized_warp_specialized_cluster_pipeline");
        CUDA_CHECK(cudaLaunchKernelEx(
            &cfg,
            optimized_warp_specialized_cluster_pipeline_kernel,
            d_A,
            d_B,
            d_C,
            num_tiles));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const float avg_ms = ms / BENCH_ITERS;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    const double max_err = verify_output(h_A, h_B, h_C, num_tiles);

    double checksum = 0.0;
    for (float value : h_C) {
        checksum += value;
    }

    std::printf("optimized_warp_specialized_cluster_pipeline: %d tiles, %.3f ms, checksum %.6f\n",
                num_tiles,
                avg_ms,
                checksum / h_C.size());
    std::printf("Verification complete (max error %.3e)\n", max_err);

#ifdef VERIFY
    VERIFY_PRINT_CHECKSUM(static_cast<float>(checksum));
#endif

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}

}  // namespace

int main() {
    NVTX_RANGE("main");
    return run_optimized(DEFAULT_TILES);
}

#else

int main() {
    std::fprintf(stderr, "SKIPPED: CUDA 13+ required for warp specialization cluster pipelines.\n");
    return 3;
}

#endif
