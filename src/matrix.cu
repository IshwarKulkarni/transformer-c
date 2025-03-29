#include <sys/types.h>
#include "matrix.cuh"
#include "matrix_ops.hpp"
#include "nodes/node.hpp"

static constexpr uint32 det_seed = 42;

uint64 MatrixInitUitls::alloced_bytes = 0;
uint64 MatrixInitUitls::freed_bytes = 0;
uint32 MatrixInitUitls::id = 0;
std::map<uint32, uint64> MatrixInitUitls::id_to_alloced_bytes;

std::random_device rdm::rd;
std::mt19937_64 rdm::det_gen(det_seed);
std::seed_seq rdm::seed({rdm::rd()});
std::mt19937_64 rdm::rdm_gen(seed);
bool rdm::deterministic = true;

uint64 ParameterBase::param_count = 0;

std::vector<NodeBase *> NodeBase::all_nodes;
std::vector<const MatrixBase *> MatrixBase::all_matrices;
std::vector<ParameterBase *> ParameterBase::all_params;

template <uint32 n_writes>
__global__ void write_dead_beef(uint32 size, uint32 *ptr)
{
    uint32 idx = threadIdx.x + blockIdx.x * blockDim.x;
    ptr += idx * n_writes;
    if (idx + n_writes < size)
    {
        for (uint32 i = 0; i < n_writes; ++i)
        {
            ptr[i] = 0xdeadbeef;
        }
    }
}

void clear_l2_cache(uint32 size_bytes)
{
    cudaErrCheck(cudaDeviceSynchronize());
    void *d_data;
    cudaErrCheck(cudaMalloc(&d_data, size_bytes));
    cudaErrCheck(cudaMemset(d_data, 1, size_bytes));

    uint32 n_elems = size_bytes / sizeof(uint32);
    uint32 *u_data = reinterpret_cast<uint32 *>(d_data);

    static constexpr uint32 n_writes = 8;
    dim3 gridDim(iDivUp(n_elems / n_writes, 1024));
    write_dead_beef<n_writes><<<gridDim, 1024>>>(n_elems, u_data);
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaFree(d_data));
}
