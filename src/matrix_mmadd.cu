#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <type_traits>
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "matrix_size_checks.hpp"
#include "types"
#include "utils.hpp"

// AccumNative => internal accumulation is done in T, else 32bit(T) if sizeof(T)
// < 4 else 64bit(T)
template <typename T, uint32 TILE_SZ = 16, bool AccumNative = false, typename PProcess>
__global__ void tiled_mmadd_shmem(Matrix<T> result, const Matrix<T> A, const Matrix<T> B,
                                  const Optional<Matrix<T>> C, PProcess pprocess)
{
    using SumType = typename std::conditional<AccumNative, T, typename AccumT<T>::type>::type;
    SumType sum{0};

    __shared__ T As[TILE_SZ][TILE_SZ + 1];
    __shared__ T Bs[TILE_SZ][TILE_SZ + 1];

    uint32 x = blockIdx.x * TILE_SZ + threadIdx.x;
    uint32 y = blockIdx.y * TILE_SZ + threadIdx.y;
    uint32 b = blockIdx.z;
    uint32 b_a = A.batch() > 1 ? b : 0;
    uint32 b_b = B.batch() > 1 ? b : 0;

    uint32 k_max = iDivUp(A.width(), TILE_SZ) * TILE_SZ;
#pragma unroll
    for (uint32 k = 0; k < k_max; k += TILE_SZ)
    {
        auto x_a = k + threadIdx.y;
        auto y_b = k + threadIdx.x;
        auto aa = (x_a < A.width() && x < A.height()) ? A(b_a, x, x_a) : T(0);
        auto bb = (y_b < B.height() && y < B.width()) ? B(b_b, y_b, y) : T(0);

        As[threadIdx.x][threadIdx.y] = aa;
        Bs[threadIdx.x][threadIdx.y] = bb;

        __syncthreads();
#pragma unroll
        for (uint32 kk = 0; kk < TILE_SZ; kk++)
        {
            sum += SumType(As[threadIdx.x][kk] * Bs[kk][threadIdx.y]);
        }
        __syncthreads();
    }

    if (x < result.height() and y < result.width())
    {
        sum += (C.is_valid() ? C->template broadcasting_fetch<0b111>(b, x, y) : T(0));
        result(b, x, y) = pprocess(sum);
    }
}

// A (b, h, w) @ B (b, w, 1) + <C(b, h,1)> -> result (b, h, 1)
// Assuming that B is column vector, if h > 1024, multiple calls to this kernel
template <typename T, uint32 BLOCK_X, typename PostProcess>
__global__ void mat_vector_mul_kernel(Matrix<T> result, const Matrix<T> A, const Matrix<T> B,
                                      const Optional<Matrix<T>> C, uint32 offset = 0,
                                      bool addToresult = false,
                                      PostProcess pProcess = Identity<T>())
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.y;
    uint32 b = blockIdx.z;

    __shared__ T As[BLOCK_X + 1];

    uint32 offset_x = x + offset;

    As[threadIdx.x] = (offset_x < A.width()) ? T(A(b, y, offset_x) * B(b, offset_x, 0)) : T(0);

    __syncthreads();

    T r = (addToresult ? result(b, y, 0) : T(0));
#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < s) As[x] += As[x + s];
        __syncthreads();
    }
    __syncthreads();
    volatile T* vAs = (volatile T*)As;
    if (x <= 32 and offset_x < A.width())
    {
        if (BLOCK_X >= 64) vAs[x] += vAs[x + 32];
        if (BLOCK_X >= 32) vAs[x] += vAs[x + 16];
        if (BLOCK_X >= 16) vAs[x] += vAs[x + 8];
        if (BLOCK_X >= 8) vAs[x] += vAs[x + 4];
        if (BLOCK_X >= 4) vAs[x] += vAs[x + 2];
        if (BLOCK_X >= 2) vAs[x] += vAs[x + 1];
    }
    __syncthreads();
    if (x == 0 and blockIdx.x < A.height())
    {
        if (C.is_valid()) r += (*C)(b, y, x);
        result(b, y, 0) = pProcess(vAs[0] + r);
    }
}

template <typename T, typename PProcess>
void mmadd(Matrix<T>& result, const Matrix<T>& A, const Matrix<T>& B, const Optional<Matrix<T>> C,
           PProcess pProcess)
{
    check_mmadd_sizes(result, A, B, C);
    if (A.height() <= 512)
    {
        constexpr uint32 TILE_SZ = 8;
        dim3 blockDim(TILE_SZ, TILE_SZ);
        dim3 gridDim(iDivUp(result.height(), TILE_SZ), iDivUp(result.width(), TILE_SZ),
                     result.batch());
        tiled_mmadd_shmem<T, TILE_SZ, false><<<gridDim, blockDim>>>(result, A, B, C, pProcess);
    }
    else if (A.height() <= 1024)
    {
        constexpr uint32 TILE_SZ = 16;
        dim3 blockDim(TILE_SZ, TILE_SZ);
        dim3 gridDim(iDivUp(result.height(), TILE_SZ), iDivUp(result.width(), TILE_SZ),
                     result.batch());
        tiled_mmadd_shmem<T, TILE_SZ, false><<<gridDim, blockDim>>>(result, A, B, C, pProcess);
    }
    else
    {
        constexpr uint32 TILE_SZ = 32;
        dim3 blockDim(TILE_SZ, TILE_SZ);
        dim3 gridDim(iDivUp(result.height(), TILE_SZ), iDivUp(result.width(), TILE_SZ),
                     result.batch());
        tiled_mmadd_shmem<T, TILE_SZ, false><<<gridDim, blockDim>>>(result, A, B, C, pProcess);
    }
    cudaErrCheck(cudaGetLastError());
}

template void mmadd<FloatT, Sigmoid<FloatT>::SigmoidF>(Matrix<FloatT>&, const Matrix<FloatT>&,
                                                       const Matrix<FloatT>&,
                                                       const Optional<Matrix<FloatT>>,
                                                       Sigmoid<FloatT>::SigmoidF);
template void mmadd<FloatT, Identity<FloatT>>(Matrix<FloatT>&, const Matrix<FloatT>&,
                                              const Matrix<FloatT>&, const Optional<Matrix<FloatT>>,
                                              Identity<FloatT>);

template void mmadd<FloatT, Relu<FloatT>::ReluF>(Matrix<FloatT>&, const Matrix<FloatT>&,
                                                 const Matrix<FloatT>&,
                                                 const Optional<Matrix<FloatT>>,
                                                 Relu<FloatT>::ReluF);

template void mmadd<FloatT, TanH<FloatT>::TanhF>(Matrix<FloatT>&, const Matrix<FloatT>&,
                                                 const Matrix<FloatT>&,
                                                 const Optional<Matrix<FloatT>>,
                                                 TanH<FloatT>::TanhF);

template void mmadd<FloatT, DividedBy<FloatT>>(Matrix<FloatT>&, const Matrix<FloatT>&,
                                               const Matrix<FloatT>&,
                                               const Optional<Matrix<FloatT>>, DividedBy<FloatT>);

template void mmadd<FloatT, Neg<FloatT>>(Matrix<FloatT>&, const Matrix<FloatT>&,
                                         const Matrix<FloatT>&, const Optional<Matrix<FloatT>>,
                                         Neg<FloatT>);

template void mmadd<FloatT, Composition<FloatT, Neg<FloatT>, DividedBy<FloatT>>>(
    Matrix<FloatT>&, const Matrix<FloatT>&, const Matrix<FloatT>&, const Optional<Matrix<FloatT>>,
    Composition<FloatT, Neg<FloatT>, DividedBy<FloatT>>);

template void mmadd<FloatT, Composition<FloatT, Neg<FloatT>, Identity<FloatT>>>(
    Matrix<FloatT>&, const Matrix<FloatT>&, const Matrix<FloatT>&, const Optional<Matrix<FloatT>>,
    Composition<FloatT, Neg<FloatT>, Identity<FloatT>>);
