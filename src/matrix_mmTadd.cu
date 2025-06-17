/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <type_traits>
#include "matrix.cuh"
#include "matrix_ops.hpp"
#include "matrix_size_checks.hpp"
#include "types"

// computes the A * B^T + C
// AccumNative => internal accumulation is done in T, else 32bit(T) if sizeof(T)
// < 4 else 64bit(T).
template <uint32 TILE_SIZE_X = 16, bool AccumNative = false, typename T, typename PostProcess>
__global__ void tiled_mmTadd_shmem(Matrix<T> result, const Matrix<T> A, const Matrix<T> B,
                                   const Optional<Matrix<T>> C, PostProcess pprocess)
{
    using SumType = typename std::conditional<AccumNative, T, typename AccumT<T>::type>::type;
    SumType sum{0};

    __shared__ T As[TILE_SIZE_X][TILE_SIZE_X + 1];
    __shared__ T Bs[TILE_SIZE_X][TILE_SIZE_X + 1];

    uint32 y = blockIdx.x * TILE_SIZE_X + threadIdx.x;
    uint32 x = blockIdx.y * TILE_SIZE_X + threadIdx.y;
    uint32 b = blockIdx.z;
    uint32 b_a = A.batch() > 1 ? b : 0;
    uint32 b_b = B.batch() > 1 ? b : 0;

    uint32 k_max = iDivUp(A.width(), TILE_SIZE_X) * TILE_SIZE_X;
#pragma unroll
    for (uint32 k = 0; k < k_max; k += TILE_SIZE_X)
    {
        uint32 x_a = k + threadIdx.y;
        uint32 x_b = k + threadIdx.x;

        auto aa = A.in_extents(b_a, y, x_a) ? A(b_a, y, x_a) : T(0);
        auto bb = B.in_extents(b_b, x, x_b) ? B(b_b, x, x_b) : T(0);

        As[threadIdx.x][threadIdx.y] = aa;
        Bs[threadIdx.x][threadIdx.y] = bb;

        __syncthreads();
#pragma unroll
        for (uint32 kk = 0; kk < TILE_SIZE_X; kk++)
        {
            sum += SumType(As[threadIdx.x][kk] * Bs[kk][threadIdx.y]);
        }
        __syncthreads();
    }

    if (x < result.width() && y < result.height())
    {
        sum += (C.is_valid() ? C->template broadcasting_fetch<0b111>(b, y, x) : T(0));
        auto out = pprocess(sum);
        NAN_INF_CHECK(out);
        result(b, y, x) = out;
    }
}

template <typename T, typename PostProcess>
__global__ void mmTadd_kernel(Matrix<T> result, const Matrix<T> A, const Matrix<T> B,
                              const Optional<Matrix<T>> C, PostProcess pProcess = Identity<T>())
{
    uint32 y = threadIdx.x;
    uint32 x = threadIdx.y;
    uint32 b = threadIdx.z + blockIdx.x * blockDim.z;
    uint32 b_a = A.batch() > 1 ? b : 0;
    uint32 b_b = B.batch() > 1 ? b : 0;

    if (result.is_oob(b, y, x)) return;

    T sum = 0;
    uint32 a_ext = A.template get_extent<WIDTH_IDX>(b_a);
    uint32 b_ext = B.template get_extent<HEIGHT_IDX>(b_b);
#pragma unroll
    for (uint32 k = 0; k < a_ext; k++)
    {
        auto aa = A.in_extents(b_a, y, k) ? A(b_a, y, k) : T(0);
        auto bb = B.in_extents(b_b, x, k) ? B(b_b, x, k) : T(0);
        sum += aa * bb;
    }

    static constexpr uint32 bits = BATCH_BIT | WIDTH_BIT | HEIGHT_BIT;
    sum += (C.is_valid() ? C->template broadcasting_fetch<bits>(b, y, x) : T(0));
    auto out = pProcess(sum);
    NAN_INF_CHECK(out);
    result(b, y, x) = out;
}

template <typename T, typename PProcess>
void mmTadd(Matrix<T>& result, const Matrix<T>& A, const Matrix<T>& B, const Optional<Matrix<T>> C,
            PProcess pProcess)
{
    LOG_MATRIX_OPS("mmTadd: ", A.shape, " @ ", B.shape, "^T", (C.is_valid() ? " + C" : " "), " -> ",
                   result.shape);
    check_mmTadd_sizes(result, A, B, C);

    for (uint32 b = 0; b < result.batch(); b++)
    {
        auto a_ext =
            A.template get_extent<HEIGHT_IDX>(b >= A.batch() ? 0 : b);  // broadcasted batch
        auto b_ext =
            B.template get_extent<HEIGHT_IDX>(b >= B.batch() ? 0 : b);  // broadcasted batch
        result.set_extents(b, a_ext, b_ext);
    }

    if (result.numels() <= 256)
    {
        dim3 blockDim(A.height(), B.height(), result.batch());
        LOG_KERNEL_SIZE("A.shape: ", A.shape, " B.shape: ", B.shape,
                        (C ? " C: " + C->shape.str() : "no C, "), " result.shape: ", result.shape,
                        " gridDim: ", 1, " blockDim: ", blockDim);
        mmTadd_kernel<T><<<1, blockDim>>>(result, A, B, C, pProcess);
    }
    else if (result.shape.size2d <= 256)
    {
        dim3 blockDim(A.height(), B.height());
        LOG_KERNEL_SIZE("A.shape: ", A.shape, " B.shape: ", B.shape,
                        (C ? " C: " + C->shape.str() : "no C, "), " result.shape: ", result.shape,
                        " gridDim: ", result.batch(), " blockDim: ", blockDim);
        mmTadd_kernel<T><<<result.batch(), blockDim>>>(result, A, B, C, pProcess);
    }
    else if (A.height() <= 1024)
    {
        constexpr uint32 BLOCK_SIZE_MM = 8;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(result.height(), BLOCK_SIZE_MM), iDivUp(result.width(), BLOCK_SIZE_MM),
                     result.batch());
        LOG_KERNEL_SIZE("A.shape: ", A.shape, " B.shape: ", B.shape,
                        (C ? " C: " + C->shape.str() : " no C, "), " result.shape: ", result.shape,
                        " gridDim: ", gridDim, " blockDim: ", blockDim);
        tiled_mmTadd_shmem<BLOCK_SIZE_MM, false><<<gridDim, blockDim>>>(result, A, B, C, pProcess);
    }
    else if (A.height() <= 2048)
    {
        constexpr uint32 BLOCK_SIZE_MM = 16;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(result.height(), BLOCK_SIZE_MM), iDivUp(result.width(), BLOCK_SIZE_MM),
                     result.batch());
        LOG_KERNEL_SIZE("A.shape: ", A.shape, " B.shape: ", B.shape,
                        (C ? " C: " + C->shape.str() : "no C, "), " result.shape: ", result.shape,
                        " gridDim: ", gridDim, " blockDim: ", blockDim);
        tiled_mmTadd_shmem<BLOCK_SIZE_MM, false><<<gridDim, blockDim>>>(result, A, B, C, pProcess);
    }
    else
    {
        constexpr uint32 BLOCK_SIZE_MM = 20;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(result.height(), BLOCK_SIZE_MM), iDivUp(result.width(), BLOCK_SIZE_MM),
                     result.batch());
        LOG_KERNEL_SIZE("A.shape: ", A.shape, " B.shape: ", B.shape,
                        (C ? " C: " + C->shape.str() : "no C, "), " result.shape: ", result.shape,
                        " gridDim: ", gridDim, " blockDim: ", blockDim);
        tiled_mmTadd_shmem<BLOCK_SIZE_MM, false><<<gridDim, blockDim>>>(result, A, B, C, pProcess);
    }
    cudaErrCheck(cudaGetLastError());
}

// clang-format off
template void mmTadd<FloatT, DividedBy<FloatT> >(Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, Optional<Matrix<FloatT> >, DividedBy<FloatT>);
template void mmTadd<FloatT, Identity<FloatT> >(Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, Optional<Matrix<FloatT> >, Identity<FloatT>);
template void mmTadd<FloatT, LeakyRelu<FloatT>::LeakyReluF>(Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, Optional<Matrix<FloatT> >, LeakyRelu<FloatT>::LeakyReluF);
template void mmTadd<FloatT, Relu<FloatT>::ReluF>(Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, Optional<Matrix<FloatT> >, Relu<FloatT>::ReluF);
template void mmTadd<FloatT, Sigmoid<FloatT>::SigmoidF>(Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, Optional<Matrix<FloatT> >, Sigmoid<FloatT>::SigmoidF);
template void mmTadd<FloatT, TanH<FloatT>::TanhF>(Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, Optional<Matrix<FloatT> >, TanH<FloatT>::TanhF);
// clang-format on
