#include <cuda_fp16.h>
#include <type_traits>
#include "../headers/matrix_ops.cuh"
#include "../headers/types"

template <typename Tc>
__device__ Tc getC(uint32 row, uint32 col, const Tc *C, uint32 cH, uint32 cW)
{
    if (C == nullptr) return Tc(0);
    uint32 y = cH > 1 ? row : 0;
    uint32 x = cW > 1 ? col : 0;
    return C[y * cW + x];
}

// computes the A * B^T + C
// AccumNative => internal accumulation is done in T, else 32bit(T) if sizeof(T)
// < 4 else 64bit(T).
template <uint32 TILE_SIZE_X = 16, bool AccumNative = false, typename T, typename PProcess>
__global__ void tiled_mmTadd_shmem(T *__restrict__ result, const T *__restrict__ A, uint32 aH,
                                   uint32 aW, const T *__restrict__ B, uint32 bH,
                                   const T *__restrict__ C, uint32 cH, uint32 cW, PProcess pprocess)
{
    using SumType = typename std::conditional<AccumNative, T, typename AccumT<T>::type>::type;
    SumType sum{0};

    __shared__ T As[TILE_SIZE_X][TILE_SIZE_X + 1];
    __shared__ T Bs[TILE_SIZE_X][TILE_SIZE_X + 1];

    uint32 y = blockIdx.x * TILE_SIZE_X + threadIdx.x;
    uint32 x = blockIdx.y * TILE_SIZE_X + threadIdx.y;

    uint32 bW = aW;

#pragma unroll
    for (uint32 k = 0; k < iDivUp(aW, TILE_SIZE_X) * TILE_SIZE_X; k += TILE_SIZE_X)
    {
        uint32 aoffset = y * aW + k + threadIdx.y;
        uint32 boffset = (k + threadIdx.x) + bW * x;  // difference between mmadd and mmTadd;
        As[threadIdx.x][threadIdx.y] = (k + threadIdx.y < aW && y < aH) ? A[aoffset] : T(0);
        Bs[threadIdx.x][threadIdx.y] = (k + threadIdx.x < bW && x < bH) ? B[boffset] : T(0);
        __syncthreads();
#pragma unroll
        for (uint32 kk = 0; kk < TILE_SIZE_X; kk++)
        {
            sum += SumType(As[threadIdx.x][kk] * Bs[kk][threadIdx.y]);
        }
        __syncthreads();
    }

    if (y < aH && x < bH)
    {
        uint32 offset = y * bH + x;
        sum += getC(y, x, C, cH, cW);
        result[offset] = pprocess(sum);
    }
}

template <typename T, typename PProcess>
__global__ void mmTadd_kernel(T *__restrict__ result, const T *__restrict__ A, uint32 aH, uint32 aW,
                              const T *__restrict__ B, uint32 bH, const T *__restrict__ C,
                              uint32 cH, uint32 cW, PProcess pprocess)
{
    uint32 y = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 x = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < aH && x < bH)
    {
        T sum = 0;
#pragma unroll
        for (uint32 k = 0; k < aW; k++)
        {
            sum += A[y * aW + k] * B[k + aW * x];
        }

        result[y * bH + x] = pprocess(sum + getC(y, x, C, cH, cW));
    }
}

template <typename T, typename PProcess>
void mmTadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
            PProcess pProcess)
{
    check_mmTadd_sizes(result, A, B, C);

    if (result.numels() <= 1024 and false)  // small matrices
    {
        // LOG("Using mmadd_kernel: ", result.numels());
        mmTadd_kernel<T><<<1, dim3(A.height, B.height)>>>(
            result.begin(), A.begin(), A.height, A.width, B.begin(), B.height,
            C ? C->begin() : nullptr, C ? C->height : 0, C ? C->width : 0, pProcess);
    }
    else if (A.height <= 1536 and false)
    {
        constexpr uint32 BLOCK_SIZE_MM = 16;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.height, BLOCK_SIZE_MM));
        tiled_mmTadd_shmem<BLOCK_SIZE_MM, false><<<gridDim, blockDim>>>(
            result.begin(), A.begin(), A.height, A.width, B.begin(), B.height,
            C ? C->begin() : nullptr, C ? C->height : 0, C ? C->width : 0, pProcess);
    }
    else
    {
        constexpr uint32 BLOCK_SIZE_MM = 32;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.height, BLOCK_SIZE_MM));
        tiled_mmTadd_shmem<BLOCK_SIZE_MM, false><<<gridDim, blockDim>>>(
            result.begin(), A.begin(), A.height, A.width, B.begin(), B.height,
            C ? C->begin() : nullptr, C ? C->height : 0, C ? C->width : 0, pProcess);
    }
    cudaErrCheck(cudaGetLastError());
}

template void mmTadd<FloatT, Identity<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                               Matrix<FloatT> const &, Matrix<FloatT> const *,
                                               Identity<FloatT>);

template void mmTadd<FloatT, Sigmoid<FloatT>::SigmoidF>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                        Matrix<FloatT> const &,
                                                        Matrix<FloatT> const *,
                                                        Sigmoid<FloatT>::SigmoidF);

template void mmTadd<FloatT, DividebBy<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                Matrix<FloatT> const &, Matrix<FloatT> const *,
                                                DividebBy<FloatT>);

template void mmTadd<FloatT, Composition<FloatT, Neg<FloatT>, Identity<FloatT>>>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Matrix<FloatT> const &, Matrix<FloatT> const *,
    Composition<FloatT, Neg<FloatT>, Identity<FloatT>>);