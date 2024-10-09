#include "../headers/matrix_ops.cuh"
#include "../headers/types"
#include <cuda_fp16.h>
#include <type_traits>

uint32 getMatrixId()
{
    static uint32 id = 0;
    return id++;
}

inline __device__ __host__ uint32 iDivUp(uint32 a, uint32 b) { return (a + b - 1) / b; }

constexpr uint32 BLOCK_SIZE_MM = 16;
constexpr uint32 BLOCK_SIZE = 32;

// AccumNative => internal accumulation is done in Tr, else 32bit(T) if sizeof(T) < 4 else 64bit(T)
template <uint32 TILE_SIZE_X = 16, bool AccumNative = false, typename Ta, typename Tb, typename Tc,
          typename Tr>
__global__ void tiled_madd_shmem(const Ta *__restrict__ A, uint32 aH, uint32 aW,
                                 const Tb *__restrict__ B, uint32 bW, const Tc *__restrict__ C,
                                 Tr *__restrict__ result)
{
    using SumType = typename std::conditional<AccumNative, Tr, typename AccumT<Tr>::type>::type;
    SumType sum{0};

    __shared__ float As[TILE_SIZE_X][TILE_SIZE_X + 1];
    __shared__ float Bs[TILE_SIZE_X][TILE_SIZE_X + 1];

    uint32 row = blockIdx.x * TILE_SIZE_X + threadIdx.x;
    uint32 col = blockIdx.y * TILE_SIZE_X + threadIdx.y;

    uint32 bH = aW;

#pragma unroll
    for (uint32 k = 0; k < iDivUp(aW, TILE_SIZE_X) * TILE_SIZE_X; k += TILE_SIZE_X)
    {
        uint32 aoffset = row * aW + k + threadIdx.y;
        uint32 boffset = (k + threadIdx.x) * bW + col;
        As[threadIdx.x][threadIdx.y] = (k + threadIdx.y < aW && row < aH) ? A[aoffset] : Ta(0);
        Bs[threadIdx.x][threadIdx.y] = (k + threadIdx.x < bH && col < bW) ? B[boffset] : Tb(0);
        __syncthreads();
#pragma unroll
        for (uint32 kk = 0; kk < TILE_SIZE_X; kk++)
        {
            sum += SumType(As[threadIdx.x][kk] * Bs[kk][threadIdx.y]);
        }
        __syncthreads();
    }
    if (row < aH && col < bW)
    {
        uint32 offset = row * bW + col;
        sum += SumType(C ? Tc(C[offset]) : Tc(0));
        result[offset] = sum;
    }
}

template <typename Ta, typename Tb, typename Tc, typename Tr>
__global__ void madd_kernel(const Ta *__restrict__ A, uint32 aH, uint32 aW,
                            const Tb *__restrict__ B, uint32 bW, const Tc *__restrict__ C,
                            Tr *__restrict__ result)
{
    uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < aH && j < bW)
    {
        Tr sum = 0;
#pragma unroll
        for (uint32 k = 0; k < aW; k++)
        {
            sum += A[i * aW + k] * B[k * bW + j];
        }
        result[i * bW + j] = sum + (C ? C[i * bW + j] : Tr(0));
    }
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
void madd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C)
{
    check_madd_sizes(result, A, B, C);

    if (result.numels() <= 1024 and false) // small matrices
    {
        // LOG("Using madd_kernel: ", result.numels());
        madd_kernel<Tr, Ta, Tb, Tc>
            <<<1, dim3(A.height, B.width)>>>(A.begin(), A.height, A.width, B.begin(), B.width,
                                             C ? C->begin() : nullptr, result.begin());
    }
    else if (result.width < BLOCK_SIZE_MM * 3 / 4) // skinny result matrix
    {
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_madd_shmem<BLOCK_SIZE_MM, true, Tr, Ta, Tb, Tc>
            <<<gridDim, blockDim>>>(A.begin(), A.height, A.width, B.begin(), B.width,
                                    C ? C->begin() : nullptr, result.begin());
    }
    else
    {
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_madd_shmem<BLOCK_SIZE_MM, false, Tr, Ta, Tb, Tc>
            <<<gridDim, blockDim>>>(A.begin(), A.height, A.width, B.begin(), B.width,
                                    C ? C->begin() : nullptr, result.begin());
    }
    cudaErrCheck(cudaDeviceSynchronize());
}

template <typename T> Matrix<T> madd(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C)
{
    Matrix<T> result(A.height, B.width);
    fill(result, 0.0f);
    madd<T, T, T, T>(result, A, B, C);
    return result;
}

template <typename T>
__global__ void transpose_kernel(T *__restrict__ result, const T *__restrict__ A, uint32 height,
                                 uint32 width)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    uint32 x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32 y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = A[y * width + x];

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (y < width && x < height) result[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

template <typename T> Matrix<T> transpose(const Matrix<T> &A)
{
    uint32 max_dim = std::max(A.width, A.height);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(iDivUp(max_dim, BLOCK_SIZE), iDivUp(max_dim, BLOCK_SIZE));
    // LOG("A: ", A.height, "x", A.width, "blockDim: ", blockDim.x, "x", blockDim.y, " gridDim: ",
    // gridDim.x, "x", gridDim.y);
    Matrix<T> res(A.width, A.height);
    transpose_kernel<T><<<gridDim, blockDim>>>(res.begin(), A.begin(), A.height, A.width);
    cudaErrCheck(cudaDeviceSynchronize());
    return res;
}

template void madd<float64, float64, float64>(Matrix<float64> &result, const Matrix<float64> &A,
                                              const Matrix<float64> &B, const Matrix<float64> *C);
template Matrix<float64> madd(const Matrix<float64> &A, const Matrix<float64> &B,
                              const Matrix<float64> *C);

template void madd<float32, float32, float32>(Matrix<float32> &result, const Matrix<float32> &A,
                                              const Matrix<float32> &B, const Matrix<float32> *C);
template Matrix<float32> madd(const Matrix<float32> &A, const Matrix<float32> &B,
                              const Matrix<float32> *C);

template void madd<float16, float16, float16, float16>(Matrix<float16> &result,
                                                       const Matrix<float16> &A,
                                                       const Matrix<float16> &B,
                                                       const Matrix<float16> *C);
template Matrix<half> madd(const Matrix<half> &A, const Matrix<half> &B, const Matrix<half> *C);

template Matrix<float32> transpose(const Matrix<float32> &A);