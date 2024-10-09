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

constexpr uint32 BLOCK_SIZE = 32;

// AccumNative => internal accumulation is done in Tr, else 32bit(T) if sizeof(T) < 4 else 64bit(T)
template <uint32 TILE_SIZE_X = 16, bool AccumNative = false, typename Ta, typename Tb, typename Tc,
          typename Tr>
__global__ void tiled_mmadd_shmem(const Ta *__restrict__ A, uint32 aH, uint32 aW,
                                 const Tb *__restrict__ B, uint32 bW, const Tc *__restrict__ C,
                                 Tr *__restrict__ result)
{
    using SumType = typename std::conditional<AccumNative, Tr, typename AccumT<Tr>::type>::type;
    SumType sum{0};

    __shared__ Ta As[TILE_SIZE_X][TILE_SIZE_X + 1];
    __shared__ Tb Bs[TILE_SIZE_X][TILE_SIZE_X + 1];

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
__global__ void mmadd_kernel(const Ta *__restrict__ A, uint32 aH, uint32 aW,
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

// A (h,w)  * B (w, 1)  + <C(h,1> -> result (h, 1)
// Assuming that B is column vector, if h > 1024, multiple calls to this kernel will be made with offset = 0, 1024, 2048, ...
template <typename Ta, typename Tb = Ta, typename Tc=Ta, typename Tr = Ta, uint32 WIDTH>
__global__ void mat_vector_mul_kernel(const Ta *A, const Tb *B, const Tc* C, Tr *result, uint32 height, uint32 width, uint32 offset = 0, bool addToresult=false)
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;
    __shared__ Tr As[WIDTH];

    uint32 aoffset_x = x + offset;

    As[threadIdx.x] = (aoffset_x < width) ? Tr(A[aoffset_x + y * width] * B[x + offset]) : Tr(0);
    
    __syncthreads();

    Tr c = (C ? C[blockIdx.x] : Tr(0));
    Tr r = (addToresult ? result[blockIdx.x] : Tr(0));
    #pragma unroll
    for (uint32 s = blockDim.x/2; s > 32; s >>= 1)
    {
        if (x < s) 
            As[x] += As[x + s];
        __syncthreads();
    }
    volatile Tr *vAs = (volatile Tr *)As;
    if (x <= 32)
    {
        if (WIDTH >= 64) vAs[x] += vAs[x + 32];
        if (WIDTH >= 32) vAs[x] += vAs[x + 16];
        if (WIDTH >= 16) vAs[x] += vAs[x + 8];
        if (WIDTH >= 8)  vAs[x] += vAs[x + 4];
        if (WIDTH >= 4)  vAs[x] += vAs[x + 2];
        if (WIDTH >= 2)  vAs[x] += vAs[x + 1];
    }
    __syncthreads();
    if (x == 0 and blockIdx.x < height) 
    {
        result[blockIdx.x] = vAs[0] + c + r;
    }
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mvadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C)
{
    check_mmadd_sizes(result, A, B, C);
    if(B.height <= 2)
    {
           mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 2>
                <<<A.height, 2>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 4)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 4>
                <<<A.height, 4>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 8)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 8>
                <<<A.height, 8>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 16)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 16>
                <<<A.height, 16>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 32)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 32>
                <<<A.height, 32>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 64)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 64>
                <<<A.height, 64>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 128)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 128>
                <<<A.height, 128>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 256)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 256>
                <<<A.height, 256>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 512)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 512>
                <<<A.height, 512>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else if(B.height <= 1024)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 1024>
                <<<A.height, 1024>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width);
    }
    else
    {
        for(uint offset = 0; offset < B.height; offset += 1024)
        {
            mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 1024>
                <<<A.height, 1024>>>(A.begin(), B.begin(), C ? C->begin() : nullptr, result.begin(), A.height, A.width, offset, offset > 0);
        }
    }
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mmadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C)
{
    check_mmadd_sizes(result, A, B, C);

    if (B.width == 1)
    {
        mvadd<Tr, Ta, Tb, Tc>(result, A, B, C);
    }
    else if (result.numels() <= 1024 and false) // small matrices
    {
        // LOG("Using mmadd_kernel: ", result.numels());
        mmadd_kernel<Tr, Ta, Tb, Tc>
            <<<1, dim3(A.height, B.width)>>>(A.begin(), A.height, A.width, B.begin(), B.width,
                                             C ? C->begin() : nullptr, result.begin());
    }
    else if (A.height <= 1536)
    {
        constexpr uint32 BLOCK_SIZE_MM = 16;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_mmadd_shmem<BLOCK_SIZE_MM, false, Tr, Ta, Tb, Tc>
            <<<gridDim, blockDim>>>(A.begin(), A.height, A.width, B.begin(), B.width,
                                    C ? C->begin() : nullptr, result.begin());
    }
    else 
    {
        constexpr uint32 BLOCK_SIZE_MM = 32;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_mmadd_shmem<BLOCK_SIZE_MM, false, Tr, Ta, Tb, Tc>
            <<<gridDim, blockDim>>>(A.begin(), A.height, A.width, B.begin(), B.width,
                                    C ? C->begin() : nullptr, result.begin());
    }
    cudaErrCheck(cudaGetLastError());
}

template <typename T> Matrix<T> mmadd(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C)
{
    Matrix<T> result(A.height, B.width);
    fill(result, 0.0f);
    mmadd<T, T, T, T>(result, A, B, C);
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
    cudaErrCheck(cudaGetLastError());
    return res;
}

template void mmadd<float32, float32, float32>(Matrix<float32> &result, const Matrix<float32> &A,
                                              const Matrix<float32> &B, const Matrix<float32> *C);
template Matrix<float32> mmadd(const Matrix<float32> &A, const Matrix<float32> &B,
                              const Matrix<float32> *C);

template Matrix<float32> transpose(const Matrix<float32> &A);

/*
template void mmadd<float64, float64, float64>(Matrix<float64> &result, const Matrix<float64> &A,
                                              const Matrix<float64> &B, const Matrix<float64> *C);
template Matrix<float64> mmadd(const Matrix<float64> &A, const Matrix<float64> &B,
                              const Matrix<float64> *C);

template void mmadd<float16, float16, float16, float16>(Matrix<float16> &result,
                                                       const Matrix<float16> &A,
                                                       const Matrix<float16> &B,
                                                       const Matrix<float16> *C);
template Matrix<half> mmadd(const Matrix<half> &A, const Matrix<half> &B, const Matrix<half> *C);
*/