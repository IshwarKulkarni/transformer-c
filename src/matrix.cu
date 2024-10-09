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
__global__ void tiled_mmadd_shmem(Tr *__restrict__ result, const Ta *__restrict__ A, uint32 aH,
                                  uint32 aW, const Tb *__restrict__ B, uint32 bW,
                                  const Tc *__restrict__ C)
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
__global__ void mmadd_kernel(Tr *__restrict__ result, const Ta *__restrict__ A, uint32 aH,
                             uint32 aW, const Tb *__restrict__ B, uint32 bW,
                             const Tc *__restrict__ C)
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
// Assuming that B is column vector, if h > 1024, multiple calls to this kernel will be made with
// assuming h = 1024, followed by h=256  ...
template <typename Ta, typename Tb = Ta, typename Tc = Ta, typename Tr = Ta, uint32 WIDTH>
__global__ void mat_vector_mul_kernel(Tr *result, const Ta *A, const Tb *B, const Tc *C,
                                      uint32 height, uint32 width, uint32 offset = 0,
                                      bool addToresult = false)
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;
    __shared__ Tr As[WIDTH];

    uint32 offset_x = x + offset;

    As[threadIdx.x] = (offset_x < width) ? Tr(A[offset_x + y * width] * B[offset_x]) : Tr(0);

    __syncthreads();

    Tr c = (C ? C[blockIdx.x] : Tr(0));
    Tr r = (addToresult ? result[blockIdx.x] : Tr(0));
#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < s) As[x] += As[x + s];
        __syncthreads();
    }
    volatile Tr *vAs = (volatile Tr *)As;
    if (x <= 32)
    {
        if (WIDTH >= 64) vAs[x] += vAs[x + 32];
        if (WIDTH >= 32) vAs[x] += vAs[x + 16];
        if (WIDTH >= 16) vAs[x] += vAs[x + 8];
        if (WIDTH >= 8) vAs[x] += vAs[x + 4];
        if (WIDTH >= 4) vAs[x] += vAs[x + 2];
        if (WIDTH >= 2) vAs[x] += vAs[x + 1];
    }
    __syncthreads();
    if (x == 0 and blockIdx.x < height)
    {
        result[blockIdx.x] = vAs[0] + c + r;
    }
}

template <typename Ta, typename Op, uint32 WIDTH>
__global__ void reduce_kernel(const Ta *A, Ta *result, Op op, uint32 height, uint32 width,
                              uint32 offset = 0, bool reducExisting = false, Ta identity = Ta(0))
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;
    __shared__ Ta As[WIDTH];

    uint32 offset_x = x + offset;

    As[threadIdx.x] = (offset_x < width) ? Ta(A[offset_x + y * width]) : identity;

    __syncthreads();

    Ta existing = (reducExisting ? result[y] : identity);

#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < s) As[x] = op(As[x], As[x + s]);
        __syncthreads();
    }
    volatile Ta *vAs = (volatile Ta *)As;
    if (x <= 32)
    {
        if (WIDTH >= 64) vAs[x] = op(vAs[x], vAs[x + 32]);
        if (WIDTH >= 32) vAs[x] = op(vAs[x], vAs[x + 16]);
        if (WIDTH >= 16) vAs[x] = op(vAs[x], vAs[x + 8]);
        if (WIDTH >= 8) vAs[x] = op(vAs[x], vAs[x + 4]);
        if (WIDTH >= 4) vAs[x] = op(vAs[x], vAs[x + 2]);
        if (WIDTH >= 2) vAs[x] = op(vAs[x], vAs[x + 1]);
    }
    __syncthreads();

    if (x == 0 and blockIdx.x < height)
    {
        result[y] = op(vAs[0], existing);
    }
}

template <typename T, typename Op>
void reduce(Matrix<T> &result, const Matrix<T> &A, const Op &op, T identity)
{
    if (A.width <= 32)
    {
        reduce_kernel<T, Op, 32><<<A.height, 32>>>(A.begin(), result.begin(), op, A.height, A.width,
                                                   0, false, identity);
    }
    else if (A.width <= 64)
    {
        reduce_kernel<T, Op, 64><<<A.height, 64>>>(A.begin(), result.begin(), op, A.height, A.width,
                                                   0, false, identity);
    }
    else if (A.width <= 128)
    {
        reduce_kernel<T, Op, 128><<<A.height, 128>>>(A.begin(), result.begin(), op, A.height,
                                                     A.width, 0, false, identity);
    }
    else if (A.width <= 256)
    {
        reduce_kernel<T, Op, 256><<<A.height, 256>>>(A.begin(), result.begin(), op, A.height,
                                                     A.width, 0, false, identity);
    }
    else if (A.width <= 512)
    {
        reduce_kernel<T, Op, 512><<<A.height, 512>>>(A.begin(), result.begin(), op, A.height,
                                                     A.width, 0, false, identity);
    }
    else
    {
        reduce_kernel<T, Op, 1024><<<A.height, 1024>>>(A.begin(), result.begin(), op, A.height,
                                                       A.width, 0, false, identity);
        constexpr uint32 WIDTH = 1024;
        for (uint offset = WIDTH; offset < A.width; offset += WIDTH)
        {
            reduce_kernel<T, Op, WIDTH><<<A.height, WIDTH>>>(
                A.begin(), result.begin(), op, A.height, A.width, offset, offset > 0, identity);
        }
    }
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mvadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C)
{
    check_mmadd_sizes(result, A, B, C);
    if (B.height <= 32)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 32><<<A.height, 32>>>(
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width);
    }
    else if (B.height <= 64)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 64><<<A.height, 64>>>(
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width);
    }
    else if (B.height <= 128)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 128><<<A.height, 128>>>(
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width);
    }
    else if (B.height <= 256)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 256><<<A.height, 256>>>(
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width);
    }
    else if (B.height <= 512)
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 512><<<A.height, 512>>>(
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width);
    }
    else
    {
        mat_vector_mul_kernel<Ta, Tb, Tc, Tr, 1024><<<A.height, 1024>>>(
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width);
        constexpr uint32 WIDTH = 1024;
        for (uint offset = 1024; offset < B.height; offset += WIDTH)
        {
            mat_vector_mul_kernel<Ta, Tb, Tc, Tr, WIDTH><<<A.height, WIDTH>>>(
                result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width,
                offset, offset > 0);
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
            <<<1, dim3(A.height, B.width)>>>(result.begin(), A.begin(), A.height, A.width,
                                             B.begin(), B.width, C ? C->begin() : nullptr);
    }
    else if (A.height <= 1536)
    {
        constexpr uint32 BLOCK_SIZE_MM = 16;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_mmadd_shmem<BLOCK_SIZE_MM, false, Tr, Ta, Tb, Tc>
            <<<gridDim, blockDim>>>(result.begin(), A.begin(), A.height, A.width, B.begin(),
                                    B.width, C ? C->begin() : nullptr);
    }
    else
    {
        constexpr uint32 BLOCK_SIZE_MM = 32;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_mmadd_shmem<BLOCK_SIZE_MM, false, Tr, Ta, Tb, Tc>
            <<<gridDim, blockDim>>>(result.begin(), A.begin(), A.height, A.width, B.begin(),
                                    B.width, C ? C->begin() : nullptr);
    }
    cudaErrCheck(cudaGetLastError());
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

template <typename T> void transpose(Matrix<T> &res, const Matrix<T> &A)
{
    if (A.height != res.width || A.width != res.height)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for transpose operation");
        throw std::runtime_error("Dimension mismatch");
    }

    if (A.width == 1)
    {
        fill(res, A.begin());
        return;
    }

    uint32 max_dim = std::max(A.width, A.height);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(iDivUp(max_dim, BLOCK_SIZE), iDivUp(max_dim, BLOCK_SIZE));
    // LOG("A: ", A.height, "x", A.width, "blockDim: ", blockDim.x, "x", blockDim.y, " gridDim: ",
    // gridDim.x, "x", gridDim.y);
    transpose_kernel<T><<<gridDim, blockDim>>>(res.begin(), A.begin(), A.height, A.width);
    cudaErrCheck(cudaGetLastError());
}

using FloatT = float64;

template void mvadd(Matrix<FloatT> &result, const Matrix<FloatT> &A, const Matrix<FloatT> &B,
                    const Matrix<FloatT> *C);

template void mmadd(Matrix<FloatT> &result, const Matrix<FloatT> &A, const Matrix<FloatT> &B,
                    const Matrix<FloatT> *C);

template void transpose(Matrix<FloatT> &res, const Matrix<FloatT> &A);

template void reduce(Matrix<FloatT> &, const Matrix<FloatT> &, const Plus<FloatT> &, FloatT);

template void reduce(Matrix<FloatT> &, const Matrix<FloatT> &, const Min<FloatT> &, FloatT);

template void reduce(Matrix<FloatT> &, const Matrix<FloatT> &, const Max<FloatT> &, FloatT);
