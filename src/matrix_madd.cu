#include "../headers/matrix_ops.cuh"
#include "../headers/types"
#include <cuda_fp16.h>
#include <type_traits>

uint32 getMatrixId()
{
    static uint32 id = 0;
    return id++;
}

// AccumNative => internal accumulation is done in T, else 32bit(T) if sizeof(T)
// < 4 else 64bit(T)
template <uint32 TILE_SIZE_X = 16, bool AccumNative = false, typename T, typename PProcess>
__global__ void tiled_mmadd_shmem(T *__restrict__ result, const T *__restrict__ A, uint32 aH,
                                  uint32 aW, const T *__restrict__ B, uint32 bW,
                                  const T *__restrict__ C, PProcess pprocess)
{
    using SumType = typename std::conditional<AccumNative, T, typename AccumT<T>::type>::type;
    SumType sum{0};

    __shared__ T As[TILE_SIZE_X][TILE_SIZE_X + 1];
    __shared__ T Bs[TILE_SIZE_X][TILE_SIZE_X + 1];

    uint32 row = blockIdx.x * TILE_SIZE_X + threadIdx.x;
    uint32 col = blockIdx.y * TILE_SIZE_X + threadIdx.y;

    uint32 bH = aW;

#pragma unroll
    for (uint32 k = 0; k < iDivUp(aW, TILE_SIZE_X) * TILE_SIZE_X; k += TILE_SIZE_X)
    {
        uint32 aoffset = row * aW + k + threadIdx.y;
        uint32 boffset = (k + threadIdx.x) * bW + col;
        As[threadIdx.x][threadIdx.y] = (k + threadIdx.y < aW && row < aH) ? A[aoffset] : T(0);
        Bs[threadIdx.x][threadIdx.y] = (k + threadIdx.x < bH && col < bW) ? B[boffset] : T(0);
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
        sum += SumType(C ? T(C[offset]) : T(0));
        result[offset] = pprocess(sum);
    }
}

template <typename T, typename PProcess>
__global__ void mmadd_kernel(T *__restrict__ result, const T *__restrict__ A, uint32 aH, uint32 aW,
                             const T *__restrict__ B, uint32 bW, const T *__restrict__ C,
                             PProcess pprocess)
{
    uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < aH && j < bW)
    {
        T sum = 0;
#pragma unroll
        for (uint32 k = 0; k < aW; k++)
        {
            sum += A[i * aW + k] * B[k * bW + j];
        }
        result[i * bW + j] = pprocess(sum + (C ? C[i * bW + j] : T(0)));
    }
}

// A (h,w)  * B (w, 1)  + <C(h,1> -> result (h, 1)
// Assuming that B is column vector, if h > 1024, multiple calls to this kernel
// will be made with assuming h = 1024, followed by h=256  ...
template <typename T, uint32 WIDTH, typename PostProcess>
__global__ void mat_vector_mul_kernel(T *result, const T *A, const T *B, const T *C, uint32 height,
                                      uint32 width, uint32 offset = 0, bool addToresult = false,
                                      PostProcess pProcess = Identity<T>())
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;
    __shared__ T As[WIDTH];

    uint32 offset_x = x + offset;

    As[threadIdx.x] = (offset_x < width) ? T(A[offset_x + y * width] * B[offset_x]) : T(0);

    __syncthreads();

    T c = (C ? C[blockIdx.x] : T(0));
    T r = (addToresult ? result[blockIdx.x] : T(0));
#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < s) As[x] += As[x + s];
        __syncthreads();
    }
    volatile T *vAs = (volatile T *)As;
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
        result[blockIdx.x] = pProcess(vAs[0] + c + r);
    }
}

template <typename T, typename PostProcess>
__global__ void outer_product(T *result, const T *A, const T *B, const T *C, uint32 rheight,
                              uint32 rwidth, bool addToresult = false,
                              PostProcess pProcess = Identity<T>())
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;

    if (x < rwidth and y < rheight)
    {
        T c = (C ? C[y] : T(0));
        T r = (addToresult ? result[y * rwidth + x] : T(0));
        result[y * rwidth + x] = pProcess(T(A[y] * B[x]) + c + r);
    }
}

template <typename T, typename PostProcess>
void mvadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
           PostProcess pProcess)
{
    check_mmadd_sizes(result, A, B, C);

    if (B.height == 1 and
        B.width <= 1024) // outer product (small enough to fit in one thread block)
    {
        outer_product<<<A.height, B.width>>>(result.begin(), A.begin(), B.begin(),
                                             (C ? C->begin() : nullptr), A.height, B.width, false,
                                             pProcess);
        return;
    }
    if (B.height <= 32)
    {
        mat_vector_mul_kernel<T, 32><<<A.height, 32>>>(result.begin(), A.begin(), B.begin(),
                                                       C ? C->begin() : nullptr, A.height, A.width,
                                                       0, false, pProcess);
    }
    else if (B.height <= 64)
    {
        mat_vector_mul_kernel<T, 64><<<A.height, 64>>>(result.begin(), A.begin(), B.begin(),
                                                       C ? C->begin() : nullptr, A.height, A.width,
                                                       0, false, pProcess);
    }
    else if (B.height <= 128)
    {
        mat_vector_mul_kernel<T, 128><<<A.height, 128>>>(result.begin(), A.begin(), B.begin(),
                                                         C ? C->begin() : nullptr, A.height,
                                                         A.width, 0, false, pProcess);
    }
    else if (B.height <= 256)
    {
        mat_vector_mul_kernel<T, 256><<<A.height, 256>>>(result.begin(), A.begin(), B.begin(),
                                                         C ? C->begin() : nullptr, A.height,
                                                         A.width, 0, false, pProcess);
    }
    else if (B.height <= 512)
    {
        mat_vector_mul_kernel<T, 512><<<A.height, 512>>>(result.begin(), A.begin(), B.begin(),
                                                         C ? C->begin() : nullptr, A.height,
                                                         A.width, 0, false, pProcess);
    }
    else if (B.height <= 1024)
    {
        LOG("Using mat_vector_mul_kernel, A: ", A.height, "x", A.width, " B: ", B.height, "x",
            B.width);
        mat_vector_mul_kernel<T, 1024><<<A.height, 1024>>>(result.begin(), A.begin(), B.begin(),
                                                           C ? C->begin() : nullptr, A.height,
                                                           A.width, 0, false, pProcess);
    }
    else if (B.height > 1024)
    {
        constexpr int32 WIDTH = 1024;
        int32 offset = 0;
        for (; offset < B.height - WIDTH; offset += WIDTH)
        {
            mat_vector_mul_kernel<T, WIDTH, Identity<T>>
                <<<A.height, WIDTH>>>( // only multi plication and addition, do not add C
                    result.begin(), A.begin(), B.begin(), nullptr, A.height, A.width, offset,
                    offset > 0);
        }

        mat_vector_mul_kernel<T, WIDTH><<<A.height, WIDTH>>>( // add C and apply post process
            result.begin(), A.begin(), B.begin(), C ? C->begin() : nullptr, A.height, A.width,
            offset, offset > 0, pProcess);
    }
}

template <typename T, typename PProcess>
void mmadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
           PProcess pProcess)
{
    check_mmadd_sizes(result, A, B, C);

    if (B.width == 1)
    {
        mvadd<T>(result, A, B, C, pProcess);
    }
    else if (result.numels() <= 1024) // small matrices
    {
        // LOG("Using mmadd_kernel: ", result.numels());
        mmadd_kernel<T><<<1, dim3(A.height, B.width)>>>(result.begin(), A.begin(), A.height,
                                                        A.width, B.begin(), B.width,
                                                        C ? C->begin() : nullptr, pProcess);
    }
    else if (A.height <= 1536)
    {
        constexpr uint32 BLOCK_SIZE_MM = 16;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_mmadd_shmem<BLOCK_SIZE_MM, false>
            <<<gridDim, blockDim>>>(result.begin(), A.begin(), A.height, A.width, B.begin(),
                                    B.width, C ? C->begin() : nullptr, pProcess);
    }
    else
    {
        constexpr uint32 BLOCK_SIZE_MM = 32;
        dim3 blockDim(BLOCK_SIZE_MM, BLOCK_SIZE_MM);
        dim3 gridDim(iDivUp(A.height, BLOCK_SIZE_MM), iDivUp(B.width, BLOCK_SIZE_MM));
        tiled_mmadd_shmem<BLOCK_SIZE_MM, false>
            <<<gridDim, blockDim>>>(result.begin(), A.begin(), A.height, A.width, B.begin(),
                                    B.width, C ? C->begin() : nullptr, pProcess);
    }
    cudaErrCheck(cudaGetLastError());
}

template <typename T, uint32 BLOCK_SIZE>
__global__ void transpose_kernel(T *__restrict__ result, const T *__restrict__ A, uint32 height,
                                 uint32 width)
{
    __shared__ float32 tile[BLOCK_SIZE][BLOCK_SIZE + 1];
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

    constexpr uint32 BLOCK_SIZE = 32;
    uint32 max_dim = std::max(A.width, A.height);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(iDivUp(max_dim, BLOCK_SIZE), iDivUp(max_dim, BLOCK_SIZE));
    transpose_kernel<T, BLOCK_SIZE>
        <<<gridDim, blockDim>>>(res.begin(), A.begin(), A.height, A.width);
    cudaErrCheck(cudaGetLastError());
}

using FloatT = float32;
template void mmadd<FloatT, Sigmoid<FloatT>::SigmoidF>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                       Matrix<FloatT> const &,
                                                       Matrix<FloatT> const *,
                                                       Sigmoid<FloatT>::SigmoidF);
template void mmadd<FloatT, Identity<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                              Matrix<FloatT> const &, Matrix<FloatT> const *,
                                              Identity<FloatT>);
template void transpose(Matrix<FloatT> &res, const Matrix<FloatT> &A);
template void mmadd<FloatT, Loge<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                          Matrix<FloatT> const &, Matrix<FloatT> const *,
                                          Loge<FloatT>);
