#include "../headers/matrix_ops.cuh"
#include <cuda_fp16.h>
#include <type_traits>
#include "../headers/types"

uint32_t getMatrixId() {
    static uint32_t id = 0;
    return id++;
}

inline __device__ __host__ uint32_t iDivUp(uint32_t a, uint32_t b){return (a + b - 1) / b;}

constexpr uint32_t BLOCK_SIZE_X = 8;

 // AccumNative => internal accumulation is done in Tr, else 32bit(T) if sizeof(T) < 4 else 64bit(T)
template<bool AccumNative=false, typename Ta, typename Tb, typename Tc, typename Tr> __global__ 
void tiled_madd_shmem(const Ta *A, uint32_t aH, uint32_t aW, const Tb *B, const Tc *C, Tr *result)
{
    uint32_t row = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    uint32_t col = blockIdx.y * BLOCK_SIZE_X + threadIdx.y;

    using SumType = typename std::conditional<AccumNative, Tr, typename AccumT<Tr>::type>::type;
    SumType sum = 0.0;

    __shared__ float As[BLOCK_SIZE_X][BLOCK_SIZE_X];
    __shared__ float Bs[BLOCK_SIZE_X][BLOCK_SIZE_X];

    #pragma unroll
    for (uint32_t k = 0; k < iDivUp(aW, BLOCK_SIZE_X) * BLOCK_SIZE_X; k += BLOCK_SIZE_X)
    {
        uint32_t aoffset = row * aW + k + threadIdx.y;
        uint32_t boffset = (k + threadIdx.x) * aH + col;
        As[threadIdx.x][threadIdx.y] = k + threadIdx.y < aW ? A[aoffset] : Ta(0);
        Bs[threadIdx.x][threadIdx.y] = k + threadIdx.x < aW ? B[boffset] : Tb(0);
        __syncthreads();
        #pragma unroll
        for (uint32_t kk = 0; kk < BLOCK_SIZE_X; kk++)
        {
            sum += SumType(As[threadIdx.x][kk] * Bs[kk][threadIdx.y]);
        }
        __syncthreads();
    }
    if (row < aH && col < aH)
    {
        sum += SumType(C ? Tc(C[row * aH + col]) : Tc(0));
        result[row * aH + col] = sum;
    }
}

template<typename T>
__global__ // A(ak, al) * B(al, bm) + C(ak, bm) = result(ak, bm)
void madd_kernel(const T *A, uint32_t ak, uint32_t al, const T *B, uint32_t bm, const T *C, T *result)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ak && j < bm)
    {
        T sum = 0;
        #pragma unroll
        for (uint32_t k = 0; k < al; k++)
        {
            sum += A[i * al + k] * B[k * bm + j];
        }
        result[i * bm + j] = sum + (C ? C[i * bm + j] : T(0));
    }
}

template<typename Tr, typename Ta, typename Tb, typename Tc>
void madd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C)
{
    if (A.width != B.height || A.height != result.height || B.width != result.width)
    {
        std::cerr << "Matrix dimensions do not match for A " << A.get_name() << " and B " << B.get_name() << std::endl;
        throw std::runtime_error("Dimension mismatch");
    }
    if(result.height != A.height || result.width != B.width)
    {
        std::cerr << "Matrix dimensions do not match for result " << result.get_name() 
                  << " and A " << A.get_name() << " and B " << B.get_name() << std::endl;
        throw std::runtime_error("Dimension mismatch");
    }

    uint32_t res_numels = result.height * result.width;
    if (res_numels < 1024)
    {
        madd_kernel<<<1, dim3(A.height, B.width)>>>(
                A.begin(), A.height, A.width, 
                B.begin(), B.width, 
                C ? C->begin() : nullptr, result.begin()
            );
    }
    else
    {
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_X);
        dim3 gridDim( iDivUp(A.height , BLOCK_SIZE_X), iDivUp(B.width, BLOCK_SIZE_X));
        tiled_madd_shmem<true, Tr, Ta, Tb, Tc><<<gridDim, blockDim>>>(
            A.begin(), A.height, A.width, 
            B.begin(), 
            C ? C->begin() : nullptr, result.begin());
    }
    cudaErrCheck(cudaDeviceSynchronize());
}

template<typename T>
__global__ 
void transpose_kernel(T *A, uint32_t rows, uint32_t cols, T *result)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols)
    {
        result[j * rows + i] = A[i * cols + j];
    }
}

template<typename T>
Matrix<T> madd(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C)
{
    Matrix<T> result(A.height, B.width);
    fill(result, 0.0f);
    madd<T, T, T,T>(result, A, B, C);
    return result;
}

template void madd<float64, float64, float64>(Matrix<float64> &result, const Matrix<float64> &A, const Matrix<float64> &B, const Matrix<float64> *C);
template Matrix<float64> madd(const Matrix<float64> &A, const Matrix<float64> &B, const Matrix<float64> *C);

template void madd<float32, float32, float32>(Matrix<float32> &result, const Matrix<float32> &A, const Matrix<float32> &B, const Matrix<float32> *C);
template Matrix<float32> madd(const Matrix<float32> &A, const Matrix<float32> &B, const Matrix<float32> *C);

template void madd<float16, float16, float16, float16>(Matrix<float16> &result, const Matrix<float16> &A, const Matrix<float16> &B, const Matrix<float16> *C);
template Matrix<half> madd(const Matrix<half> &A, const Matrix<half> &B, const Matrix<half> *C);
