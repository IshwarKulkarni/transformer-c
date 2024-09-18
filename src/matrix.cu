
#include "../headers/matrix_ops.cuh"

uint32_t getMatrixId() {
    static uint32_t id = 0;
    return id++;
}

constexpr uint32_t BLOCK_SIZE_X = 7;  // should be same as BLOCK_SIZE_Y

template<typename T> __global__
void tiled_madd_shmem(T *A, uint32_t arows, uint32_t acols, T *B, uint32_t bcols, T *C, T *result)
{
    uint32_t row = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    uint32_t col = blockIdx.y * BLOCK_SIZE_X + threadIdx.y;
    if (row < arows && col < bcols)
    {
        T sum = 0;
        __shared__ T As[BLOCK_SIZE_X][BLOCK_SIZE_X];
        __shared__ T Bs[BLOCK_SIZE_X][BLOCK_SIZE_X];

        #pragma unroll
        for (uint32_t k = 0; k < acols; k += BLOCK_SIZE_X)
        {
            uint32_t aoffset = row * acols + k + threadIdx.y;
            uint32_t boffset = (k + threadIdx.x) * bcols + col;
            As[threadIdx.x][threadIdx.y] = k + threadIdx.y < acols ? A[aoffset] : 0.f;
            Bs[threadIdx.x][threadIdx.y] = k + threadIdx.x < arows ? B[boffset] : 0.f;
            __syncthreads();
            #pragma unroll
            for (uint32_t kk = 0; kk < BLOCK_SIZE_X; kk++)
            {
                sum += As[threadIdx.x][kk] * Bs[kk][threadIdx.y];
            }
            __syncthreads();
        }
        result[row * bcols + col] = sum + (C ? C[row * bcols + col] : 0);  // Transpose result
    }
}

template<typename T>
__global__ // A(ak, al) * B(al, bm) + C(ak, bm) = result(ak, bm)
void madd_kernel(T *A, uint32_t ak, uint32_t al, T *B, uint32_t bm, T *C, T *result)
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
        result[i * bm + j] = sum + (C ? C[i * bm + j] : 0);
    }
}

template<typename T>
void madd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C)
{
    if (A.cols != B.rows)
    {
        std::cerr << "Matrix dimensions do not match for A " << A.get_name() << " and B " << B.get_name() << std::endl;
        throw std::runtime_error("Dimension mismatch");
    }
    if(result.rows != A.rows || result.cols != B.cols)
    {
        std::cerr << "Matrix dimensions do not match for result " << result.get_name() << " and A " << A.get_name() << " and B " << B.get_name() << std::endl;
        throw std::runtime_error("Dimension mismatch");
    }

    uint32_t res_numels = result.rows * result.cols;
    if (res_numels < BLOCK_SIZE_X * BLOCK_SIZE_X)
    {
        madd_kernel<<<1, dim3(A.rows, B.cols)>>>(A.data.get(), A.rows, A.cols, B.data.get(), B.cols, C ? C->data.get() : nullptr, result.data.get());
    }
    else
    {
        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_X);
        dim3 gridDim( (A.rows + BLOCK_SIZE_X-1) / BLOCK_SIZE_X, (B.cols + BLOCK_SIZE_X-1) / BLOCK_SIZE_X);
        //std::cout << "Matrix dimensions " << result.rows << "x" << result.cols << " performing madd with 2D grid " << gridDim.x << ", " << gridDim.y << std::endl;
        tiled_madd_shmem<<<gridDim, blockDim>>>(A.data.get(), A.rows, A.cols, B.data.get(), B.cols, C ? C->data.get() : nullptr, result.data.get());
    }

    madd_kernel<<<1, dim3(A.rows, B.cols)>>>(A.data.get(), A.rows, A.cols, B.data.get(), B.cols, C ? C->data.get() : nullptr, result.data.get());
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
    Matrix<T> result(A.rows, B.cols);
    fill(result, 0.0f);
    madd(result, A, B, C);
    return result;
}


template void madd(Matrix<float> &result, const Matrix<float> &A, const Matrix<float> &B, const Matrix<float> *C);
template Matrix<float> madd(const Matrix<float> &A, const Matrix<float> &B, const Matrix<float> *C);
