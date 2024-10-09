#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <type_traits>
#include "matrix.cuh"


template<typename T>
void fill(Matrix<T> &A, float value)
{
    std::fill(A.begin(), A.end(), value);
}

template<typename T>
inline void fill(Matrix<T>& A, const float* values)
{
    std::copy(values, values + A.nuemls(), A.begin());
}

template<typename T>
bool same(const Matrix<T>&A, const Matrix<T> &B, float eps=1e-5)
{
    return std::equal(A.begin(), A.end(), B.begin(), [eps](T a, T b) { return std::abs(a - b) < eps; });
}

template<typename T>
void madd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C);

template<typename T>
Matrix<T> madd(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C);

/**
template<typename T1, typename T2, typename result_type, typename F>
__global__ void apply_kernel(const T1 *A, const T2* B, result_type *result, uint32_t numels, F f)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numels)
    {
        result[i] = f(A[i], B[i]);
    }
}

template<typename T1, typename T2, typename F>
void apply(const Matrix<T1> &A, const Matrix<T2>& B, F f)
{
    using result_type = typename std::result_of<F(T1, T2)>::type;
    Matrix<result_type> result(A.rows, A.cols);

    if(B.rows != A.rows || B.cols != A.cols)
    {
        std::cerr << "Matrix dimensions do not match for result " << B.get_name() << " and A " << A.get_name() << std::endl;
        throw std::runtime_error("Dimension mismatch");
    }

    uint32_t numels = result.rows * result.cols;
    dim3 blockDim(256);
    dim3 gridDim((numels + blockDim.x - 1) / blockDim.x);
    apply_kernel<<<gridDim, blockDim>>>(A.begin(), result.data.get(), numels, f);
    cudaErrCheck(cudaDeviceSynchronize());
}
*/
#endif // MATRIX_OPS_CUH