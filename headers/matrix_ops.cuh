#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include "matrix.cuh"
#include "types"
#include <random>
#include "utils.hpp"

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

template<class T>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> 
normal_init(uint32 m, uint32 n, float32 mean=0.f, float32 std=1.f)
{
    Timer t("normal_init");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<typename AccumT<T>::type> dist(mean, std);
    std::vector<T> data(m * n);
    std::generate(data.begin(), data.end(), [&dist, &gen]() { return dist(gen); });
    return Matrix<T>(m, n, data.data());
}

template<class T>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> xavier_init(uint32 m, uint32 n)
{
    return normal_init<T>(m, n, 0.f, std::sqrt(2.0 / (m + n)));
}


#endif // MATRIX_OPS_CUH