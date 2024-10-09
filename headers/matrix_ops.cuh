#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include "matrix.cuh"
#include "types"
#include "utils.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <random>
#include <vector>

template <typename T> void fill(Matrix<T> &A, float value) { std::fill(A.begin(), A.end(), value); }

template <typename T> inline void fill(Matrix<T> &A, const float *values)
{
    std::copy(values, values + A.nuemls(), A.begin());
}

template <typename T> bool same(const Matrix<T> &A, const Matrix<T> &B, float eps = 1e-5)
{
    return std::equal(A.begin(), A.end(), B.begin(),
                      [eps](T a, T b) { return std::abs(a - b) < eps; });
}

template <typename T>
void madd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C);

template <typename T> Matrix<T> madd(const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C);

template <typename T> Matrix<T> transpose(const Matrix<T> &A);

template <typename T> Matrix<T> transposeCPU(const Matrix<T> &A)
{
    Matrix<T> res(A.width, A.height);
    for (uint32 y = 0; y < A.height; y++)
    {
        for (uint32 x = 0; x < A.width; x++)
        {
            res(y, x) = A(x, y);
        }
    }
    return res;
}

template <class T>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type>
normal_init(uint32 height, uint32 width, float32 mean = 0.f, float32 std = 1.f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    using gen_type = typename AccumT<T>::type;
    std::normal_distribution<float32> dist(mean, std);
    std::vector<T> values(height * width);
    std::generate(values.begin(), values.end(), [&dist, &gen]() { return dist(gen); });
    Matrix<T> out(height, width, values.data());
    return out;
}

template <class T>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type>
xavier_init(uint32 height, uint32 width)
{
    return normal_init<T>(height, width, 0.f, std::sqrt(2.0 / (height + width)));
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
bool check_madd_sizes(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B,
                      const Matrix<Tc> *C)
{
    if (A.width != B.height || A.height != result.height || B.width != result.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for A ", A.get_name(), " and B ",
            B.get_name(), " and result ", result.get_name());
        throw std::runtime_error("Dimension mismatch");
    }
    if (result.height != A.height || result.width != B.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for result ", result.get_name(), " and A ",
            A.get_name(), " and B ", B.get_name());
        throw std::runtime_error("Dimension mismatch");
    }
    if (C and (C->height != A.height or C->width != B.width))
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for C ", C->get_name(), " and A ",
            A.get_name(), " and B ", B.get_name());
        throw std::runtime_error("Dimension mismatch");
    }
    return true;
}

#endif // MATRIX_OPS_CUH