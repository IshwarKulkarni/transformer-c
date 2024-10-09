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

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mvadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C);

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mmadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C);

template <typename T> void transpose(Matrix<T> &res, const Matrix<T> &A);

// reduces along width, identity is identity under that operation
template <typename T, typename Op>
void reduce(Matrix<T> &result, const Matrix<T> &A, const Op &op = Op(), T identity = Op::Identity);

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mmaddCPU(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C)
{
    check_mmadd_sizes(result, A, B, C);
    for (uint32 y = 0; y < A.height; y++)
    {
        for (uint32 x = 0; x < B.width; x++)
        {
            Tr value = 0;
            for (uint32 k = 0; k < A.width; k++)
            {
                value += A(k, y) * B(x, k);
            }
            if (C)
            {
                value += C->operator()(x, y);
            }
            result(x, y) = value;
        }
    }
}

template <typename T> inline void fillCPU(Matrix<T> &A, const T &value)
{
    std::fill(A.begin(), A.end(), value);
}

template <typename T> inline void fill(Matrix<T> &A, const T *values)
{
    cudaMemcpy(A.begin(), values, A.numels() * sizeof(T), cudaMemcpyDefault);
}

template <typename T> bool same(const Matrix<T> &A, const Matrix<T> &B, float eps = 1e-5)
{
    return std::equal(A.begin(), A.end(), B.begin(),
                      [eps](T a, T b) { return std::abs(a - b) < eps; });
}

template <typename T> void transposeCPU(Matrix<T> &res, const Matrix<T> &A)
{
    for (uint32 y = 0; y < A.height; y++)
    {
        for (uint32 x = 0; x < A.width; x++)
        {
            res(y, x) = A(x, y);
        }
    }
}

template <typename T> struct Plus
{
    static constexpr T Identity = 0;
    __host__ __device__ inline T operator()(T a, T b) const { return a + b; }
};

template <typename T> struct Max
{
    static constexpr T Identity = std::numeric_limits<T>::lowest();
    __host__ __device__ inline T operator()(T a, T b) const { return (a > b ? a : b); }
};

template <typename T> struct Min
{
    static constexpr T Identity = std::numeric_limits<T>::max();
    __host__ __device__ inline T operator()(T a, T b) const { return (a <= b ? a : b); }
};

template <typename T, typename Op>
void reduceCPU(Matrix<T> &result, const Matrix<T> &A, T identity = Op::Identity,
               const Op &op = Op())
{
    if (result.height != A.height || result.width != 1)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for reduce operation");
        throw std::runtime_error("Dimension mismatch");
    }
    for (uint32 y = 0; y < A.height; y++)
    {
        T value = identity;
        for (uint32 x = 0; x < A.width; x++)
        {
            value = op(value, A(x, y));
        }
        result(0, y) = value;
    }
}

template <typename T> void sum(Matrix<T> &res, const Matrix<T> &A) { reduce<T, Plus<T>>(res, A); }
template <typename T> void max(Matrix<T> &res, const Matrix<T> &A) { reduce<T, Max<T>>(res, A); }
template <typename T> void min(Matrix<T> &res, const Matrix<T> &A) { reduce<T, Min<T>>(res, A); }

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
bool check_mmadd_sizes(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B,
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