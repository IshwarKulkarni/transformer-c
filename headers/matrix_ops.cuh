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

inline __device__ __host__ uint32 iDivUp(uint32 a, uint32 b) { return (a + b - 1) / b; }

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mvadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C);

template <typename Tr, typename Ta, typename Tb, typename Tc>
void mmadd(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B, const Matrix<Tc> *C);

template <typename T> void transpose(Matrix<T> &res, const Matrix<T> &A);

// reduces along width, identity is identity under that operation
template <typename T, typename Op>
void reduce(Matrix<T> &result, const Matrix<T> &A, const Op &op = Op(), T identity = Op::Identity);

template <typename T> inline void fill(Matrix<T> &A, const T *values)
{
    cudaMemcpy(A.begin(), values, A.numels() * sizeof(T), cudaMemcpyDefault);
}

template <typename Ta, typename Tb = Ta> struct Plus
{
    static constexpr Ta Identity = 0;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a + b; }
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

template <typename Ta, typename Tb = Ta> struct Sub
{
    static constexpr Ta Identity = 0;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a - b; }
};

template <typename Ta, typename Tb = Ta> struct Mul
{
    static constexpr Ta Identity = 1;
    __host__ __device__ inline Ta operator()(Ta a, Tb b) const { return a * b; }
};

template <typename T> void reduce_sum(Matrix<T> &res, const Matrix<T> &A)
{
    reduce<T, Plus<T>>(res, A);
}
template <typename T> void reduce_max(Matrix<T> &res, const Matrix<T> &A)
{
    reduce<T, Max<T>>(res, A);
}

template <typename T> void reduce_min(Matrix<T> &res, const Matrix<T> &A)
{
    reduce<T, Min<T>>(res, A);
}

template <typename T> struct Neg
{
    inline __host__ __device__ T operator()(const T x) { return -x; }
};

template <typename Ta, typename Tb = Ta, typename Tr = Ta, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op);

template <typename Ta, typename Tr = Ta, typename Op>
void unary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, Op op);

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