#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include "functors.cuh"
#include "matrix.cuh"
#include "types"
#include "utils.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <functional>
#include <random>
#include <vector>

inline __device__ __host__ uint32 iDivUp(uint32 a, uint32 b) { return (a + b - 1) / b; }

//////////////////////////////////////////////////////////////////////////////////////
// Matrix ops
//////////////////////////////////////////////////////////////////////////////////////

// reduces along width, identity is identity scalar under that operation, PostProcess is unary
// functor applied to final result
template <typename T, typename ReduceOp = Plus<T>, typename PostProcess = Identity<T>>
void reduce(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduction = ReduceOp(),
            T identity = ReduceOp::Identity, PostProcess process = PostProcess());

// reduces along height, for column vectors, throws if .width > 1
template <typename T, typename ReduceOp, typename PostProcess>
void reduce_column_vec(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduceOp, T identity,
                       PostProcess postProcess);

template <typename T, typename PostProcess = Identity<T>>
void mvadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
           PostProcess process = PostProcess());

template <typename T, typename PProcess = Identity<T>>
void mmadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
           PProcess process = PProcess());

template <typename T> void transpose(Matrix<T> &res, const Matrix<T> &A);

template <typename Ta, typename Tb = Ta, typename Tr = Ta, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op);

template <typename Ta, typename Tr = Ta, typename Op>
void unary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, Op op);

//////////////////////////////////////////////////////////////////////////////////////
// Matrix initializations
//////////////////////////////////////////////////////////////////////////////////////

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

template <typename T> inline void fill(Matrix<T> &A, const T *values)
{
    cudaMemcpy(A.begin(), values, A.numels() * sizeof(T), cudaMemcpyDefault);
}
//////////////////////////////////////////////////////////////////////////////////////
// Specialized calls to above functions
//////////////////////////////////////////////////////////////////////////////////////

template <typename T> void reduce_mean(Matrix<T> &result, const Matrix<T> &A)
{
    if (A.width > 1)
        reduce(result, A, Plus<T>(), T(0), DividebBy<T>(A.width));
    else if (A.height > 1 and A.width == 1 and result.width == 1)
        reduce_column_vec(result, A, Plus<T>(), T(0), DividebBy<T>(A.height));
    else
        throw std::runtime_error("Invalid dimensions for mean reduction " + A.shape_string() +
                                 " to " + result.shape_string());
}

template <typename T> void reduce_sum(Matrix<T> &result, const Matrix<T> &A)
{
    if (A.width > 1)
        reduce(result, A, Plus<T>(), T(0));
    else if (A.height > 1 and A.width == 1 and result.width == 1)
        reduce_column_vec(result, A, Plus<T>(), T(0));
    else
        throw std::runtime_error("Invalid dimensions for sum reduction " + A.shape_string() +
                                 " to " + result.shape_string());
}

//////////////////////////////////////////////////////////////////////////////////////
// Size checks
//////////////////////////////////////////////////////////////////////////////////////

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