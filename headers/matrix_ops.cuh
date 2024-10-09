#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <random>
#include <vector>
#include "functors.cuh"
#include "matrix.cuh"
#include "types"
#include "utils.hpp"

inline __device__ __host__ uint32 iDivUp(uint32 a, uint32 b) { return (a + b - 1) / b; }

inline uint32 nextPow2(uint32 n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
//////////////////////////////////////////////////////////////////////////////////////
// Specialized matrix ops
//////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void softmax_gradient(Matrix<T> &s_grad_out, const Matrix<T> &s_out, const Matrix<T> &grad_in);

//////////////////////////////////////////////////////////////////////////////////////
// Matrix ops
//////////////////////////////////////////////////////////////////////////////////////

// reduces along width, identity is identity scalar under that operation, PostProcess is unary
// functor applied to final result
template <typename T, typename ReduceOp = Plus<T>, typename PostProcess = Identity<T>>
void reduce(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduction = ReduceOp(),
            T identity = ReduceOp::Identity, PostProcess pProcess = PostProcess());

// reduces along height, for column vectors, throws if .width > 1
template <typename T, typename ReduceOp, typename PostProcess>
void reduce_column_vec(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduceOp, T identity,
                       PostProcess postProcess);

template <typename T, typename PostProcess = Identity<T>>
void mvadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
           PostProcess pProcess = PostProcess());

template <typename T, typename PProcess = Identity<T>>
void mmadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
           PProcess pProcess = PProcess());

template <typename T, typename Op = Identity<T>>
void transpose(Matrix<T> &res, const Matrix<T> &A, Op op = Op());

template <typename Ta, typename Tb = Ta, typename Tr = Ta, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op);

template <typename Ta, typename Tr = Ta, typename Op>
void unary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, Op op);

//////////////////////////////////////////////////////////////////////////////////////
// Matrix initializations
//////////////////////////////////////////////////////////////////////////////////////

template <class T>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> normal_init(
    uint32 height, uint32 width, float32 mean = 0.f, float32 std = 1.f)
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
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> xavier_init(
    uint32 height, uint32 width)
{
    return normal_init<T>(height, width, 0.f, std::sqrt(2.0 / (height + width)));
}

template <typename T>
inline void fill(Matrix<T> &A, const T *values)
{
    cudaMemcpy(A.begin(), values, A.numels() * sizeof(T), cudaMemcpyDefault);
}

template <typename T>
inline void fill(Matrix<T> &A, const Matrix<T> &B)
{
    if (A.height != B.height or A.width != B.width)
    {
        LOG(RED, "Dimension mismatch: A, B: ", A.shape_str, " != ", B.shape_str);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
    fill(A, B.begin());
}

template <typename FloatT>
Matrix<FloatT> read_csv(const std::string &filename)
{
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open())
    {
        throw runtime_error_with_backtrace("Could not open file " + filename);
    }
    uint32 m, n;
    file >> m >> n;
    std::vector<FloatT> data(m * n);
    using readT = typename AccumT<FloatT>::type;
    std::copy(std::istream_iterator<readT>(file), std::istream_iterator<readT>(), data.begin());
    Matrix<FloatT> matrix(m, n, data.data());
    return matrix;
}

template <typename FloatT>
Matrix<FloatT> shaped_like(const Matrix<FloatT> &like, const FloatT *values = nullptr)
{
    return Matrix<FloatT>(like.height, like.width, values);
}

template <typename FloatT>
Matrix<FloatT> I(uint32 n)
{
    Matrix<FloatT> m(n, n);
    for (uint32 i = 0; i < n; i++)
    {
        m(i, i) = 1;
    }
    return m;
}

template <typename FloatT>
Matrix<FloatT> zeros(uint32 m, uint32 n)
{
    Matrix<FloatT> r(m, n);
    cudaMemset(r.begin(), 0, r.numels() * sizeof(FloatT));
    return r;
}

//////////////////////////////////////////////////////////////////////////////////////
// Specialized calls to above functions
//////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void reduce_mean(Matrix<T> &result, const Matrix<T> &A)
{
    if (A.width > 1)
        reduce(result, A, Plus<T>(), T(0), DividebBy<T>(A.width));
    else if (A.height > 1 and A.width == 1 and result.width == 1)
        reduce_column_vec(result, A, Plus<T>(), T(0), DividebBy<T>(A.height));
    else
        throw runtime_error_with_backtrace("Invalid dimensions for mean reduction " + A.shape_str +
                                           " to " + result.shape_str);
}

template <typename T>
void reduce_sum(Matrix<T> &result, const Matrix<T> &A)
{
    if (A.width > 1)
        reduce(result, A, Plus<T>(), T(0));
    else if (A.height > 1 and A.width == 1 and result.width == 1)
        reduce_column_vec(result, A, Plus<T>(), T(0), Identity<T>());
    else
        throw runtime_error_with_backtrace("Invalid dimensions for sum reduction " + A.shape_str +
                                           " to " + result.shape_str);
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
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, A ", A.name, " and B ", B.name,
            " and result ", result.name);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
    if (result.height != A.height || result.width != B.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, result ", result.name, " and A ",
            A.name, " and B ", B.name);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
    if (C and
        ((C->height != A.height and C->height != 1) or (C->width != B.width and C->width != 1)))
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, C ", C->name, " and A ", A.name,
            " and B ", B.name);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
    return true;
}

template <typename T>
void check_broadcast_sizes(const Matrix<T> &res, const Matrix<T> &A, const Matrix<T> &B)
{
    uint32 rh = std::max(A.height, B.height);
    uint32 rw = std::max(A.width, B.width);

    if (res.height != rh or res.width != rw)
    {
        LOG(RED, "Dimension mismatch in Broadcating Binary Op: Result: ", res.shape_str,
            ", A: ", A.shape_str, " & B: ", B.shape_str);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }

    if (!(A.numels() == 1 or (rh == A.height and rw == A.width) or
          (rh != A.height and A.height == 1) or (rh != A.width and A.width == 1)))
    {
        LOG(RED, "Dimension mismatch in Broadcating Binary Op: Res: ", A.shape_str,
            " & A: ", A.shape_str);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }

    if (!(B.numels() == 1 or (rh == B.height and rw == B.width) or
          (rh != B.height and B.height == 1) or (rh != B.width and B.width == 1)))
    {
        LOG(RED, "Dimension mismatch in Broadcating Binary Op: Res: ", A.shape_str,
            " & B: ", B.shape_str);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
}

template <typename T>
inline void check_softmax_grad_sizes(const Matrix<T> &s_grad_out, const Matrix<T> &s_out,
                                     const Matrix<T> &grad_in)
{
    auto size_or_tx_match = [&s_grad_out](uint32 h, uint32 w) {
        if (h == s_grad_out.height and w == s_grad_out.width) return true;
        return (w == s_grad_out.height and h == s_grad_out.width);
    };
    if (!size_or_tx_match(s_out.height, s_out.width) or
        !size_or_tx_match(grad_in.height, grad_in.width))
    {
        LOG(RED, "Dimension mismatch in softmax gradient: s_grad_out: ", s_grad_out.shape_str,
            ", s_out: ", s_out.shape_str, " & grad_in: ", grad_in.shape_str);
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
}

#endif  // MATRIX_OPS_CUH