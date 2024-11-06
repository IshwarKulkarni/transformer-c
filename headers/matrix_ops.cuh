#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include <algorithm>
#include <functional>
#include <iterator>
#include <random>
#include <type_traits>
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

template <typename T>
inline void multiply(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B)
{
    mmadd<T, Identity<T>>(result, A, B, nullptr);
}

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
    std::normal_distribution<gen_type> dist(mean, std);
    std::vector<T> values(height * width);
    std::generate(values.begin(), values.end(), [&dist, &gen]() { return dist(gen); });
    return Matrix<T>(height, width, values.data());
}

template <class T>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> xavier_init(
    uint32 height, uint32 width)
{
    return normal_init<T>(height, width, 0.f, std::sqrt(2.0 / (height + width)));
}

// uniform random initialization, enabled only for floating point types
template <typename FloatT>
Matrix<is_floating_point<FloatT>> uniform(uint32 m, uint32 n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<FloatT> dist(0, 1);
    std::vector<FloatT> values(m * n);
    std::generate(values.begin(), values.end(), [&dist, &gen]() { return dist(gen); });
    return Matrix<FloatT>(m, n, values.data());
}

template <typename FloatT>
void arange(Matrix<is_floating_point<FloatT>> &A)
{
    for (uint32 i = 0; i < A.numels(); i++)
    {
        A.begin()[i] = FloatT(i);
    }
}

template <typename FloatT>
void arange_over_sum(Matrix<is_floating_point<FloatT>> &A)
{
    uint32 n = A.numels();
    FloatT sum = n * (n - 1) / 2;
    for (uint32 i = 0; i < n; i++)
    {
        A.begin()[i] = FloatT(i) / sum;
    }
}

template <typename FloatT>
Matrix<is_floating_point<FloatT>> arange_over_sum(uint32 m, uint32 n)
{
    Matrix<is_floating_point<FloatT>> A(m, n);
    arange_over_sum(A);
    return A;
}

template <typename T>
inline void fill(Matrix<T> &A, const T *values)
{
    if (values != nullptr)
    {
        cudaErrCheck(cudaMemcpy(A.begin(), values, A.numels() * sizeof(T), cudaMemcpyDefault));
        return;
    }
    cudaMemset(A.begin(), 0, A.numels() * sizeof(T));
}

template <typename T>
inline void fill(Matrix<T> &A, const Matrix<T> &B)
{
    if (A.height != B.height or A.width != B.width)
    {
        LOG(RED, "Dimension mismatch: A, B: ", A.shape_str, " != ", B.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    fill(A, B.begin());
}

template <typename T>
Matrix<T> &operator<<=(Matrix<T> &mat, const std::initializer_list<T> &values)
{
    if (values.size() != mat.numels())
    {
        throw_rte_with_backtrace("Values size mismatch for ", mat.name, " expected ", mat.numels(),
                                 " got ", values.size());
    }
    fill(mat, values.begin());
    return mat;
}

template <typename T>
std::ifstream &operator>>(std::ifstream &file, Matrix<T> &mat)
{
    uint32 m = 0, n = 0;
    file >> m >> n;
    if (m != mat.height or n != mat.width)
    {
        throw_rte_with_backtrace("Dimension do not match for ", mat.name, " got ", m, "x", n);
    }
    using readT = typename AccumT<FloatT>::type;
    std::copy_n(std::istream_iterator<readT>(file), m * n, mat.begin());
    return file;
}

template <typename FloatT>
Matrix<FloatT> read_csv(const std::string &filename)
{
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open())
    {
        throw_rte_with_backtrace("Could not open file ", filename);
    }
    uint32 m, n;
    file >> m >> n;
    Matrix<FloatT> matrix(m, n);
    using readT = typename AccumT<FloatT>::type;
    std::copy_n(std::istream_iterator<readT>(file), m * n, matrix.begin());
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

template <typename T = FloatT>
Matrix<T> ones(uint32 m, uint32 n)
{
    Matrix<T> r(m, n);
    fillCPU(r, FloatT(1.));
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
        throw_rte_with_backtrace("Invalid dimensions for mean reduction ", A.shape_str, " to ",
                                 result.shape_str);
}

template <typename T>
void reduce_sum(Matrix<T> &result, const Matrix<T> &A)
{
    if (A.width > 1)
        reduce(result, A, Plus<T>(), T(0));
    else if (A.height > 1 and A.width == 1 and result.width == 1)
        reduce_column_vec(result, A, Plus<T>(), T(0), Identity<T>());
    else
        throw_rte_with_backtrace("Invalid dimensions for sum reduction ", A.shape_str, " to ",
                                 result.shape_str);
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
        throw_rte_with_backtrace("Dimension mismatch");
    }
    if (result.height != A.height || result.width != B.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, result ", result.name, " and A ",
            A.name, " and B ", B.name);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    if (C and
        ((C->height != A.height and C->height != 1) or (C->width != B.width and C->width != 1)))
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, C ", C->name, " and A ", A.name,
            " and B ", B.name);
        throw_rte_with_backtrace("Dimension mismatch");
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
        throw_rte_with_backtrace("Dimension mismatch");
    }

    if (!(A.numels() == 1 or (rh == A.height and rw == A.width) or
          (rh != A.height and A.height == 1) or (rh != A.width and A.width == 1)))
    {
        LOG(RED, "Dimension mismatch in Broadcating Binary Op: Res: ", A.shape_str,
            " & A: ", A.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }

    if (!(B.numels() == 1 or (rh == B.height and rw == B.width) or
          (rh != B.height and B.height == 1) or (rh != B.width and B.width == 1)))
    {
        LOG(RED, "Dimension mismatch in Broadcating Binary Op: Res: ", A.shape_str,
            " & B: ", B.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
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
        throw_rte_with_backtrace("Dimension mismatch");
    }
}

#endif  // MATRIX_OPS_CUH