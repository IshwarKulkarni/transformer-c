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

// same as mmadd but B is transposed. I.e. result = A * B^T + C (if C is not null)
template <typename T, typename PProcess = Identity<T>>
void mmTadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
            PProcess pProcess = PProcess());

template <typename T>
inline void multiply(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B)
{
    mmadd<T, Identity<T>>(result, A, B, nullptr);
}

template <typename T, typename Op = Identity<T>>
void transpose(Matrix<T> &res, const Matrix<T> &A, Op op = Op());

template <typename Ta, typename Tb = Ta, typename Tr = Ta, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op = Op());

template <typename Ta, typename Tr = Ta, typename Op>
void unary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, Op op = Op());

template <typename T, typename Op = Identity<T>>
void concat(Matrix<T> &res, const std::vector<Matrix<T> *> &inputs, Op op = Op());

template <typename T, typename Op = Identity<T>>
void split(std::vector<Matrix<T> *> &outputs, const Matrix<T> &res, Op op = Op());

// ```
//    if  0<p<=1
//      res[x,y] = (mask[x, y] = (rand() < drop_prob)) ? 0 : res[x, y];
//    else:
//      res[x,y] = A[x,y] * mask[x,y]
// ```
template <typename T>
void dropout(Matrix<T> &res, const Matrix<T> &in, Matrix<float32> &mask, float32 drop_prob);

//////////////////////////////////////////////////////////////////////////////////////
// Matrix initializations
//////////////////////////////////////////////////////////////////////////////////////

struct rdm
{
    static std::random_device rd;
    static std::mt19937 rdm_gen;
    static std::mt19937 det_gen;
    static std::seed_seq seed;

    static bool deterministic;
    static std::mt19937 &gen()
    {
        if (deterministic) return det_gen;
        return rdm_gen;
    }
};

template <class T = FloatT>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> normal_init(
    uint32 height, uint32 width, float32 mean = 0.f, float32 std = 1.f, std::string name = "Matrix")
{
    using gen_type = typename AccumT<T>::type;
    std::normal_distribution<gen_type> dist(mean, std);
    Matrix<T> out(height, width, name);
    std::generate(out.begin(), out.end(), [&dist]() { return dist(rdm::gen()); });
    return out;
}

template <typename T = FloatT>
inline void normal_init(Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> &out,
                        float32 mean = 0.f, float32 std = 1.f)
{
    using gen_type = typename AccumT<T>::type;
    std::normal_distribution<gen_type> dist(mean, std);
    std::generate(out.begin(), out.end(), [&dist]() { return dist(rdm::gen()); });
}

template <typename T = FloatT>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> xavier_uniform_init(
    uint32 height, uint32 width, std::string name = "Matrix")
{
    return normal_init<T>(height, width, 0.f, std::sqrt(6.0 / (height + width)), name);
}

template <typename T = FloatT>
inline void xavier_uniform_init(
    Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> &out)
{
    return normal_init<T>(out, 0.f, std::sqrt(6.0 / (out.height + out.width)));
}

template <typename T = FloatT>
inline void kaiming_init(Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> &out)
{
    return normal_init<T>(out, 0.f, std::sqrt(2.0 / out.height));
}

// uniform random initialization, enabled only for floating point types
template <typename T>
Matrix<is_floating_point<T>> uniform(uint32 m, uint32 n, T min = 0, T max = 1)
{
    std::uniform_real_distribution<T> dist(min, max);
    Matrix<T> out(m, n);
    std::generate(out.begin(), out.end(), [&dist]() { return dist(rdm::gen()); });
    return out;
}

template <typename T>
void arange(Matrix<is_floating_point<T>> &A)
{
    for (uint32 i = 0; i < A.numels(); i++)
    {
        A.begin()[i] = T(i);
    }
}

template <typename T>
void eye(Matrix<is_floating_point<T>> &A)
{
    if (A.height != A.width)
    {
        throw_rte_with_backtrace("Eye matrix must be square");
    }
    for (uint32 i = 0; i < A.height; i++)
    {
        A(i, i) = 1;
    }
}

template <typename T>
Matrix<is_floating_point<T>> eye(uint32 n)
{
    Matrix<is_floating_point<T>> A(n, n);
    eye(A);
    return A;
}

template <typename T>
void arange_over_sum(Matrix<is_floating_point<T>> &A)
{
    uint32 n = A.numels();
    T sum = n * (n - 1) / 2;
    for (uint32 i = 0; i < n; i++)
    {
        A.begin()[i] = T(i) / sum;
    }
}

template <typename T>
Matrix<is_floating_point<T>> arange_over_sum(uint32 m, uint32 n)
{
    Matrix<is_floating_point<T>> A(m, n);
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
        LOG(RED, "Dimension mismatch for fill A, B: ", A.shape_str, " != ", B.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    fill(A, B.begin());
}

template <typename Ta, typename Tb>
Matrix<Ta> &operator<<=(Matrix<Ta> &mat, const std::initializer_list<Tb> &values)
{
    if (values.size() != mat.numels())
    {
        throw_rte_with_backtrace("Values size mismatch for ", mat.name, " expected ", mat.numels(),
                                 " got ", values.size());
    }
    std::copy_n(values.begin(), mat.numels(), mat.begin());
    return mat;
}

template <typename T>
std::ifstream &operator>>(std::ifstream &file, Matrix<T> &mat)
{
    if (!file.is_open())
    {
        throw_rte_with_backtrace("File not open");
    }
    uint32 m = 0, n = 0;
    file >> m >> n;
    if (m != mat.height or n != mat.width)
    {
        throw_rte_with_backtrace("Dimension mismatch in reading file for ", mat.name, mat.shape_str,
                                 " got ", m, "x", n);
    }
    using readT = typename AccumT<T>::type;
    std::copy_n(std::istream_iterator<readT>(file), m * n, mat.begin());
    return file;
}

template <typename T>
Matrix<FloatT> read_csv(std::ifstream &file)
{
    if (!file.is_open())
    {
        throw_rte_with_backtrace("Could not open file");
    }
    uint32 m, n;
    file >> m >> n;
    Matrix<FloatT> matrix(m, n);
    using readT = typename AccumT<FloatT>::type;
    std::copy_n(std::istream_iterator<readT>(file), m * n, matrix.begin());
    return matrix;
}

template <typename T>
Matrix<T> read_csv(const std::string &filename)
{
    std::ifstream file(filename, std::ios::in);
    if (!file)
    {
        throw_rte_with_backtrace("Could not open file ", filename);
    }
    return read_csv<T>(file);
}

template <typename T>  // write files that can be read with `numpy.loadtxt(filename,comments='#',
                       // delimiter=',')`
void write_csv(const Matrix<T> &m, const std::string &file = "")
{
    std::ofstream f(file.size() > 0 ? file : m.name + ".csv");
    f << '#' << m.width << ' ' << m.height << '\n';
    for (uint32 y = 0; y < m.height; y++)
    {
        for (uint32 x = 0; x < m.width; x++)
        {
            f << m(y, x) << (x == m.width - 1 ? "" : ",");
        }
        f << '\n';
    }
}

template <typename T>
Matrix<T> shaped_like(const Matrix<T> &like)
{
    return Matrix<T>(like.height, like.width);
}

template <typename T>
Matrix<T> I(uint32 n)
{
    Matrix<T> m(n, n);
    for (uint32 i = 0; i < n; i++)
    {
        m(i, i) = 1;
    }
    return m;
}

template <typename T>
Matrix<T> zeros(uint32 m, uint32 n)
{
    Matrix<T> r(m, n);
    cudaMemset(r.begin(), 0, r.numels() * sizeof(T));
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
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, A ", A.name, A.shape_str,
            " and B ", B.name, B.shape_str, " and result ", result.name, result.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    if (result.height != A.height || result.width != B.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, result ", result.name,
            result.shape_str, " and A ", A.name, A.shape_str, " and B ", B.name, B.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    if (C and
        ((C->height != A.height and C->height != 1) or (C->width != B.width and C->width != 1)))
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for MMADD, C ", C->name, C->shape_str,
            " and A ", A.name, A.shape_str, " and B ", B.name, B.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    return true;
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
bool check_mmTadd_sizes(Matrix<Tr> &result, const Matrix<Ta> &A, const Matrix<Tb> &B,
                        const Matrix<Tc> *C)
{
    if (!(A.width == B.width && A.height == result.height && B.height == result.width))
    {
        LOG(RED, "Dimension mismatch in mmTadd: A: ", A.shape_str, " * B: ", B.shape_str,
            " -> Result: ", result.shape_str);
        throw_rte_with_backtrace("Dimension mismatch");
    }
    if (C == nullptr) return true;

    // check if C is broadcastable
    bool C_shape_valid = C->shape() == result.shape() or
                         (C->height == 1 and C->width == result.width) or
                         (C->width == 1 and C->height == result.height);

    if (!C_shape_valid)
    {
        LOG(RED, "Dimension mismatch in mmTadd: Result: ", result.shape_str,
            " & C: ", C->shape_str);
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

template <typename T>  // bilinear interpolation, like texture mode border
T bilinear_sample(const Matrix<T> &m, float64 y, float64 x)
{
    if (x < 0 or x >= 1 or y < 0 or y >= 1) return 0;
    uint32 y0 = static_cast<uint32>(y * (m.height - 1));
    uint32 x0 = static_cast<uint32>(x * (m.width - 1));
    uint32 y1 = y0 + 1;
    uint32 x1 = x0 + 1;
    if (y1 >= m.height) y1 = y0;
    if (x1 >= m.width) x1 = x0;

    float64 y_frac = y * m.height - y0;
    float64 x_frac = x * m.width - x0;

    float64 v0 = float64(m(y0, x0)) * (1 - y_frac) * (1 - x_frac);
    float64 v1 = float64(m(y0, x1)) * (1 - y_frac) * x_frac;
    float64 v2 = float64(m(y1, x0)) * y_frac * (1 - x_frac);
    float64 v3 = float64(m(y1, x1)) * y_frac * x_frac;
    float64 v(v0 + v1 + v2 + v3);
    return T(v);
}

typedef Matrix<FloatT> Matrixf;
// return 0 if out of bounds, return grid point value if on grid within "eps" distance
// https://www.paulinternet.nl/?page=bicubic, should probably use the faster implem here, or jsut
// use textures;
template <typename T>
T sample(const Matrix<T> &m, float64 y, float64 x, float64 eps = 1e-8)
{
    auto cubicInterpolate = [](const std::array<double, 4> &p, float64 x) {
        return p[1] + 0.5 * x *
                          (p[2] - p[0] +
                           x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] +
                                x * (3.0 * (p[1] - p[2]) + p[3] - p[0])));
    };

    y *= m.height;
    x *= m.width;
    int32 y1 = static_cast<int32>(y);
    int32 x1 = static_cast<int32>(x);

    if (std::abs(y - y1) < eps and std::abs(x - x1) < eps and x1 < m.width and y1 < m.height)
    {
        return m(y1, x1);
    }

    std::array<std::array<double, 4>, 4> p;

    for (int32 j = -1; j <= 2; ++j)
    {
        for (int32 i = -1; i <= 2; ++i)
        {
            int32 yj = std::max(0, std::min(static_cast<int32>(m.height - 1), y1 + j));
            int32 xi = std::max(0, std::min(static_cast<int32>(m.width - 1), x1 + i));
            p[i + 1][j + 1] = m(yj, xi);
        }
    }

    std::array<double, 4> arr;
    for (int32 i = 0; i < 4; ++i)
    {
        arr[i] = cubicInterpolate(p[i], y - y1);
    }

    return cubicInterpolate(arr, x - x1);
}

// this does not currently work, returns gradients that are some scalar multiple of actual.
template <typename T>
T gradient_x(const Matrix<T> &m, float64 y, float64 x, float64 d = 1e-2, float64 eps = 1e-8)
{
    auto mid = sample(m, y, x, eps);
    if (x < 0 or x >= 1) return mid;
    if (std::abs(x) < eps)  //−3f(x)+4f(x+∆x)−f(x+2∆x)/2∆x, forwards difference
        return (-3 * mid + 4 * sample(m, y, x + d, eps) - sample(m, y, x + 2 * d, eps)) / (2 * d);

    if (std::abs(x - 1) < eps)  // 3f(x)−4f(x−∆x)+f(x−2∆x) /(2∆x), backwards difference
        return (3 * mid - 4 * sample(m, y, x - d, eps) + sample(m, y, x - 2 * d, eps)) / (2 * d);

    auto x2 = sample(m, y, x + d, eps);
    auto x1 = sample(m, y, x - d, eps);
    return (x2 - x1) / (2 * d);
}

// this does not currently work
template <typename T>
T gradient_y(const Matrix<T> &m, float64 y, float64 x, float64 d = 1e-2, float64 eps = 1e-8)
{
    auto mid = sample(m, y, x, eps);
    if (y < 0 or y >= 1) return mid;
    if (std::abs(y) < eps)  //−3f(x)+4f(x+∆x)−f(x+2∆x)/2∆x, forwards difference
        return (-3 * mid + 4 * sample(m, y + d, x, eps) - sample(m, y + 2 * d, x, eps)) / (2 * d);

    if (std::abs(y - 1) < eps)  // 3f(x)−4f(x−∆x)+f(x−2∆x) /(2∆x), backwards difference
        return (3 * mid - 4 * sample(m, y - d, x, eps) + sample(m, y - 2 * d, x, eps)) / (2 * d);

    auto y2 = sample(m, y + d, x, eps);
    auto y1 = sample(m, y - d, x, eps);
    return (y2 - y1) / (2 * d);
}

/*
Gradient using second order centerered difference at (y, x), fwd/bwd difference at edges
*/
template <typename T>
std::pair<T, T> gradient_xy(const Matrix<T> &m, float64 y, float64 x, float64 h = 1e-2,
                            float64 eps = 1e-8)
{
    return {gradient_y(m, y, x, h, eps), gradient_x(m, y, x, h, eps)};
}

template <typename T>
inline void resample(const Matrix<T> &mat_in, Matrix<T> &mat_out)
{
    for (uint32 y = 0; y < mat_out.height; y++)
    {
        float64 y_ = y / (float64)mat_out.height;
        for (uint32 x = 0; x < mat_out.width; x++)
        {
            float64 x_ = x / (float64)mat_out.width;
            mat_out(y, x) = bilinear_sample(mat_in, y_, x_);
        }
    }
}

template <typename T>
inline Matrix<T> resample(const Matrix<T> &mat_in, uint32 height, uint32 width)
{
    Matrix<T> mat_out(height, width);
    resample(mat_in, mat_out);
    return mat_out;
}

#endif  // MATRIX_OPS_CUH