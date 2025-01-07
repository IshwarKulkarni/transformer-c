#ifndef MATRIX_OPS_CUH
#define MATRIX_OPS_CUH

#include <cuda_runtime.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <random>
#include <type_traits>
#include <vector>
#include "errors.hpp"
#include "functors.cuh"
#include "logger.hpp"
#include "matrix.cuh"
#include "types"
#include "utils.hpp"

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

// reduces along `dim`, identity is identity scalar under that operation, PostProcess is unary
// functor applied to final result
// Default template args are to sum along width.
template <typename T, uint32 dim = 0, typename ReduceOp = Plus<T>,
          typename PostProcess = Identity<T>>
void reduce(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduction = ReduceOp(),
            T identity = ReduceOp::Identity, PostProcess pProcess = PostProcess());

template <typename T, typename PostProcess = Identity<T>>
void mvadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B,
           const Optional<Matrix<T>> C = {}, PostProcess pProcess = PostProcess());

template <typename T, typename PProcess = Identity<T>>
void mmadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B,
           const Optional<Matrix<T>> C = {}, PProcess pProcess = PProcess());

// same as mmadd but B is transposed. I.e. result = A * B^T + C (if C is not null)
template <typename T, typename PProcess = Identity<T>>
void mmTadd(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B,
            const Optional<Matrix<T>> C = {}, PProcess pProcess = PProcess());

template <typename T>
inline void multiply(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B)
{
    mmadd<T, Identity<T>>(result, A, B, {});
}

template <typename T, typename Op = Identity<T>>
void transpose(Matrix<T> &res, const Matrix<T> &A, Op op = Op());

template <typename Tr, typename Ta, typename Tb, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op = Op());

template <typename Ta, typename Tr = Ta, typename Op>
void unary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, Op op = Op());

template <typename T, uint32 Dim = 0,
          typename Op = Identity<T>>  // Dim = 0: width, 1:height, 2:batch
void concat(Matrix<T> &res, const std::vector<Matrix<T> *> &inputs, Op op = Op());

template <typename T, uint32 Dim = 0, typename Op = Identity<T>>
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
// Specialized calls to above functions
//////////////////////////////////////////////////////////////////////////////////////

template <typename T, uint32 dim = 0>
void reduce_mean(Matrix<T> &result, const Matrix<T> &A)
{
    reduce<T, dim>(result, A, Plus<T>(), T(0), DividedBy<T>(A.shape[dim]));
}

//////////////////////////////////////////////////////////////////////////////////////
// Some of the above calls that write to same matrix
//////////////////////////////////////////////////////////////////////////////////////

// inplace binary apply A = A op B
template <typename Ta, typename Tb, typename Op>
void binary_apply(Matrix<Ta> &A, const Matrix<Tb> &B, Op op = Op())
{
    binary_apply<Ta, Ta, Tb, Op>(A, A, B, op);
}

// inplace unary apply A = op(A)
template <typename Ta, typename Op>
void unary_apply(Matrix<Ta> &A, Op op = Op())
{
    unary_apply<Ta, Ta, Op>(A, A, op);
}

// inplace reduce, only elements where `index` == 0 are valid after this call,
// where `index` is batch if dim = 2, height if dim = 1, width if dim = 0
template <typename T = FloatT, uint32 dim = 0, typename ReduceOp = Plus<T>,
          typename PostProcess = Identity<T>>
void reduce(Matrix<T> &A, ReduceOp reduction = ReduceOp(), T identity = ReduceOp::Identity,
            PostProcess pProcess = PostProcess())
{
    reduce<T, dim>(A, A, reduction, identity, pProcess);
}

// Inplace reduce, same semantics as "broadcasting_fetch" in Matrix<T>.
// Reduce along dimension n when n'th bit is turned on, in order of width, height, batch.
// Reduction is done is opposite order, i.e. batch is first (if enabled)
template <typename T = FloatT, uint32 DimBits = WIDTH_BIT, typename RedOp = Plus<T>>
inline void reduce_multiple(Matrix<T> &A, RedOp reduction = RedOp(), T identity = RedOp::Identity)
{
    static_assert(DimBits <= 0b111, "dim bits must in [0b00, 0b111]");
    if (DimBits & BATCH_BIT) reduce<T, BATCH_IDX, RedOp>(A, reduction, identity);
    if (DimBits & HEIGHT_BIT) reduce<T, HEIGHT_IDX, RedOp>(A, reduction, identity);
    if (DimBits & WIDTH_BIT) reduce<T, WIDTH_IDX, RedOp>(A, reduction, identity);
}

// inplace reduce to scalar by reducing along batch, height and width in that order (as necessary).
// Post process is applied to the final result.
template <typename T, typename RedOp = Plus<T>, typename PostProcess = Identity<T>>
inline void reduce_to_scalar(Matrix<T> &A, RedOp reduction = RedOp(), T identity = RedOp::Identity,
                             PostProcess pProcess = PostProcess())
{
    if (A.shape[BATCH_IDX] > 1) reduce<T, BATCH_IDX, RedOp>(A, reduction, identity);
    if (A.shape[HEIGHT_IDX] > 1) reduce<T, HEIGHT_IDX, RedOp>(A, reduction, identity);
    if (A.shape[WIDTH_IDX] > 1) reduce<T, WIDTH_IDX, RedOp>(A, reduction, identity, pProcess);
    if (A.shape[WIDTH_IDX] == 1)
    {
        throw_rte_with_backtrace("Cannot apply post process");
    }
}

template <typename T>
inline void mean_batch(Matrix<T> &result, const Matrix<T> &A)
{
    reduce<T, BATCH_IDX>(result, A, Plus<T>(), T(0), DividedBy<T>(A.batch()));
}
//////////////////////////////////////////////////////////////////////////////////////
// Matrix initializations
//////////////////////////////////////////////////////////////////////////////////////

// clears L2 cache by allocating and writing to `size` bytes of memory
void clear_l2_cache(uint32 size = 256 * 1024 * 1024);

struct rdm
{
    static std::random_device rd;
    static std::mt19937_64 rdm_gen;
    static std::mt19937_64 det_gen;
    static std::seed_seq seed;

    static bool deterministic;
    static std::mt19937_64 &gen()
    {
        if (deterministic) return det_gen;
        return rdm_gen;
    }
};

template <typename T, typename DistType>
inline void gen(Matrix<T> &out, DistType dist)
{
    auto start = out.get().get();
    std::generate(start, start + out.numels(), [&dist]() { return dist(rdm::gen()); });
}

template <class T = FloatT>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> normal_init(
    Shape shape, float32 mean = 0.f, float32 std = 1.f, std::string name = "Matrix")
{
    Matrix<T> out(shape, name);
    gen(out, std::normal_distribution<typename AccumT<T>::type>(mean, std));
    return out;
}

template <typename T = FloatT>
inline void normal_init(Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> &out,
                        float32 mean = 0.f, float32 std = 1.f)
{
    using gen_type = typename AccumT<T>::type;
    std::normal_distribution<gen_type> dist(mean, std);
    gen(out, dist);
}

template <typename T = FloatT>
Matrix<T> init_argv(const char **argv, uint32 argc_offset = 2);

template <typename T = FloatT>
inline Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> xavier_uniform_init(
    Shape shape, std::string name = "Matrix")
{
    return normal_init<T>(shape, 0.f, std::sqrt(6.0 / (shape.height + shape.width)), name);
}

template <typename T = FloatT>
inline void xavier_uniform_init(
    Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> &out)
{
    return normal_init<T>(out, 0.f, std::sqrt(6.0 / (out.height() + out.width())));
}

template <typename T = FloatT>
inline void kaiming_init(Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> &out)
{
    return normal_init<T>(out, 0.f, std::sqrt(2.0 / out.height()));
}

// uniform random initialization, enabled only for floating point types
template <typename T>
Matrix<is_floating_point<T>> uniform(Shape shape, T min = 0, T max = 1)
{
    std::uniform_real_distribution<T> dist(min, max);
    Matrix<T> out(shape);
    return gen(out, dist);
}

template <typename T>
void arange(Matrix<is_floating_point<T>> &A)
{
    for (uint32 b = 0; b < A.batch(); b++)
    {
        for (uint32 i = 0; i < A.numels(); i++)
        {
            A[b * A.shape.size2d() + i] = T(i);
        }
    }
}

template <typename T>
void eye(Matrix<is_floating_point<T>> &A)
{
    if (A.height() != A.width())
    {
        throw_rte_with_backtrace("Eye matrix must be square");
    }

    for (uint32 b = 0; b < A.batch(); b++)
    {
        for (uint32 i = 0; i < A.height(); i++)
        {
            A(b, i, i) = 1;
        }
    }
}

template <typename T>
Matrix<is_floating_point<T>> eye(uint32 n)
{
    Matrix<is_floating_point<T>> A({1, n, n});
    eye(A);
    return A;
}

template <typename Ta, typename Tb>
inline Matrix<Ta> &operator<<=(Matrix<Ta> &mat, const std::initializer_list<Tb> &values)
{
    if (values.size() != mat.numels())
    {
        throw_rte_with_backtrace("Values size mismatch for ", mat.name, mat.shape, " expected ",
                                 mat.numels(), " got ", values.size());
    }
    mat.copy(values.begin());
    return mat;
}

template <typename T>
std::ifstream &operator>>(std::ifstream &file, Matrix<T> &mat)
{
    if (!file.is_open())
    {
        throw_rte_with_backtrace("File not open");
    }
    uint32 b = 0, h = 0, w = 0;
    char c;
    file >> c;
    file >> b >> h >> w;
    if (c != '#' or file.bad())
    {
        throw_rte_with_backtrace("Invalid file format, expected # b h w");
    }
    Shape shape(b, h, w);
    if (mat.shape.shape2d() != shape.shape2d())
    {
        throw_rte_with_backtrace("Shape mismatch in >> expected ", mat.shape, " got ", shape);
    }
    using readT = typename AccumT<T>::type;
    std::vector<readT> vec;
    vec.reserve(shape.numels);
    for (size_t i = 0; i < shape.numels and file; i++)
    {
        readT e;
        file >> e;
        vec.push_back(e);
    }
    if ((!file.eof() and file.bad()) or vec.size() != shape.numels)
        throw_rte_with_backtrace(shape.numels, " elements not read");

    if (shape.batch == 1)
    {
        if (mat.batch() != 1)
            LOG(YELLOW "Broadcasting 1 batch to ", mat.batch(), " in read >> for ", mat.name);
        for (uint32 i = 0; i < mat.batch(); ++i) mat.copy(vec.data(), {i});
    }
    else if (mat.batch() != shape.batch)
    {
        throw_rte_with_backtrace("Batch size mismatch in >> expected ", mat.batch(), " got ",
                                 shape.batch);
    }
    else
    {
        mat.copy(vec.data());
    }
    return file;
}

template <typename T>
Matrix<T> read_csv(std::ifstream &file);

template <typename T>
Matrix<T> read_csv(const std::string &filename);

// Read sparse matrix from file, format is:
// # batch height width
// batch, height, width: value
template <typename T>
Matrix<typename std::enable_if<is_floating_point<T>::value, T>::type> read_sparse(
    const std::string &filename)
{
    auto fd = fopen(filename.c_str(), "r");
    if (!fd) throw_rte_with_backtrace("Could not open file ", filename);

    uint32 b = 0, h = 0, w = 0;
    if (fscanf(fd, "# %u %u %u\n", &b, &h, &w) != 3)
        throw_rte_with_backtrace("Invalid file format, expected # b h w");

    Matrix<T> m({b, h, w});
    while (!feof(fd))
    {
        T val;
        if (fscanf(fd, "%u %u %u: %f\n", &b, &h, &w, &val) != 4)
            throw_rte_with_backtrace("Invalid file format, expected b h w: val");
        m(b, h, w) = val;
    }
}

template <typename T>  // write files that can be read with `numpy.loadtxt(filename,comments='#',
                       // delimiter=',')` followed by reshape.
void write_csv(const Matrix<T> &m, const std::string &file = "")
{
    std::ofstream f(file.size() > 0 ? file : m.name + ".csv");
    f << '#' << m.batch() << ' ' << m.height() << ' ' << m.width() << '\n';
    for (uint32 b = 0; b < m.batch(); b++)
    {
        for (uint32 y = 0; y < m.height(); y++)
        {
            for (uint32 x = 0; x < m.width(); x++)
            {
                f << m(b, y, x) << (x == m.width() - 1 ? "" : ",");
            }
            f << '\n';
        }
        f << '\n';
    }
}

template <typename T>
void write_binary(const Matrix<T> &m, const std::string &file = "");

template <typename T>
Matrix<T> read_binary(const std::string &file = "");

template <typename T>
Matrix<T> shaped_like(const Matrix<T> &like, const std::string &name = "Matrix")
{
    return Matrix<T>(like.shape, name);
}

template <typename T>
Matrix<T> I(uint32 n)
{
    Matrix<T> m(1, n, n);
    for (uint32 i = 0; i < n; i++)
    {
        m(i, i) = 1;
    }
    return m;
}

template <typename T>  // bilinear interpolation, like texture mode border
T bilinear_sample(const Matrix<T> &m, uint32 b, float64 y, float64 x);

typedef Matrix<FloatT> Matrixf;
// return 0 if out of bounds, return grid point value if on grid within "eps" distance
// https://www.paulinternet.nl/?page=bicubic, should probably use the faster implem here, or jsut
// use textures;
template <typename T>
T sample(const Matrix<T> &m, uint32 b, float64 y, float64 x, float64 eps = 1e-8);

// this does not currently work, returns gradients that are some scalar multiple of actual.
template <typename T>
T gradient_x(const Matrix<T> &m, uint32 b, float64 y, float64 x, float64 d = 1e-2,
             float64 eps = 1e-8)
{
    auto mid = sample(m, b, y, x, eps);
    if (x < 0 or x >= 1) return mid;
    if (std::abs(x) < eps)  // −3f(x)+4f(x+∆x)−f(x+2∆x)/2∆x, forwards difference
        return (-3 * mid + 4 * sample(m, b, y, x + d, eps) - sample(m, y, x + 2 * d, eps)) /
               (2 * d);

    if (std::abs(x - 1) < eps)  // 3f(x)−4f(x−∆x)+f(x−2∆x) /(2∆x), backwards difference
        return (3 * mid - 4 * sample(m, b, y, x - d, eps) + sample(m, y, x - 2 * d, eps)) / (2 * d);

    auto x2 = sample(m, b, y, x + d, eps);
    auto x1 = sample(m, b, y, x - d, eps);
    return (x2 - x1) / (2 * d);
}

// this does not currently work
template <typename T>
T gradient_y(const Matrix<T> &m, uint32 b, float64 y, float64 x, float64 d = 1e-2,
             float64 eps = 1e-8)
{
    auto mid = sample(m, b, y, x, eps);
    if (y < 0 or y >= 1) return mid;
    if (std::abs(y) < eps)  // −3f(x)+4f(x+∆x)−f(x+2∆x)/2∆x, forwards difference
        return (-3 * mid + 4 * sample(m, b, y + d, x, eps) - sample(m, y + 2 * d, x, eps)) /
               (2 * d);

    if (std::abs(y - 1) < eps)  // 3f(x)−4f(x−∆x)+f(x−2∆x) /(2∆x), backwards difference
        return (3 * mid - 4 * sample(m, b, y - d, x, eps) + sample(m, y - 2 * d, x, eps)) / (2 * d);

    auto y2 = sample(m, b, y + d, x, eps);
    auto y1 = sample(m, b, y - d, x, eps);
    return (y2 - y1) / (2 * d);
}

/*
Gradient using second order centerered difference at (y, x), fwd/bwd difference at edges
*/
template <typename T>
std::pair<T, T> gradient_xy(const Matrix<T> &m, uint32 b, float64 y, float64 x, float64 h = 1e-2,
                            float64 eps = 1e-8)
{
    return {gradient_y(m, b, y, x, h, eps), gradient_x(m, b, y, x, h, eps)};
}

template <typename T>
inline void resample(const Matrix<T> &mat_in, Matrix<T> &mat_out)
{
    if (mat_in.batch() != mat_out.batch())
    {
        throw_rte_with_backtrace("Batch dimensions do not match for resample, in: ", mat_in.shape,
                                 " out: ", mat_out.shape);
    }
    for (uint32 b = 0; b < mat_in.batch(); b++)
    {
        for (uint32 y = 0; y < mat_out.height(); y++)
        {
            float64 y_ = y / (float64)mat_out.height();
            for (uint32 x = 0; x < mat_out.width(); x++)
            {
                float64 x_ = x / (float64)mat_out.width();
                mat_out(y, x) = bilinear_sample(mat_in, y_, x_);
            }
        }
    }
}

template <typename T>
inline Matrix<T> resample(const Matrix<T> &mat_in, uint32 height, uint32 width)
{
    Matrix<T> mat_out(mat_in.batch(), height, width);
    resample(mat_in, mat_out);
    return mat_out;
}

#endif  // MATRIX_OPS_CUH