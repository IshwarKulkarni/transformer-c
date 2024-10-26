#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <algorithm>
#include "functors.cuh"
#include "matrix_ops.cuh"

template <typename T, typename PostProcess = Identity<T>>
void mmaddCPU(Matrix<T> &result, const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> *C,
              PostProcess unary = PostProcess())
{
    check_mmadd_sizes(result, A, B, C);
    for (uint32 y = 0; y < A.height; y++)
    {
        for (uint32 x = 0; x < B.width; x++)
        {
            T value = 0;
            for (uint32 k = 0; k < A.width; k++)
            {
                value += A(y, k) * B(k, x);
            }
            if (C)
            {
                value += (*C)(y, x);
            }
            result(y, x) = unary(value);
        }
    }
}

template <typename T>
void transposeCPU(Matrix<T> &res, const Matrix<T> &A)
{
    for (uint32 y = 0; y < A.height; y++)
    {
        for (uint32 x = 0; x < A.width; x++)
        {
            res(x, y) = A(y, x);
        }
    }
}

template <typename T, typename Reduction = Plus<T>, typename PostProcess = Identity<T>>
void reduceCPU(Matrix<T> &result, const Matrix<T> &A, const Reduction &op = Reduction(),
               T identity = Reduction::Identity, PostProcess pProcess = PostProcess())
{
    if (result.height != A.height || result.width != 1)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for reduce operation");
        throw runtime_error_with_backtrace("Dimension mismatch");
    }
    for (uint32 y = 0; y < A.height; y++)
    {
        T value = identity;
        for (uint32 x = 0; x < A.width; x++)
        {
            value = op(value, A(y, x));
        }
        result(y, 0) = pProcess(value);
    }
}

template <typename T>
void reduce_meanCPU(Matrix<T> &result, const Matrix<T> &A)
{
    for (uint32 y = 0; y < A.height; y++)
    {
        T value = 0;
        for (uint32 x = 0; x < A.width; x++)
        {
            value += A(y, x);
        }
        result(y, 0) = value / A.width;
    }
}

template <typename T>
inline void fillCPU(Matrix<T> &A, T value)
{
    std::fill(A.begin(), A.end(), value);
}

template <typename T>
inline void fillCPU(Matrix<T> &A, const T *values)
{
    std::copy_n(values, A.numels(), A.begin());
}

template <typename T>
bool sameCPU(const Matrix<T> &A, const T *B, float32 eps = 1e-5)
{
    return std::equal(A.begin(), A.end(), B, [eps](T a, T b) { return std::abs(a - b) < eps; });
}

template <typename T>
bool sameCPU(const Matrix<T> &A, const Matrix<T> &B, float32 eps = 1e-5)
{
    if (A.height != B.height || A.width != B.width)
    {
        return false;
    }
    return std::equal(A.begin(), A.end(), B.begin(),
                      [eps](T a, T b) { return std::abs(a - b) < eps; });
}

template <typename Ta, typename Tb = Ta, typename Tr = Ta, typename Reduction>
inline void binary_applyCPU(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B,
                            Reduction &op)
{
    // a and b's dimensions should match result dimensions either on height or
    // width or have numels
    // 1
    if ((A.height != res.height && A.width != res.width && A.numels() != 1) ||
        (B.height != res.height && B.width != res.width && B.numels() != 1))
    {
        LOG(RED, "Matrix dimensions do not match for binary operation");
        throw runtime_error_with_backtrace("Dimension mismatch");
    }

    // always broadcast either axis on either matrix
    for (uint32 y = 0; y < res.height; y++)
    {
        for (uint32 x = 0; x < res.width; x++)
        {
            uint32 ax = A.width > 1 ? x : 0;
            uint32 ay = A.height > 1 ? y : 0;
            uint32 by = B.height > 1 ? y : 0;

            res(x, y) = op(A(ay, ax), B(by, by));
        }
    }
}

template <typename Ta, typename Tr, typename Reduction>
void unary_applyCPU(Matrix<Tr> &res, const Matrix<Ta> &A, Reduction op)
{
    for (uint32 y = 0; y < res.height; y++)
    {
        for (uint32 x = 0; x < res.width; x++)
        {
            uint32 ax = A.width > 1 ? x : 0;
            uint32 ay = A.height > 1 ? y : 0;
            res(y, x) = op(A(ay, ax));
        }
    }
}

#endif
