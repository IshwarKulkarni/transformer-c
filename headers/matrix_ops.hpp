
#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

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
                            value += A(k, y) * B(x, k);
                        }
                    if (C)
                        {
                            value += C->operator()(x, y);
                        }
                    result(x, y) = unary(value);
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
                    res(y, x) = A(x, y);
                }
        }
}

template <typename T, typename Op = Plus<T>, typename PostProcess = Identity<T>>
void reduceCPU(Matrix<T> &result, const Matrix<T> &A, T identity = Op::Identity,
               const Op &op = Op(), PostProcess unary = PostProcess())
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
            result(0, y) = unary(value);
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
                    value += A(x, y);
                }
            result(0, y) = value / A.width;
        }
}

template <typename T>
inline void fillCPU(Matrix<T> &A, T value)
{
    std::fill(A.begin(), A.end(), value);
}

template <typename T>
bool sameCPU(const Matrix<T> &A, const Matrix<T> &B, float32 eps = 1e-5)
{
    return std::equal(A.begin(), A.end(), B.begin(),
                      [eps](T a, T b) { return std::abs(a - b) < eps; });
}

template <typename Ta, typename Tb = Ta, typename Tr = Ta, typename Op>
inline void binary_applyCPU(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op &op)
{
    // a and b's dimensions should match result dimensions either on height or
    // width or have numels
    // 1
    if ((A.height != res.height && A.width != res.width && A.numels() != 1) ||
        (B.height != res.height && B.width != res.width && B.numels() != 1))
        {
            LOG(RED, "Matrix dimensions do not match for binary operation");
            throw std::runtime_error("Dimension mismatch");
        }

    // always broadcast either axis on either matrix
    for (uint32 y = 0; y < res.height; y++)
        {
            for (uint32 x = 0; x < res.width; x++)
                {
                    uint32 ax = A.width > 1 ? x : 0;
                    uint32 ay = A.height > 1 ? y : 0;
                    uint32 bx = B.width > 1 ? x : 0;
                    uint32 by = B.height > 1 ? y : 0;

                    res(x, y) = op(A(ax, ay), B(bx, by));
                }
        }
}

template <typename Ta, typename Tr, typename Op>
void unary_applyCPU(Matrix<Tr> &res, const Matrix<Ta> &A, Op op)
{
    for (uint32 y = 0; y < res.height; y++)
        {
            for (uint32 x = 0; x < res.width; x++)
                {
                    uint32 ax = A.width > 1 ? x : 0;
                    uint32 ay = A.height > 1 ? y : 0;
                    res(x, y) = op(A(ax, ay));
                }
        }
}

#endif
