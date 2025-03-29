#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include <algorithm>
#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.hpp"
#include "matrix_size_checks.hpp"

template <typename T, typename PostProcess = Identity<T>>
void mmaddCPU(Matrix<T>& result, const Matrix<T>& A, const Matrix<T>& B,
              const Optional<Matrix<T>> C = {}, PostProcess unary = PostProcess())
{
    check_mmadd_sizes(result, A, B, C);
    for (uint32 b = 0; b < A.batch(); b++)
        for (uint32 y = 0; y < A.height(); y++)
        {
            for (uint32 x = 0; x < B.width(); x++)
            {
                T value = 0;
                for (uint32 k = 0; k < A.width(); k++)
                {
                    value += A.template broadcasting_fetch<BATCH_BIT>(b, y, k) *
                             B.template broadcasting_fetch<BATCH_BIT>(b, k, x);
                }
                if (C)
                {
                    value += (*C)(b, (C->height() == 1) ? 0 : y, (C->width() == 1) ? 0 : x);
                }
                result(b, y, x) = unary(value);
            }
        }
}

// result = A * B^T + C
template <typename T, typename PostProcess = Identity<T>>
void mmTaddCPU(Matrix<T>& result, const Matrix<T>& A, const Matrix<T>& B,
               const Optional<Matrix<T>> C = {}, PostProcess unary = PostProcess())
{
    check_mmTadd_sizes(result, A, B, C);
    for (uint32 b = 0; b < A.batch(); b++)
        for (uint32 y = 0; y < A.height(); y++)
        {
            for (uint32 x = 0; x < B.height(); x++)
            {
                T value = 0;
                for (uint32 k = 0; k < A.width(); k++)
                {
                    value += A.template broadcasting_fetch<0b100>(b, y, k) *
                             B.template broadcasting_fetch<0b100>(b, x, k);
                }
                if (C)
                {
                    value += (*C)(b, y, x);
                }
                result(b, y, x) = unary(value);
            }
        }
}

template <typename T, typename PostProcess = Identity<T>>
void transposeCPU(Matrix<T>& res, const Matrix<T>& A, PostProcess unary = PostProcess())
{
    if (res.shape.t() != A.shape or A.batch() != res.batch())
        throw_rte_with_backtrace("Shapes don't match for transpose:", A.shape, "->", res.shape);
    for (uint32 b = 0; b < res.batch(); b++)
    {
        for (uint32 y = 0; y < A.height(); y++)
        {
            for (uint32 x = 0; x < A.width(); x++)
            {
                res(b, x, y) = unary(A(b, y, x));
            }
        }
    }
}

template <typename T = FloatT, uint32 dim = 0, typename Reduction = Plus<T>,
          typename PostProcess = Identity<T>>
void reduceCPU(Matrix<T>& result, const Matrix<T>& A, const Reduction& op = Reduction(),
               T identity = Reduction::Identity, PostProcess pProcess = PostProcess())
{
    check_reduction_sizes<T, dim>(result, A);
    uint32 l0 = A.shape[dim], l1, l2;  // l0 is reduction dimension
    if (dim == 0)
    {
        l1 = A.batch();
        l2 = A.height();
    }
    if (dim == 1)
    {
        l1 = A.batch();
        l2 = A.width();
    }
    if (dim == 2)
    {
        l1 = A.height();
        l2 = A.width();
    }

    for (uint32 i2 = 0; i2 < l2; i2++)
        for (uint32 i1 = 0; i1 < l1; i1++)
        {
            T reduced = identity;
            for (uint32 i0 = 0; i0 < l0; i0++)
            {
                reduced = op(reduced, A.template index<dim>(i0, i1, i2));
            }
            result.template index<dim>(0, i1, i2) = pProcess(reduced);
        }
}

template <typename T, uint32 dim = 0>
void reduce_meanCPU(Matrix<T>& result, const Matrix<T>& A)
{
    check_reduction_sizes<T, dim>(result, A);
    reduceCPU<T, dim>(result, A, Plus<T>(), T(0), DividedBy<T>(A.shape[dim]));
}

template <typename T, typename Tb = T, typename Tr = T, typename Reduction>
inline void binary_applyCPU(Matrix<Tr>& res, const Matrix<T>& A, const Matrix<Tb>& B,
                            const Reduction& op)
{
    // a and b's dimensions should match result dimensions either on height or
    // width or have numels
    // 1
    if ((A.height() != res.height() && A.width() != res.width() && A.numels() != 1) ||
        (B.height() != res.height() && B.width() != res.width() && B.numels() != 1))
    {
        throw_rte_with_backtrace("Dimension mismatch, A: ", A.shape, " B: ", B.shape,
                                 " Res: ", res.shape);
    }

    // always broadcast either axis on either matrix
    for (uint32 b = 0; b < res.batch(); b++)
        for (uint32 y = 0; y < res.height(); y++)
            for (uint32 x = 0; x < res.width(); x++)
            {
                auto a_ = A.template broadcasting_fetch<7>(b, y, x);
                auto b_ = B.template broadcasting_fetch<7>(b, y, x);
                res(b, y, x) = op(a_, b_);
            }
}

template <typename T, typename Tr, typename Reduction>
void unary_applyCPU(Matrix<Tr>& res, const Matrix<T>& A, Reduction op)
{
    for (uint32 b = 0; b < res.batch(); b++)
        for (uint32 y = 0; y < res.height(); y++)
            for (uint32 x = 0; x < res.width(); x++)
            {
                uint32 ax = A.width() > 1 ? x : 0;
                uint32 ay = A.height() > 1 ? y : 0;
                res(b, y, x) = op(A(b, ay, ax));
            }
}

template <typename T>
float64 sum_absCPU(const Matrix<T>& A)
{
    float64 sum = 0;
    for (uint32 b = 0; b < A.batch(); b++)
        for (uint32 y = 0; y < A.height(); y++)
        {
            for (uint32 x = 0; x < A.width(); x++)
            {
                sum += std::abs(A(b, y, x));
            }
        }
    return sum;
}

template <typename T>
float64 sum_squaredCPU(const Matrix<T>& A)
{
    float64 sum = 0;
    for (uint32 b = 0; b < A.batch(); b++)
        for (uint32 y = 0; y < A.height(); y++)
        {
            for (uint32 x = 0; x < A.width(); x++)
            {
                sum += A(y, x) * A(b, y, x);
            }
        }
    return sum;
}

template <typename T>
float64 sumCPU(const Matrix<T>& A)
{
    float64 sum = 0;
    for (uint32 b = 0; b < A.batch(); b++)
        for (uint32 y = 0; y < A.height(); y++)
        {
            for (uint32 x = 0; x < A.width(); x++)
            {
                sum += A(b, y, x);
            }
        }
    return sum;
}

#endif
