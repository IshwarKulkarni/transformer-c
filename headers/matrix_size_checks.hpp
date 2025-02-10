#ifndef MATRIX_SIZE_CHECK
#define MATRIX_SIZE_CHECK

#include "matrix.cuh"
#include "types"

//////////////////////////////////////////////////////////////////////////////////////
// Size checks
//////////////////////////////////////////////////////////////////////////////////////

template <uint32 dim>
inline bool broadcastable(const Shape &A, const Shape &Res)
{
    return A[dim] == Res[dim] or A[dim] == 1;
}

template <uint32 dim, typename T>
inline bool broadcastable(const Matrix<T> &A, const Matrix<T> &Res)
{
    return broadcastable<dim>(A.shape, Res.shape);
}

template <uint32 dim, typename T>
inline bool broadcastable(const Optional<Matrix<T>> &A, const Matrix<T> &Res)
{
    return A.is_valid() ? broadcastable<dim>(A->shape, Res.shape) : true;
}

inline bool mmABRes(const Shape &A, const Shape &B, const Shape &Res)
{
    return A.width == B.height and A.height == Res.height and B.width == Res.width and
           broadcastable<2>(A, Res) and broadcastable<2>(B, Res);
}

#ifndef DISABLE_SIZE_CHECK

template <typename Tr, typename T, typename Tb, typename Tc>
void check_mmadd_sizes(Matrix<Tr> &result, const Matrix<T> &A, const Matrix<Tb> &B,
                       const Optional<Matrix<Tc>> C)
{
    if (mmABRes(A.shape, B.shape, result.shape) and broadcastable<2>(C, result)) return;

    if (C.is_valid() and C->shape != result.shape)
    {
        throw_rte_with_backtrace(RED, "Dimension mismatch in mmadd: A: ", A.shape,
                                 " * B: ", B.shape, " -> ", result.shape, " & C: ", C->shape);
    }
    throw_rte_with_backtrace(RED, "Dimension mismatch in mmadd: A: ", A.shape, " * B: ", B.shape,
                             " -> ", result.shape);
}

// check if A and B are compatible for mmTadd operation in height and width dimension,
// and if C is valid, check if it is compatible with result,
// and if batch dimensions match or are broadcastable
template <typename Tr, typename T, typename Tb, typename Tc>
void check_mmTadd_sizes(Matrix<Tr> &result, const Matrix<T> &A, const Matrix<Tb> &B,
                        const Optional<Matrix<Tc>> C)
{
    if (mmABRes(A.shape, B.shape.t(), result.shape) and broadcastable<2>(C, result) and
        broadcastable<1>(C, result) and broadcastable<0>(C, result))
        return;

    if (C.is_valid() and C->shape != result.shape)
    {
        throw_rte_with_backtrace(RED, "Dimension mismatch in mmTadd: A: ", A.shape,
                                 " * B: ", B.shape, " -> ", result.shape, " & C: ", C->shape);
    }
    throw_rte_with_backtrace(RED, "Dimension mismatch in mmTadd: A: ", A.shape, " * B: ", B.shape,
                             " -> ", result.shape);
}

template <typename T>  // used in ternary_apply
void check_broadcast_sizes(const Matrix<T> &res, const Matrix<T> &A, const Matrix<T> &B,
                           const Matrix<T> &C)
{
    if (broadcastable<0>(A, res) and broadcastable<1>(B, res) and broadcastable<1>(A, res) and
        broadcastable<0>(B, res) and broadcastable<2>(A, res) and broadcastable<2>(B, res) and
        broadcastable<0>(C, res) and broadcastable<2>(C, res) and broadcastable<2>(C, res))
        return;

    throw_rte_with_backtrace(RED, "Dimension mismatch in binary_apply: A: ", A.shape,
                             " & B: ", B.shape, " -> ", res.shape);
}

template <typename T>  // used in binary_apply
void check_broadcast_sizes(const Matrix<T> &res, const Matrix<T> &A, const Matrix<T> &B)
{
    if (broadcastable<0>(A, res) and broadcastable<1>(B, res) and broadcastable<1>(A, res) and
        broadcastable<0>(B, res) and broadcastable<2>(A, res) and broadcastable<2>(B, res))
        return;

    throw_rte_with_backtrace(RED, "Dimension mismatch in binary_apply: A: ", A.shape,
                             " & B: ", B.shape, " -> ", res.shape);
}

template <typename T>  // used in binary_apply
void check_broadcast_sizes(const Matrix<T> &res, const Matrix<T> &A)
{
    if (broadcastable<0>(A, res) and broadcastable<1>(A, res) and broadcastable<2>(A, res)) return;

    throw_rte_with_backtrace(RED, "Dimension mismatch in unary_apply: A: ", A.shape, " -> ",
                             res.shape);
}

template <typename T>
inline void check_softmax_grad_sizes(const Matrix<T> &s_grad_out, const Matrix<T> &s_out,
                                     const Matrix<T> &grad_in)
{
    if (s_grad_out.batch() != s_out.batch() or s_grad_out.batch() != grad_in.batch())
    {
        throw_rte_with_backtrace(
            "Batch dimensions do not match for softmax gradient, s_grad_out: ", s_grad_out.shape,
            ", s_out: ", s_out.shape, " & grad_in: ", grad_in.shape);
    }

    auto size_or_tx_match = [&s_grad_out](uint32 h, uint32 w) {
        if (h == s_grad_out.height() and w == s_grad_out.width()) return true;
        return (w == s_grad_out.height() and h == s_grad_out.width());
    };
    if (!size_or_tx_match(s_out.height(), s_out.width()) or
        !size_or_tx_match(grad_in.height(), grad_in.width()))
    {
        LOG(RED, "Dimension mismatch in softmax gradient: s_grad_out: ", s_grad_out.shape,
            ", s_out: ", s_out.shape, " & grad_in: ", grad_in.shape);
        throw_rte_with_backtrace("Dimension mismatch");
    }
}

template <typename T, uint32 dim>
void check_reduction_sizes(const Matrix<T> &result, const Matrix<T> &A)
{
    if (result.id == A.id) return;  // in-place reduction
    for (uint32 i = 0; i < 3; i++)
    {
        if ((i == dim and result.shape[i] != 1) or (i != dim and result.shape[i] != A.shape[i]))
        {
            throw_rte_with_backtrace(RED, "Dimension mismatch for reduction in dim ", dim,
                                     " with A: ", A.shape, " & Result: ", result.shape);
        }
    }
}

#else

template <typename Tr, typename Ta, typename Tb, typename Tc>
void check_mmadd_sizes(Matrix<Tr> &, const Matrix<Ta> &, const Matrix<Tb> &,
                       const Optional<Matrix<Tc>>)
{
}

template <typename Tr, typename Ta, typename Tb, typename Tc>
void check_mmTadd_sizes(Matrix<Tr> &, const Matrix<Ta> &, const Matrix<Tb> &,
                        const Optional<Matrix<Tc>>)
{
}

template <typename T>  // used in ternary_apply
void check_broadcast_sizes(const Matrix<T> &res, const Matrix<T> &A, const Matrix<T> &B)
{
}

template <typename T>  // used in binary_apply
void check_broadcast_sizes(const Matrix<T> &, const Matrix<T> &, const Matrix<T> &)
{
}

template <typename T>  // used in binary_apply
void check_broadcast_sizes(const Matrix<T> &, const Matrix<T> &)
{
}

template <typename T>
inline void check_softmax_grad_sizes(const Matrix<T> &, const Matrix<T> &, const Matrix<T> &)
{
}

template <typename T, uint32 dim>
void check_reduction_sizes(const Matrix<T> &, const Matrix<T> &)
{
}
#endif

#endif
