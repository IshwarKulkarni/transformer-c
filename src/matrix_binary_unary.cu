#include <cuda_device_runtime_api.h>
#include "../headers/matrix_ops.cuh"

template <typename Ta, typename Tb, typename Tr, typename Op>
__global__ void binary_apply_kernel(Tr *__restrict__ result, const Ta *__restrict__ A,
                                    const Tb *__restrict__ B, uint32 resH, uint32 resW, uint32 aH,
                                    uint32 aW, uint32 bH, uint32 bW, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resW || y >= resH) return;

    uint32 Axy[2] = {x, y};
    uint32 Bxy[2] = {x, y};

    if (aW == 1) Axy[0] = 0;  // broadcast along x axis
    if (aH == 1) Axy[1] = 0;
    if (bW == 1) Bxy[0] = 0;
    if (bH == 1) Bxy[1] = 0;

    result[y * resW + x] = op(A[Axy[0] + aW * Axy[1]], B[Bxy[0] + bW * Bxy[1]]);
}

template <typename Ta, typename Tb, typename Tr, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op)
{
    check_broadcast_sizes<Ta>(res, A, B);

    dim3 block(32, 32);
    dim3 grid((res.width + block.x - 1) / block.x, (res.height + block.y - 1) / block.y);
    binary_apply_kernel<Ta, Tb, Tr, Op><<<grid, block>>>(res.begin(), A.begin(), B.begin(),
                                                         res.height, res.width, A.height, A.width,
                                                         B.height, B.width, op);
}

template <typename Ta, typename Tr, typename Op>
__global__ void unary_apply_kernel(Tr *__restrict__ result, const Ta *__restrict__ A, uint32 resH,
                                   uint32 resW, uint32 aH, uint32 aW, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= resW || y >= resH) return;

    uint32 Axy[2] = {x, y};

    if (aW == 1) Axy[0] = 0;  // broadcast along x axis
    if (aH == 1) Axy[1] = 0;

    result[y * resW + x] = op(A[Axy[0] + aW * Axy[1]]);
}

template <typename Ta, typename Tr, typename Op>
void unary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, Op op)
{
    dim3 block(32, 32);
    dim3 grid((res.width + block.x - 1) / block.x, (res.height + block.y - 1) / block.y);
    unary_apply_kernel<Ta, Tr, Op>
        <<<grid, block>>>(res.begin(), A.begin(), res.height, res.width, A.height, A.width, op);
}

using FloatT = float32;
template void binary_apply<FloatT, FloatT, FloatT, Plus<FloatT, FloatT>>(Matrix<FloatT> &,
                                                                         Matrix<FloatT> const &,
                                                                         Matrix<FloatT> const &,
                                                                         Plus<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, Neg<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                       Neg<FloatT>);

template void
binary_apply<FloatT, FloatT, FloatT, Composition<FloatT, Sub<FloatT, FloatT>, Square<FloatT>>>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Matrix<FloatT> const &,
    Composition<FloatT, Sub<FloatT, FloatT>, Square<FloatT>>);

template void binary_apply<FloatT, FloatT, FloatT,
                           Composition<FloatT, Sub<FloatT, FloatT>, IntegerMultiplier<FloatT, -2>>>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Matrix<FloatT> const &,
    Composition<FloatT, Sub<FloatT, FloatT>, IntegerMultiplier<FloatT, -2>>);

template void binary_apply<FloatT, FloatT, FloatT, WeightUpdate<FloatT, FloatT>>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Matrix<FloatT> const &, WeightUpdate<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, Sigmoid<FloatT>::SigmoidB>(Matrix<FloatT> &,
                                                                     Matrix<FloatT> const &,
                                                                     Sigmoid<FloatT>::SigmoidB);
template void binary_apply<FloatT, FloatT, FloatT, Mul<FloatT, FloatT>>(Matrix<FloatT> &,
                                                                        Matrix<FloatT> const &,
                                                                        Matrix<FloatT> const &,
                                                                        Mul<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, DividebBy<FloatT>>(Matrix<FloatT> &,
                                                             Matrix<FloatT> const &,
                                                             DividebBy<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, Sub<FloatT, FloatT>>(Matrix<FloatT> &,
                                                                        Matrix<FloatT> const &,
                                                                        Matrix<FloatT> const &,
                                                                        Sub<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, Square<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                          Square<FloatT>);

template void unary_apply<FloatT, FloatT, MultiplyBy<FloatT>>(Matrix<FloatT> &,
                                                              Matrix<FloatT> const &,
                                                              MultiplyBy<FloatT>);

template void unary_apply<FloatT, FloatT, Relu<FloatT>::ReluB>(Matrix<FloatT> &,
                                                               Matrix<FloatT> const &,
                                                               Relu<FloatT>::ReluB);

template void unary_apply<FloatT, FloatT, TanH<FloatT>::TanhB>(Matrix<FloatT> &,
                                                               Matrix<FloatT> const &,
                                                               TanH<FloatT>::TanhB);

template void unary_apply<FloatT, FloatT, Abs<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                       Abs<FloatT>);
