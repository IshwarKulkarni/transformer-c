#include "../headers/matrix_ops.cuh"

template <typename Ta, typename ReduceOp, uint32 BLOCK_X, typename PostProcess>
__global__ void reduce_kernel(const Ta *A, Ta *result, ReduceOp reduceOp, uint32 height,
                              uint32 width, uint32 offset = 0, bool reducExisting = false,
                              Ta identity = Ta(0), PostProcess postProcess = PostProcess())
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;
    __shared__ Ta As[BLOCK_X + 1];

    uint32 offset_x = x + offset;

    As[threadIdx.x] = (offset_x < width) ? Ta(A[offset_x + y * width]) : identity;

    Ta existing = (reducExisting ? result[y] : identity);

#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < width and x < s and x) As[x] = reduceOp(As[x], As[x + s]);
        __syncthreads();
    }
    __syncthreads();

    volatile Ta *vAs = (volatile Ta *)As;
    if (x <= 32 and offset_x < width)
    {
        if (BLOCK_X >= 64) vAs[x] = reduceOp(vAs[x], vAs[x + 32]);
        if (BLOCK_X >= 32) vAs[x] = reduceOp(vAs[x], vAs[x + 16]);
        if (BLOCK_X >= 16) vAs[x] = reduceOp(vAs[x], vAs[x + 8]);
        if (BLOCK_X >= 8) vAs[x] = reduceOp(vAs[x], vAs[x + 4]);
        if (BLOCK_X >= 4) vAs[x] = reduceOp(vAs[x], vAs[x + 2]);
        if (BLOCK_X >= 2) vAs[x] = reduceOp(vAs[x], vAs[x + 1]);
    }
    __syncthreads();

    if (x == 0)
    {
        result[y] = postProcess(reduceOp(vAs[0], existing));
    }
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduceOp, T identity,
            PostProcess postProcess)
{
    if (A.width < 32)
    {
        reduce_kernel<T, ReduceOp, 32><<<A.height, 32>>>(A.begin(), result.begin(), reduceOp,
                                                         A.height, A.width, 0, false, identity,
                                                         postProcess);
    }
    else if (A.width < 64)
    {
        reduce_kernel<T, ReduceOp, 64><<<A.height, 64>>>(A.begin(), result.begin(), reduceOp,
                                                         A.height, A.width, 0, false, identity,
                                                         postProcess);
    }
    else if (A.width < 128)
    {
        reduce_kernel<T, ReduceOp, 128><<<A.height, 128>>>(A.begin(), result.begin(), reduceOp,
                                                           A.height, A.width, 0, false, identity,
                                                           postProcess);
    }
    else if (A.width < 256)
    {
        reduce_kernel<T, ReduceOp, 256><<<A.height, 256>>>(A.begin(), result.begin(), reduceOp,
                                                           A.height, A.width, 0, false, identity,
                                                           postProcess);
    }
    else if (A.width < 512)
    {
        reduce_kernel<T, ReduceOp, 512><<<A.height, 512>>>(A.begin(), result.begin(), reduceOp,
                                                           A.height, A.width, 0, false, identity,
                                                           postProcess);
    }
    else if (A.width < 1024)
    {
        reduce_kernel<T, ReduceOp, 1024><<<A.height, 1024>>>(A.begin(), result.begin(), reduceOp,
                                                             A.height, A.width, 0, false, identity,
                                                             postProcess);
    }
    else
    {
        constexpr int32 BLOCK_X = 256;
        int32 offset = 0;
        for (; offset < A.width - BLOCK_X; offset += BLOCK_X)
        {
            reduce_kernel<T, ReduceOp, BLOCK_X, Identity<T>>
                <<<A.height, BLOCK_X>>>(A.begin(), result.begin(), reduceOp, A.height, A.width,
                                      offset, offset > 0, identity);
        }
        reduce_kernel<T, ReduceOp, BLOCK_X><<<A.height, BLOCK_X>>>(A.begin(), result.begin(), reduceOp,
                                                               A.height, A.width, offset, true,
                                                               identity, postProcess);
    }
}

using FloatT = float32;
template void reduce<FloatT, Plus<FloatT, FloatT>, DividebBy<FloatT>>(Matrix<FloatT> &,
                                                                      Matrix<FloatT> const &,
                                                                      Plus<FloatT, FloatT>, FloatT,
                                                                      DividebBy<FloatT>);
template void reduce<FloatT, Plus<FloatT, FloatT>, Identity<FloatT>>(Matrix<FloatT> &,
                                                                     Matrix<FloatT> const &,
                                                                     Plus<FloatT, FloatT>, FloatT,
                                                                     Identity<FloatT>);
template void reduce<FloatT, Max<FloatT>, Identity<FloatT>>(Matrix<FloatT> &,
                                                            Matrix<FloatT> const &, Max<FloatT>,
                                                            FloatT, Identity<FloatT>);
template void reduce<FloatT, Min<FloatT>, Identity<FloatT>>(Matrix<FloatT> &,
                                                            Matrix<FloatT> const &, Min<FloatT>,
                                                            FloatT, Identity<FloatT>);

template void reduce<FloatT, Plus<FloatT, FloatT>, Sigmoid<FloatT>::SigmoidF>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Plus<FloatT, FloatT>, FloatT,
    Sigmoid<FloatT>::SigmoidF);

template void reduce<FloatT, Plus<FloatT, FloatT>, Exp<FloatT>>(Matrix<FloatT> &,
                                                                Matrix<FloatT> const &,
                                                                Plus<FloatT, FloatT>, FloatT,
                                                                Exp<FloatT>);