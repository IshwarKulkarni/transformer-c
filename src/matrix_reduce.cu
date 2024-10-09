#include "../headers/matrix_ops.cuh"

template <typename Ta, typename Op, uint32 WIDTH>
__global__ void reduce_kernel(const Ta *A, Ta *result, Op op, uint32 height, uint32 width,
                              uint32 offset = 0, bool reducExisting = false, Ta identity = Ta(0))
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;
    __shared__ Ta As[WIDTH];

    uint32 offset_x = x + offset;

    As[threadIdx.x] = (offset_x < width) ? Ta(A[offset_x + y * width]) : identity;

    __syncthreads();

    Ta existing = (reducExisting ? result[y] : identity);

#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < s) As[x] = op(As[x], As[x + s]);
        __syncthreads();
    }
    volatile Ta *vAs = (volatile Ta *)As;
    if (x <= 32)
    {
        if (WIDTH >= 64) vAs[x] = op(vAs[x], vAs[x + 32]);
        if (WIDTH >= 32) vAs[x] = op(vAs[x], vAs[x + 16]);
        if (WIDTH >= 16) vAs[x] = op(vAs[x], vAs[x + 8]);
        if (WIDTH >= 8) vAs[x] = op(vAs[x], vAs[x + 4]);
        if (WIDTH >= 4) vAs[x] = op(vAs[x], vAs[x + 2]);
        if (WIDTH >= 2) vAs[x] = op(vAs[x], vAs[x + 1]);
    }
    __syncthreads();

    if (x == 0 and blockIdx.x < height)
    {
        result[y] = op(vAs[0], existing);
    }
}

template <typename T, typename Op>
void reduce(Matrix<T> &result, const Matrix<T> &A, const Op &op, T identity)
{
    if (A.width <= 32)
    {
        reduce_kernel<T, Op, 32><<<A.height, 32>>>(A.begin(), result.begin(), op, A.height, A.width,
                                                   0, false, identity);
    }
    else if (A.width <= 64)
    {
        reduce_kernel<T, Op, 64><<<A.height, 64>>>(A.begin(), result.begin(), op, A.height, A.width,
                                                   0, false, identity);
    }
    else if (A.width <= 128)
    {
        reduce_kernel<T, Op, 128><<<A.height, 128>>>(A.begin(), result.begin(), op, A.height,
                                                     A.width, 0, false, identity);
    }
    else if (A.width <= 256)
    {
        reduce_kernel<T, Op, 256><<<A.height, 256>>>(A.begin(), result.begin(), op, A.height,
                                                     A.width, 0, false, identity);
    }
    else if (A.width <= 512)
    {
        reduce_kernel<T, Op, 512><<<A.height, 512>>>(A.begin(), result.begin(), op, A.height,
                                                     A.width, 0, false, identity);
    }
    else
    {
        reduce_kernel<T, Op, 1024><<<A.height, 1024>>>(A.begin(), result.begin(), op, A.height,
                                                       A.width, 0, false, identity);
        constexpr uint32 WIDTH = 1024;
        for (uint offset = WIDTH; offset < A.width; offset += WIDTH)
        {
            reduce_kernel<T, Op, WIDTH><<<A.height, WIDTH>>>(
                A.begin(), result.begin(), op, A.height, A.width, offset, offset > 0, identity);
        }
    }
}

using FloatT = float32;

template void reduce(Matrix<FloatT> &, const Matrix<FloatT> &, const Plus<FloatT> &, FloatT);

template void reduce(Matrix<FloatT> &, const Matrix<FloatT> &, const Min<FloatT> &, FloatT);

template void reduce(Matrix<FloatT> &, const Matrix<FloatT> &, const Max<FloatT> &, FloatT);
