#include "../headers/matrix_ops.cuh"
#include <vector_types.h>

template <typename T, uint32 BLOCK_Y, uint32 BLOCK_X, typename ReduceOp, typename PostProcess>
__global__ void reduce_kernel_small(const T *A, T *result, ReduceOp reduceOp, uint32 height,
                                    uint32 width, T identity = T(0),
                                    PostProcess postProcess = PostProcess())
{
    uint32 x = threadIdx.x;
    uint32 y = threadIdx.y;
    __shared__ T As[BLOCK_Y + 1][BLOCK_X + 1];

    if (x < width and y < height)
        As[y][x] = T(A[x + y * width]);
    else
        As[y][x] = identity;
    __syncthreads();
    for (uint32 s = 16; s > 0; s >>= 1)
    {
        if (x < s and x < width) As[y][x] = reduceOp(As[y][x], As[y][x + s]);
        __syncthreads();
    }

    if (x == 0 and y < height)
    {
        result[y] = postProcess(As[y][0]);
    }
}

// it's best to read this kernel as reducing along the width of the matrix
template <typename T, typename ReduceOp, uint32 BLOCK_X, typename PostProcess>
__global__ void reduce_kernel(const T *A, T *result, ReduceOp reduceOp, uint32 height, uint32 width,
                              uint32 offset = 0, bool reducExisting = false, T identity = T(0),
                              PostProcess postProcess = PostProcess())
{
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x; // always < height
    __shared__ T As[BLOCK_X + 1];

    uint32 offset_x = x + offset;

    As[x] = (offset_x < width) ? T(A[offset_x + y * width]) : identity;

    T existing = (reducExisting ? result[y] : identity);

#pragma unroll
    for (uint32 s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (x < width and x < s and x) As[x] = reduceOp(As[x], As[x + s]);
        __syncthreads();
    }
    __syncthreads();

    volatile T *vAs = (volatile T *)As;
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

    if (x == 0 and y < height)
    {
        result[y] = postProcess(reduceOp(vAs[0], existing));
    }
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce_kernel_launcher(const T *A, T *result, ReduceOp op, uint32 n_reductions, uint32 aspan,
                            uint32 n_outputs, uint32 offset = 0, bool reducExisting = false,
                            T identity = T(0), PostProcess pProcess = PostProcess())
{
    if (n_outputs != n_reductions)
    {
        throw std::runtime_error("Number of outputs must be equal to number of reductions");
    }
    if (aspan <= 32)
    {
        if (n_reductions < 24) // small matrix special case
        {
            dim3 blockSize(aspan, n_reductions);
            reduce_kernel_small<T, 24, 32, ReduceOp, PostProcess>
                <<<1, blockSize>>>(A, result, op, n_reductions, aspan, identity, pProcess);
        }
        else
        {
            reduce_kernel<T, ReduceOp, 32><<<n_outputs, 32>>>(A, result, op, n_reductions, aspan, 0,
                                                              false, identity, pProcess);
        }
    }
    else if (aspan <= 64)
    {
        reduce_kernel<T, ReduceOp, 64>
            <<<n_outputs, 64>>>(A, result, op, n_reductions, aspan, 0, false, identity, pProcess);
    }
    else if (aspan <= 128)
    {
        reduce_kernel<T, ReduceOp, 128>
            <<<n_outputs, 128>>>(A, result, op, n_reductions, aspan, 0, false, identity, pProcess);
    }
    else if (aspan <= 256)
    {
        reduce_kernel<T, ReduceOp, 256>
            <<<n_outputs, 256>>>(A, result, op, n_reductions, aspan, 0, false, identity, pProcess);
    }
    else if (aspan <= 512)
    {
        reduce_kernel<T, ReduceOp, 512>
            <<<n_outputs, 512>>>(A, result, op, n_reductions, aspan, 0, false, identity, pProcess);
    }
    else if (aspan <= 1024)
    {
        reduce_kernel<T, ReduceOp, 1024>
            <<<n_outputs, 1024>>>(A, result, op, n_reductions, aspan, 0, false, identity, pProcess);
    }
    else
    {
        constexpr int32 BLOCK_X = 256;
        int32 offset = 0;
        for (; offset < aspan - BLOCK_X; offset += BLOCK_X)
        {
            reduce_kernel<T, ReduceOp, BLOCK_X, Identity<T>><<<n_outputs, BLOCK_X>>>(
                A, result, op, n_reductions, aspan, offset, offset > 0, identity);
        }
        reduce_kernel<T, ReduceOp, BLOCK_X><<<n_outputs, BLOCK_X>>>(
            A, result, op, n_reductions, aspan, offset, true, identity, pProcess);
    }
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce_column_vec(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduceOp, T identity,
                       PostProcess postProcess)
{
    if (!(A.width == 1 and result.width == 1 and result.height == 1))
    {
        if (A.height == 1)
        {
            LOG("No reduction needed, copying");
            fill(result, A.begin());
            return;
        }
        throw std::runtime_error("Invalid dimensions for column-reduction " + A.shape_string() +
                                 " to " + result.shape_string());
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, 1, A.height, 1, 0, false, identity,
                           postProcess);
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduceOp, T identity,
            PostProcess postProcess)
{

    if (A.width == 1 and result.width == 1)
    {
        if (result.height == A.height)
        {
            LOG("No reduction needed, copying");
            fill(result, A.begin());
            return;
        }
        if (result.height == 1)
        {
            LOG("For reducing to scalar from a column vector, use reduce_column_vec function");
        }
        throw std::runtime_error("Invalid dimensions for row-reduction " + A.shape_string() +
                                 " to " + result.shape_string());
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, A.height, A.width, result.height, 0,
                           false, identity, postProcess);
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

template void reduce_column_vec<FloatT, Plus<FloatT, FloatT>, DividebBy<FloatT>>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Plus<FloatT, FloatT>, FloatT, DividebBy<FloatT>);