#include <cuda_device_runtime_api.h>
#include <vector_types.h>
#include "../headers/matrix_ops.cuh"

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
        if (x < s and x < width) As[y][x] = reduceOp(y, x, As[y][x], As[y][x + s]);
        __syncthreads();
    }

    if (x == 0 and y < height)
    {
        result[y] = postProcess(y, x, As[y][0]);
    }
}

// it's best to read this kernel as reducing along the width of the matrix
template <typename T, typename ReduceOp, uint32 BLOCK_X, typename PostProcess>
__global__ void reduce_kernel(const T *A, T *result, ReduceOp reduceOp, uint32 height, uint32 width,
                              bool reducExisting = false, T identity = T(0),
                              PostProcess postProcess = PostProcess())
{
    __shared__ T sdata[BLOCK_X + 1];
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;

    A += y * width;  // this thread block only deals with one row.

    sdata[x] = x < width ? A[x] : identity;
    for (uint32 otherIdx = x + blockDim.x; otherIdx < width; otherIdx += blockDim.x)
    {
        sdata[x] = reduceOp(y, x, sdata[x], A[otherIdx]);
    }

    __syncthreads();

    if (BLOCK_X >= 1024)
    {
        if (x < 512)
        {
            sdata[x] = reduceOp(y, x, sdata[x], sdata[x + 512]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 512)
    {
        if (x < 256)
        {
            sdata[x] = reduceOp(y, x, sdata[x], sdata[x + 256]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 256)
    {
        if (x < 128)
        {
            sdata[x] = reduceOp(y, x, sdata[x], sdata[x + 128]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 128)
    {
        if (x < 64)
        {
            sdata[x] = reduceOp(y, x, sdata[x], sdata[x + 64]);
        }
        __syncthreads();
    }

    volatile T *vdata = (volatile T *)sdata;
    ;
    if (x < 32)
    {
        if (BLOCK_X >= 64) vdata[x] = reduceOp(y, x, vdata[x], vdata[x + 32]);
        if (BLOCK_X >= 32) vdata[x] = reduceOp(y, x, vdata[x], vdata[x + 16]);
        if (BLOCK_X >= 16) vdata[x] = reduceOp(y, x, vdata[x], vdata[x + 8]);
        if (BLOCK_X >= 8) vdata[x] = reduceOp(y, x, vdata[x], vdata[x + 4]);
        if (BLOCK_X >= 4) vdata[x] = reduceOp(y, x, vdata[x], vdata[x + 2]);
        if (BLOCK_X >= 2) vdata[x] = reduceOp(y, x, vdata[x], vdata[x + 1]);
    }

    if (x == 0 and y < height)
    {
        result[y] = postProcess(y, 0, sdata[0]);
    }
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce_kernel_launcher(const T *A, T *result, ReduceOp op, uint32 n_reductions, uint32 aspan,
                            uint32 n_outputs, uint32 offset = 0, bool reducExisting = false,
                            T identity = T(0), PostProcess pProcess = PostProcess())
{
    if (n_outputs != n_reductions)
    {
        throw runtime_error_with_backtrace(
            "Number of outputs must be equal to number of reductions");
    }
    else if (aspan <= 32)
    {
        reduce_kernel<T, ReduceOp, 32>
            <<<n_outputs, 16>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan <= 64)
    {
        reduce_kernel<T, ReduceOp, 64>
            <<<n_outputs, 32>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan <= 128)
    {
        reduce_kernel<T, ReduceOp, 128>
            <<<n_outputs, 64>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan <= 256)
    {
        reduce_kernel<T, ReduceOp, 256>
            <<<n_outputs, 128>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan <= 512)
    {
        reduce_kernel<T, ReduceOp, 512>
            <<<n_outputs, 256>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan < 1024)
    {
        reduce_kernel<T, ReduceOp, 1024>
            <<<n_outputs, 512>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else
    {
        reduce_kernel<T, ReduceOp, 1880>
            <<<n_outputs, 940>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
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
        throw runtime_error_with_backtrace("Invalid dimensions for column-reduction " +
                                           A.shape_str + " to " + result.shape_str);
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
            LOG("For reducing to scalar from a column vector, use reduce_column_vec "
                "function");
        }
        throw runtime_error_with_backtrace("Invalid dimensions for row-reduction " + A.shape_str +
                                           " to " + result.shape_str);
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, A.height, A.width, result.height, 0,
                           false, identity, postProcess);
    cudaErrCheck(cudaDeviceSynchronize());
}

using FloatT = float64;
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

template void reduce_column_vec<FloatT, Plus<FloatT, FloatT>, Identity<FloatT>>(
    Matrix<FloatT> &, Matrix<FloatT> const &, Plus<FloatT, FloatT>, FloatT, Identity<FloatT>);