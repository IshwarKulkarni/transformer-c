#include <cuda_device_runtime_api.h>
#include <vector_types.h>
#include "../headers/matrix_ops.cuh"

// it's best to read this kernel as reducing along the width of the matrix
template <typename T, typename ReduceOp, uint32 BLOCK_X, typename PostProcess>
__global__ void reduce_kernel(const T *A, T *result, ReduceOp reduceOp, uint32 height, uint32 width,
                              bool reducExisting = false, T identity = T(0),
                              PostProcess postProcess = PostProcess())
{
    __shared__ T As[BLOCK_X + 1];
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;

    A += y * width;  // this thread block only deals with one row.

    As[x] = x < width ? A[x] : identity;
#pragma unrolls
    for (uint32 otherIdx = x + blockDim.x; otherIdx < width; otherIdx += blockDim.x)
    {
        As[x] = reduceOp(y, x, As[x], A[otherIdx]);
    }

    __syncthreads();

    if (BLOCK_X >= 1024)
    {
        if (x < 512)
        {
            As[x] = reduceOp(y, x, As[x], As[x + 512]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 512)
    {
        if (x < 256)
        {
            As[x] = reduceOp(y, x, As[x], As[x + 256]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 256)
    {
        if (x < 128)
        {
            As[x] = reduceOp(y, x, As[x], As[x + 128]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 128)
    {
        if (x < 64)
        {
            As[x] = reduceOp(y, x, As[x], As[x + 64]);
        }
        __syncthreads();
    }

    volatile T *vdata = (volatile T *)As;
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
        result[y] = postProcess(y, 0, As[0]);
    }
}

template <typename T, typename ReduceOp, uint32 BLOCK, typename PostProcess>
__global__ void reduce_kernel_small(const T *A, T *result, ReduceOp reduceOp, uint32 height,
                                    uint32 width, bool reducExisting = false, T identity = T(0),
                                    PostProcess postProcess = PostProcess())
{
    uint32 y = threadIdx.y;

    A += y * width;  // this thread block only deals with one row.
    T out = A[0];
#pragma unrolls
    for (uint32 i = 1; i < width; i++)
    {
        out = reduceOp(y, i, A[i], out);
    }
    result[y] = postProcess(y, 0, out);
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce_kernel_launcher(const T *A, T *result, ReduceOp op, uint32 n_reductions, uint32 aspan,
                            uint32 n_outputs, bool reducExisting = false, T identity = T(0),
                            PostProcess pProcess = PostProcess())
{
    if (n_outputs != n_reductions)
    {
        throw runtime_error_with_backtrace(
            "Number of outputs must be equal to number of reductions");
    }
    else if (aspan <= 8 and n_reductions <= 16)
    {
        reduce_kernel_small<T, ReduceOp, 16><<<1, dim3(1, n_reductions)>>>(
            A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan <= 8)
    {
        reduce_kernel<T, ReduceOp, 8>
            <<<n_outputs, 4>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
    }
    else if (aspan <= 16)
    {
        reduce_kernel<T, ReduceOp, 16>
            <<<n_outputs, 8>>>(A, result, op, n_reductions, aspan, false, identity, pProcess);
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
            throw runtime_error_with_backtrace("A is scalar consider copying instead");
        }
        throw runtime_error_with_backtrace("Invalid dimensions for column-reduction " +
                                           A.shape_str + " to " + result.shape_str);
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, 1, A.height, 1, false, identity,
                           postProcess);
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce(Matrix<T> &result, const Matrix<T> &A, ReduceOp reduceOp, T identity,
            PostProcess postProcess)
{
    if (A.width == 1 and result.width == 1)
    {
        if (result.height == A.height and std::is_same<PostProcess, Identity<T>>::value)
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
                                           " to " + result.shape_str + " use unary_apply instead");
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, A.height, A.width, result.height,
                           false, identity, postProcess);
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

template void reduce<FloatT, Plus<FloatT, FloatT>, Neg<FloatT>>(Matrix<FloatT> &,
                                                                Matrix<FloatT> const &,
                                                                Plus<FloatT, FloatT>, FloatT,
                                                                Neg<FloatT>);

template void reduce_column_vec<FloatT, Plus<FloatT, FloatT>, Neg<FloatT>>(Matrix<FloatT> &,
                                                                           Matrix<FloatT> const &,
                                                                           Plus<FloatT, FloatT>,
                                                                           FloatT, Neg<FloatT>);