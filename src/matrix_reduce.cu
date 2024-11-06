#include <cuda_device_runtime_api.h>
#include <vector_types.h>
#include "../headers/matrix_ops.cuh"

/*
 Special kernel for computing softmax gradient (composing this with existing functions is expensive)
 this kernel is laid out in 3 dimensions: 1d block, 2d grid. softmax_out and gradient_in are same
sized 2d matrices, output_size: height, h; and num_outputs: width, w; each column of first matrix is
generated by one softmax (i.e. they sum to 1.0). Each thread block reads the entire column (index i)
of gradient_in and softmax_out produces one output element in gradient_out, thus the gradient_out
size matches that of the entire grid.

Essentially we are computing
   J_s(i) * G(i)
     where J_s(i) is the jacobian of the softmax function for a ith output column, and
     G(i) is the incoming gradient for that column. Each of

      J_s(i),  --caled J below--,  is a matrix with values:
      J(x, y) = O(x) * (1 - O(x)) if x == y and O(k) is k'th element in softmax_out's i'th column;
      J(x, y) = -O(x) * O(y) if x != y

    Dot product each yth row of J with column G(i) produces yth element of ith column of
gradient_out

This kernel is launched as
    softmax_gradient<<< dim3(OH, 1, 1), dim3(output_height, output_with,1) >>> (...., output_height,
output_width) OH = next_pow_2(output_height) output_height = size of column vector roduced by
softmax output_widht: num_output s . Number of column vectors produced by softmax forward function.

    Summation for the dot product is accomplished like the reduction operation in other kernel in
this file.
*/
template <typename T, uint32 BLOCK_SIZE>
__global__ void softmax_grad_kernel(T *__restrict__ gradient_out, const T *__restrict__ softmax_out,
                                    const T *__restrict__ gradient_in, bool sout_isT, bool gin_isT)
{
    uint32 x = threadIdx.x;  // indexes the jacobian cols; and rows for softmax_out and gradient_in,
                             // could be > width of softmax_outs upto next power of 2
    uint32 col_idx = blockIdx.x;  // == column offset of input and output,
    uint32 y = blockIdx.y;        // == row offset of input and output

    uint32 height = gridDim.y;
    uint32 width = gridDim.x;
    uint32 sin_offset = sout_isT ? x + col_idx * height : col_idx + x * width;
    uint32 gin_offset = gin_isT ? x + col_idx * height : col_idx + x * width;
    bool oob = x >= height or col_idx >= width;

    __shared__ T s_outputs[BLOCK_SIZE + 1];
    for (uint32 i = col_idx; i < BLOCK_SIZE + 1 and x == 0; i++)
    {
        s_outputs[i] = 0;
    }

    s_outputs[x] = oob ? 0 : softmax_out[sin_offset];

    __syncthreads();
    T Jsi = ((x == y) ? s_outputs[x] * (1 - s_outputs[x]) : -(s_outputs[y] * s_outputs[x]));

    s_outputs[x] = Jsi * (oob ? 0 : gradient_in[gin_offset]);

    if (BLOCK_SIZE >= 1024)
    {
        if (x < 512)
        {
            s_outputs[x] = s_outputs[x] + s_outputs[x + 512];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 512)
    {
        if (x < 256)
        {
            s_outputs[x] = s_outputs[x] + s_outputs[x + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256)
    {
        if (x < 128)
        {
            s_outputs[x] = s_outputs[x] + s_outputs[x + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128)
    {
        if (x < 64)
        {
            s_outputs[x] = s_outputs[x] + s_outputs[x + 64];
        }
        __syncthreads();
    }

    volatile T *vdata = (volatile T *)s_outputs;

    if (x < 32)
    {
        if (BLOCK_SIZE >= 64) vdata[x] = vdata[x] + vdata[x + 32];
        if (BLOCK_SIZE >= 32) vdata[x] = vdata[x] + vdata[x + 16];
        if (BLOCK_SIZE >= 16) vdata[x] = vdata[x] + vdata[x + 8];
        if (BLOCK_SIZE >= 8) vdata[x] = vdata[x] + vdata[x + 4];
        if (BLOCK_SIZE >= 4) vdata[x] = vdata[x] + vdata[x + 2];
        if (BLOCK_SIZE >= 2) vdata[x] = vdata[x] + vdata[x + 1];
    }

    __syncthreads();

    if (x == 0)
    {
        gradient_out[blockIdx.x + blockIdx.y * width] = vdata[x];
    }
}

template <typename T>
void softmax_gradient(Matrix<T> &s_grad_out, const Matrix<T> &s_out, const Matrix<T> &grad_in)
{
    dim3 gridDim(s_grad_out.width, s_grad_out.height, 1);
    uint32 span = s_grad_out.height;
    uint32 w = nextPow2(span);
    dim3 blockDim(w, 1, 1);

    check_softmax_grad_sizes(s_grad_out, s_out, grad_in);

    bool sout_isT = (s_out.width == s_grad_out.height and s_out.height == s_grad_out.width);
    bool gin_isT = (grad_in.width == s_grad_out.height and grad_in.height == s_grad_out.width);

    if (span <= 8)
    {
        softmax_grad_kernel<T, 8><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                         grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 16)
    {
        softmax_grad_kernel<T, 16><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                          grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 32)
    {
        softmax_grad_kernel<T, 32><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                          grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 64)
    {
        softmax_grad_kernel<T, 64><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                          grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 128)
    {
        softmax_grad_kernel<T, 128><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                           grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 256)
    {
        softmax_grad_kernel<T, 256><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                           grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 512)
    {
        softmax_grad_kernel<T, 512><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                           grad_in.begin(), sout_isT, gin_isT);
    }
    else if (span <= 1024)
    {
        softmax_grad_kernel<T, 1024><<<gridDim, blockDim>>>(s_grad_out.begin(), s_out.begin(),
                                                            grad_in.begin(), sout_isT, gin_isT);
    }
    else
    {
        throw_rte_with_backtrace("Span too large for softmax gradient");
    }
}

// it's best to read this kernel as reducing along the width of the matrix
template <typename T, typename ReduceOp, uint32 BLOCK_X, typename PostProcess>
__global__ void reduce_kernel(const T *A, T *result, ReduceOp reduceOp, uint32 height, uint32 width,
                              T identity = T(0), PostProcess postProcess = PostProcess())
{
    __shared__ T As[BLOCK_X + 1];
    uint32 x = threadIdx.x;
    uint32 y = blockIdx.x;

    A += y * width;  // this thread block only deals with one row.

    As[x] = x < width ? A[x] : identity;
#pragma unrolls
    for (uint32 otherIdx = x + blockDim.x; otherIdx < width; otherIdx += blockDim.x)
    {
        As[x] = reduceOp(As[x], A[otherIdx]);
    }

    __syncthreads();

    if (BLOCK_X >= 1024)
    {
        if (x < 512)
        {
            As[x] = reduceOp(As[x], As[x + 512]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 512)
    {
        if (x < 256)
        {
            As[x] = reduceOp(As[x], As[x + 256]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 256)
    {
        if (x < 128)
        {
            As[x] = reduceOp(As[x], As[x + 128]);
        }
        __syncthreads();
    }
    if (BLOCK_X >= 128)
    {
        if (x < 64)
        {
            As[x] = reduceOp(As[x], As[x + 64]);
        }
        __syncthreads();
    }

    volatile T *vdata = (volatile T *)As;
    ;
    if (x < 32)
    {
        if (BLOCK_X >= 64) vdata[x] = reduceOp(vdata[x], vdata[x + 32]);
        if (BLOCK_X >= 32) vdata[x] = reduceOp(vdata[x], vdata[x + 16]);
        if (BLOCK_X >= 16) vdata[x] = reduceOp(vdata[x], vdata[x + 8]);
        if (BLOCK_X >= 8) vdata[x] = reduceOp(vdata[x], vdata[x + 4]);
        if (BLOCK_X >= 4) vdata[x] = reduceOp(vdata[x], vdata[x + 2]);
        if (BLOCK_X >= 2) vdata[x] = reduceOp(vdata[x], vdata[x + 1]);
    }

    if (x == 0 and y < height)
    {
        result[y] = postProcess(As[0]);
    }
}

template <typename T, typename ReduceOp, uint32 BLOCK, typename PostProcess>
__global__ void reduce_kernel_small(const T *A, T *result, ReduceOp reduceOp, uint32 height,
                                    uint32 width, T identity = T(0),
                                    PostProcess postProcess = PostProcess())
{
    uint32 y = threadIdx.y;

    A += y * width;  // this thread block only deals with one row.
    T out = A[0];
#pragma unrolls
    for (uint32 i = 1; i < width; i++)
    {
        out = reduceOp(A[i], out);
    }
    result[y] = postProcess(out);
}

template <typename T, typename ReduceOp, typename PostProcess>
void reduce_kernel_launcher(const T *A, T *result, ReduceOp op, uint32 n_reductions, uint32 aspan,
                            uint32 n_outputs, T identity = T(0),
                            PostProcess pProcess = PostProcess())
{
    if (n_outputs != n_reductions)
    {
        throw_rte_with_backtrace("Number of outputs must be equal to number of reductions");
    }
    else if (aspan <= 8 and n_reductions <= 16)
    {
        reduce_kernel_small<T, ReduceOp, 16>
            <<<1, dim3(1, n_reductions)>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 8)
    {
        reduce_kernel<T, ReduceOp, 8>
            <<<n_outputs, 4>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 16)
    {
        reduce_kernel<T, ReduceOp, 16>
            <<<n_outputs, 8>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 32)
    {
        reduce_kernel<T, ReduceOp, 32>
            <<<n_outputs, 16>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 64)
    {
        reduce_kernel<T, ReduceOp, 64>
            <<<n_outputs, 32>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 128)
    {
        reduce_kernel<T, ReduceOp, 128>
            <<<n_outputs, 64>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 256)
    {
        reduce_kernel<T, ReduceOp, 256>
            <<<n_outputs, 128>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan <= 512)
    {
        reduce_kernel<T, ReduceOp, 512>
            <<<n_outputs, 256>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else if (aspan < 1024)
    {
        reduce_kernel<T, ReduceOp, 1024>
            <<<n_outputs, 512>>>(A, result, op, n_reductions, aspan, identity, pProcess);
    }
    else
    {
        reduce_kernel<T, ReduceOp, 1880>
            <<<n_outputs, 940>>>(A, result, op, n_reductions, aspan, identity, pProcess);
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
            throw_rte_with_backtrace("A is scalar consider copying instead");
        }
        throw_rte_with_backtrace("Invalid dimensions for column-reduction ", A.shape_str, " to ",
                                 result.shape_str, " or invalid postProcess");
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, 1, A.height, 1, identity,
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
        throw_rte_with_backtrace("Invalid dimensions for row-reduction ", A.shape_str, " to ",
                                 result.shape_str, " or invalid postProcess");
    }
    reduce_kernel_launcher(A.begin(), result.begin(), reduceOp, A.height, A.width, result.height,
                           identity, postProcess);
}

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

template void softmax_gradient<FloatT>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                       Matrix<FloatT> const &);