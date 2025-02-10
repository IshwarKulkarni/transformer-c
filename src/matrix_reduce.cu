#include <cuda_device_runtime_api.h>
#include <vector_types.h>
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "matrix_size_checks.hpp"
#include "utils.hpp"

//#define LOG_SIZE(...) LOG(__VA_ARGS__)
#define LOG_SIZE(...)

#define OLD_SOFTMAX_GRAD 1

// number of elements to read and sum per thread in sm_grad_kernel:
static constexpr uint32 SM_GRAD_READ_CT = 4;

template <typename T, uint32 BLOCK_X, typename ReduceOp>
__device__ void rolling_reduce(T* As, const ReduceOp& reduceOp, uint32 x)
{
    __syncthreads();

    if (BLOCK_X >= 1024 and x < 512)
    {
        As[x] = reduceOp(As[x], As[x + 512]);
        __syncthreads();
    }
    if (BLOCK_X >= 512 and x < 256)
    {
        As[x] = reduceOp(As[x], As[x + 256]);
        __syncthreads();
    }
    if (BLOCK_X >= 256 and x < 128)
    {
        As[x] = reduceOp(As[x], As[x + 128]);
        __syncthreads();
    }
    if (BLOCK_X >= 128 and x < 64)
    {
        As[x] = reduceOp(As[x], As[x + 64]);
        __syncthreads();
    }

    volatile T* vdata = (volatile T*)As;
    if (x < 32)
    {
        if (BLOCK_X >= 64) vdata[x] = reduceOp(vdata[x], vdata[x + 32]);
        if (BLOCK_X >= 32) vdata[x] = reduceOp(vdata[x], vdata[x + 16]);
        if (BLOCK_X >= 16) vdata[x] = reduceOp(vdata[x], vdata[x + 8]);
        if (BLOCK_X >= 8) vdata[x] = reduceOp(vdata[x], vdata[x + 4]);
        if (BLOCK_X >= 4) vdata[x] = reduceOp(vdata[x], vdata[x + 2]);
        if (BLOCK_X >= 2) vdata[x] = reduceOp(vdata[x], vdata[x + 1]);
    }
    __syncthreads();
}

template <typename T>
__device__ void vectorized_read(const T* global, T* local, uint32 numreads, uint32 offset)
{
    // if num reads == 4 and sizeof(T) == 4, then we can do a vectorized read
    // if num reads == 4 and sizeof(T) == 8, then we can do a vectorized read
    global += offset;
    if constexpr (sizeof(T) == 4)
    {
        bool is_aligned = offset % 4 == 0;
        if (numreads == 8 and is_aligned)
        {
            const double4* g = (double4*)global;
            double4* l = (double4*)local;
            l[0] = g[0];
        }
        else if (numreads == 4 and is_aligned)
        {
            const float4* g = (float4*)global;
            float4* l = (float4*)local;
            *l = *g;
            return;
        }
        else if (numreads == 2 and is_aligned)
        {
            const float2* g = (float2*)global;
            float2* l = (float2*)local;
            l[0] = g[0];
            return;
        }
    }
    else if constexpr (sizeof(T) == 8 and offset % 8 == 0)
    {
        bool is_aligned = offset % 8 == 0;
        if (numreads == 4 and is_aligned)
        {
            const double4* g = (double4*)global;
            double4* l = (double4*)local;
            l[0] = g[0];
            return;
        }

        else if (numreads == 2 and is_aligned)
        {
            double2* g = (double2*)global;
            double2* l = (double2*)local;
            l[0] = g[0];
            return;
        }
    }
    for (uint32 i = 0; i < numreads; i++)
    {
        local[i] = global[i];
    }
}

/*
 Special kernel for computing softmax gradient (composing this with existing functions is expensive)
 this kernel is laid out in 3 dimensions: 1d block, 3d grid. softmax_out and gradient_in are same
sized 2d matrices (with batch), output_size: height, h; and num_outputs: width, w; each column of
first matrix is generated by one softmax (i.e. they sum to 1.0). Each thread block reads the entire
column (index i) of gradient_in and softmax_out produces one output element in gradient_out, thus
the gradient_out size matches that of the entire grid.

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
    softmax_gradient<<< dim3(OH, 1, 1), dim3(output_height, output_with, batch) >>> (....,
output_height, output_width) OH = next_pow_2(output_height) output_height = size of column vector
roduced by softmax output_widht: num_output s . Number of column vectors produced by softmax forward
function.

    Summation for the dot product is accomplished like the reduction operation in other kernel in
this file.
*/
template <typename T, uint32 BLOCK_SIZE>
__global__ void sm_grad_kernel(Matrix<T> gradient_out, const Matrix<T> sm_out,
                               const Matrix<T> grad_in)
{
    __shared__ T s_outputs[BLOCK_SIZE + 1];
    __shared__ T s_output_y;

    // could be > width of softmax_outs upto next power of 2:
    uint32 t = threadIdx.x;  // indexes the jacobian cols; and rows for softmax_out and gradient_in,
    uint32 x = blockIdx.x;
    uint32 y = blockIdx.y;
    uint32 b = blockIdx.z;

    uint32 offset = sm_out.shape.offset(b, x, 0);
    const T* __restrict__ sm_out_ = sm_out.begin() + offset;
    const T* __restrict__ grd_in_ = grad_in.begin() + offset;

    if (t == 0) s_output_y = sm_out_[y];

    uint32 w_min = t * SM_GRAD_READ_CT;
    uint32 w_max = w_min + SM_GRAD_READ_CT;
    if (w_max > sm_out.width()) w_max = sm_out.width();
    __syncthreads();

    T Js = 0;
    for (uint32 w = w_min; w < w_max; w++)
    {
        T output = sm_out_[w];
        T Jsw = (w == y ? output * (1 - output) : -(s_output_y * output));
        Jsw *= grd_in_[w];
        Js += Jsw;
    }

    s_outputs[t] = Js;

    rolling_reduce<T, BLOCK_SIZE, Plus<T, T>>(s_outputs, Plus<T, T>(), t);

    if (threadIdx.x == 0) gradient_out(b, y, x) = s_outputs[0];
}

// s_out.shape and grad_in.shape should be same as s_grad_out.shape.t()
template <typename T>
void softmax_gradient(Matrix<T>& s_grad_out, const Matrix<T>& s_out, const Matrix<T>& grad_in)
{
    dim3 gridDim(s_grad_out.width(), s_grad_out.height(), s_grad_out.batch());
    uint32 span = iDivUp(grad_in.width(), SM_GRAD_READ_CT);
    uint32 w = nextPow2(span);
    dim3 blockDim(w, 1, 1);

    if (s_grad_out.shape.t() != s_out.shape or s_grad_out.shape.t() != grad_in.shape)
    {
        throw_rte_with_backtrace(
            "Dimension mismatch in softmax gradient, s_grad_out: ", s_grad_out.shape,
            ", s_out: ", s_out.shape, " & grad_in: ", grad_in.shape);
    }

    LOG_SIZE("s_grad_out: ", s_grad_out.shape, ", s_out: ", s_out.shape,
             ", grad_in: ", grad_in.shape, ", in_width:", grad_in.width(), ", span: ", span,
             ", w: ", w, ", grid: ", gridDim, ", block: ", blockDim);

    if (span <= 4)
        sm_grad_kernel<T, 4><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 8)
        sm_grad_kernel<T, 8><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 16)
        sm_grad_kernel<T, 16><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 32)
        sm_grad_kernel<T, 32><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 64)
        sm_grad_kernel<T, 64><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 128)
        sm_grad_kernel<T, 128><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 256)
        sm_grad_kernel<T, 256><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 512)
        sm_grad_kernel<T, 512><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else if (span <= 1024)
        sm_grad_kernel<T, 1024><<<gridDim, blockDim>>>(s_grad_out, s_out, grad_in);
    else
        throw_rte_with_backtrace("Span too large for softmax gradient");
    cudaErrCheck(cudaGetLastError());
}

// Reduces a matrix along a dimension using a reduction operation,
// and applies a post processing function to the result.
// The span of reduction (i.e. the dimension being reduced) is blockDim.x
// (i1=gridDim.x, i2=gridDim.y) is the size of the matrix in other two dimensions
// Should probably rewrite with thread groups.
template <typename T, uint32 dim, typename ReduceOp, uint32 BLOCK_X, typename PostProcess>
__global__ void reduce_kernel_rolling(Matrix<T> result, const Matrix<T> A, ReduceOp reduceOp,
                                      T identity = T(0), PostProcess postProcess = PostProcess())
{
    __shared__ T As[BLOCK_X + 1];
    uint32 i0 = threadIdx.x;  // reduction dimension index
    uint32 i1 = blockIdx.x;   // 1st of other two dimensions in order of w, h, b
    uint32 i2 = blockIdx.y;   // 2nd of other two dimensions in order of w, h, b

    As[i0] = i0 < A.shape[dim] ? A.template index<dim>(i0, i1, i2) : identity;
#pragma unroll
    for (uint32 otherIdx = i0 + blockDim.x; otherIdx < A.shape[dim]; otherIdx += blockDim.x)
    {
        As[i0] = reduceOp(As[i0], A.template index<dim>(otherIdx, i1, i2));
    }

    rolling_reduce<T, BLOCK_X, ReduceOp>(As, reduceOp, i0);
    if (i0 == 0)
    {
        result.template index<dim>(0, i1, i2) = postProcess(As[0]);
    }
}

// Reduces a matrix along a dimension using a reduction operation and postPorcess'es the restult
// This kernel is used when number of elements to reduce is < 32
template <typename T, uint32 dim, typename ReduceOp, typename PostProcess>
__global__ void reduce_kernel_linear(Matrix<T> result, const Matrix<T> A, uint32 l1, uint32 l2,
                                     ReduceOp reduceOp, T identity = T(0),
                                     PostProcess postProcess = PostProcess())
{
    uint32 i1 = threadIdx.x + blockDim.x * blockIdx.x;
    uint32 i2 = threadIdx.y + blockDim.y * blockIdx.y;

    if (i1 >= l1 or i2 >= l2) return;

    T res = A.template index<dim>(0, i1, i2);
#pragma unroll
    for (uint32 i0 = 1; i0 < A.shape[dim]; i0++)
    {
        res = reduceOp(res, A.template index<dim>(i0, i1, i2));
    }

    result.template index<dim>(0, i1, i2) = postProcess(res);
}

template <typename T, uint32 dim, typename ReduceOp, typename PostProcess>
void reduce(Matrix<T>& result, const Matrix<T>& A, ReduceOp op, T identity, PostProcess pProcess)
{
    static_assert(dim < 3, "Invalid dimension for reduction");
    if (A.shape[dim] == 1)
    {
        if (std::is_same<PostProcess, Identity<T>>::value)
            throw_rte_with_backtrace("Invalid reduction operation: dim == 1 for shape: ", A.shape);
        throw_rte_with_backtrace(YELLOW, "Skipping reduction of dimension ", dim,
                                 " of size 1, use unary operation");
    }
    check_reduction_sizes<T, dim>(result, A);

    uint32 l0 = A.shape[dim], l1, l2;  // l0 is |reduction dimension|
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

    if (l0 <= 20)
    {
        static constexpr uint32 max_threads = 160;
        uint32 b1 = std::min(l1, max_threads);
        uint32 b2 = std::min(max_threads / b1, l2);
        dim3 blockDim(b1, b2, 1);
        dim3 gridDim(iDivUp(l1, b1), iDivUp(l2, b2), 1);
        LOG_SIZE("Reduce ", A.shape, " to ", result.shape, " on dim [", dim, "] Block: ", blockDim,
                 " Grid: ", gridDim, " using reduce_kernel_linear");
        reduce_kernel_linear<T, dim, ReduceOp>
            <<<gridDim, blockDim>>>(result, A, l1, l2, op, identity, pProcess);
        return;
    }

    l0 = std::min(1024u, nextPow2(A.shape[dim]));
    l0 = l0 >= 4 ? l0 / 2 : 2;  // minimum 2 reductions.

    dim3 blockDim(l0, 1, 1);
    dim3 gridDim(l1, l2);
    LOG_SIZE("Reduce ", A.shape, " to ", result.shape, " on dim [", dim, "] Block: ", blockDim,
             " Grid: ", gridDim, " using reduce_kernel_rolling");

    if (l0 <= 2)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 2>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 4)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 4>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 8)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 8>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 16)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 16>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 32)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 32>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 64)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 64>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 128)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 128>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 256)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 256>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 <= 512)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 512>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else if (l0 < 1024)
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 1024>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    else
    {
        reduce_kernel_rolling<T, dim, ReduceOp, 1880>
            <<<gridDim, blockDim>>>(result, A, op, identity, pProcess);
    }
    cudaErrCheck(cudaGetLastError());
}

template void softmax_gradient<FloatT>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                       Matrix<FloatT> const&);

template void reduce<FloatT, 0u, Plus<FloatT, FloatT>, Identity<FloatT>>(Matrix<FloatT>&,
                                                                         Matrix<FloatT> const&,
                                                                         Plus<FloatT, FloatT>,
                                                                         FloatT, Identity<FloatT>);

template void reduce<FloatT, 0u, Plus<FloatT, FloatT>, DividedBy<FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Plus<FloatT, FloatT>, FloatT, DividedBy<FloatT>);

template void reduce<FloatT, 1u, Plus<FloatT, FloatT>, Sigmoid<FloatT>::SigmoidF>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Plus<FloatT, FloatT>, FloatT,
    Sigmoid<FloatT>::SigmoidF);

template void reduce<FloatT, 1u, Min<FloatT>, Identity<FloatT>>(Matrix<FloatT>&,
                                                                Matrix<FloatT> const&, Min<FloatT>,
                                                                FloatT, Identity<FloatT>);

template void reduce<FloatT, 2u, Plus<FloatT, FloatT>, DividedBy<FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Plus<FloatT, FloatT>, FloatT, DividedBy<FloatT>);

template void reduce<FloatT, 2u, Plus<FloatT, FloatT>, Sigmoid<FloatT>::SigmoidF>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Plus<FloatT, FloatT>, FloatT,
    Sigmoid<FloatT>::SigmoidF);

template void reduce<FloatT, 2u, Plus<FloatT, FloatT>, Identity<FloatT>>(Matrix<FloatT>&,
                                                                         Matrix<FloatT> const&,
                                                                         Plus<FloatT, FloatT>,
                                                                         FloatT, Identity<FloatT>);

template void reduce<FloatT, 0u, Plus<FloatT, FloatT>, Sigmoid<FloatT>::SigmoidF>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Plus<FloatT, FloatT>, FloatT,
    Sigmoid<FloatT>::SigmoidF);

template void reduce<FloatT, 1u, Plus<FloatT, FloatT>, Identity<FloatT>>(Matrix<FloatT>&,
                                                                         Matrix<FloatT> const&,
                                                                         Plus<FloatT, FloatT>,
                                                                         FloatT, Identity<FloatT>);

template void reduce<FloatT, 1u, Plus<FloatT, FloatT>, DividedBy<FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Plus<FloatT, FloatT>, FloatT, DividedBy<FloatT>);

template void reduce<FloatT, 0u, Min<FloatT>, Identity<FloatT>>(Matrix<FloatT>&,
                                                                Matrix<FloatT> const&, Min<FloatT>,
                                                                FloatT, Identity<FloatT>);

template void reduce<FloatT, 2u, Min<FloatT>, Identity<FloatT>>(Matrix<FloatT>&,
                                                                Matrix<FloatT> const&, Min<FloatT>,
                                                                FloatT, Identity<FloatT>);

template void reduce<FloatT, 0u, Plus<FloatT, FloatT>, Loge<FloatT>>(Matrix<FloatT>&,
                                                                     Matrix<FloatT> const&,
                                                                     Plus<FloatT, FloatT>, FloatT,
                                                                     Loge<FloatT>);

template void reduce<FloatT, 1u, Plus<FloatT, FloatT>, Loge<FloatT>>(Matrix<FloatT>&,
                                                                     Matrix<FloatT> const&,
                                                                     Plus<FloatT, FloatT>, FloatT,
                                                                     Loge<FloatT>);

template void reduce<float, 0u, Plus<float, float>, Neg<float>>(Matrix<float>&,
                                                                Matrix<float> const&,
                                                                Plus<float, float>, float,
                                                                Neg<float>);

template void reduce<float, 1u, Plus<float, float>, Neg<float>>(Matrix<float>&,
                                                                Matrix<float> const&,
                                                                Plus<float, float>, float,
                                                                Neg<float>);

template void reduce<float, 2u, Plus<float, float>, Neg<float>>(Matrix<float>&,
                                                                Matrix<float> const&,
                                                                Plus<float, float>, float,
                                                                Neg<float>);