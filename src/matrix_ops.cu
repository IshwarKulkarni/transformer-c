#include "curand_kernel.h"
#include "matrix_ops.cuh"
#include "matrix_size_checks.hpp"

template <typename T, uint32 BLOCK_SIZE, typename Op>
__global__ void transpose_kernel(Matrix<T> res, const Matrix<T> A, Op op)
{
    __shared__ float32 tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    uint32 x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32 y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    uint32 b = blockIdx.z;

    if (!A.is_oob(b, y, x)) tile[threadIdx.y][threadIdx.x] = A(b, y, x);

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (res.is_oob(b, y, x)) return;
    auto v = (tile[threadIdx.x][threadIdx.y]);
    v = op(v);
    res(b, y, x) = v;
}

template <typename T, typename Op>
void transpose(Matrix<T>& res, const Matrix<T>& A, Op op)
{
    if (A.shape != res.shape.t())
        throw_rte_with_backtrace(
            BOLD, RED, "Matrix dimensions do not match for transpose operation: ", A.shape, " -> ",
            res.shape);

    if (A.width() == 1 and std::is_same<Op, Identity<T>>::value)
    {
        if (res.id != A.id) res.copy(A.begin());
        return;
    }

    uint32 max_dim = std::max(A.width(), A.height());
    if (max_dim < 16)
    {
        constexpr uint32 BLOCK_SIZE = 16;
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        auto gdim = iDivUp(max_dim, BLOCK_SIZE);
        dim3 gridDim(gdim, gdim, A.batch());
        transpose_kernel<T, BLOCK_SIZE, Op><<<gridDim, blockDim>>>(res, A, op);
    }
    else
    {
        constexpr uint32 BLOCK_SIZE = 32;
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        auto gdim = iDivUp(max_dim, BLOCK_SIZE);
        dim3 gridDim(gdim, gdim, A.batch());
        transpose_kernel<T, BLOCK_SIZE, Op><<<gridDim, blockDim>>>(res, A, op);
    }
    cudaErrCheck(cudaGetLastError());
}

// concat along width, one thread for each output element
template <typename T, uint32 Dim = 0, typename Op = Identity<T>>
__global__ void concat_kernel(Matrix<T> res, const Matrix<const FloatT*> inputs, Shape in_shape,
                              uint32 num_in, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 b = blockIdx.z * blockDim.z + threadIdx.z;

    if (b >= in_shape.batch or y >= in_shape.height or x >= in_shape.width) return;

    uint32 offset = in_shape.offset(b, y, x);

    for (uint32 i = 0; i < num_in; ++i)
    {
        auto in = *(inputs(i, 0) + offset);
        if (Dim == 0)
            res(b, y, x + i * in_shape.width) = op(in);
        else if (Dim == 1)
            res(b, y + i * in_shape.height, x) = op(in);
        else if (Dim == 2)
            res(b + i * in_shape.batch, y, x) = op(in);
    }
}

// TODO; will blow up in multi-threaded environment
static Matrix<const FloatT*> concat_ptrs({128, 1}, "FloatTPtrsForConcat");

template <typename T, uint32 Dim, typename Op>
void concat(Matrix<T>& res, const std::vector<Matrix<T>*>& inputs, Op op)
{
    static_assert(Dim < 3, "Invalid dimension for concatenation, 0: width, 1: height, 2: batch");

    uint32 n = inputs.size();
    if (n == 0) throw_rte_with_backtrace("No matrices to concat");
    if (n > concat_ptrs.height()) throw_rte_with_backtrace("Too many matrices to concat");

    // match all shapes:
    concat_ptrs.reset();
    auto in_shape = inputs[0]->shape;
    for (uint32 i = 0; i < n; ++i)
    {
        auto mp = inputs[i];
        concat_ptrs(i, 0) = mp->begin();
        if (mp->shape != in_shape)
            throw_rte_with_backtrace("Matrix ", *mp, " shaped ", mp->shape,
                                     " has different size for concat");
    }

    Shape exp_res_shape = in_shape.set(Dim, in_shape[Dim] * n);
    if (exp_res_shape != res.shape)
    {
        throw_rte_with_backtrace(
            BOLD, RED, "Matrix dimensions do not match for concatenation operation: ", n, "*",
            in_shape, " -> ", res.shape, " along dimension ", Dim, " expected: ", exp_res_shape);
    }

    dim3 blockDim(24, 24, 1);
    dim3 gridDim = inputs[0]->grid(blockDim);
    concat_kernel<T, Dim, Op><<<gridDim, blockDim>>>(res, concat_ptrs, in_shape, inputs.size(), op);
    cudaErrCheck(cudaGetLastError());
}

static Matrix<FloatT*> split_mat_ptrs({128, 1}, "FloatTPtrsForSplit");

template <typename T, uint32 Dim, typename Op>
__global__ void split_kernel(Matrix<T*> splits, const Matrix<T> merged, Shape split_shape,
                             uint32 num_outs, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 b = blockIdx.z * blockDim.z + threadIdx.z;

    if (split_shape.is_oob(b, y, x)) return;
    uint32 split_offset = split_shape.offset(b, y, x);

    for (uint32 i = 0; i < num_outs; ++i)
    {
        if (Dim == 0)
            *(splits(i, 0) + split_offset) = merged(b, y, x + i * split_shape.width);
        else if (Dim == 1)
            *(splits(i, 0) + split_offset) = merged(b, y + i * split_shape.height, x);
        else if (Dim == 2)
            *(splits(i, 0) + split_offset) = merged(b + i * split_shape.batch, y, x);
    }
}

template <typename T, uint32 Dim, typename Op>
void split(std::vector<Matrix<T>*>& outputs, const Matrix<T>& input, Op op)
{
    uint32 n = outputs.size();
    if (n == 0) throw_rte_with_backtrace("Zero output matrices for split");
    if (n > split_mat_ptrs.height()) throw_rte_with_backtrace("Too many matrices to split");

    // match all shapes:
    split_mat_ptrs.reset();
    auto out_shape = outputs[0]->shape;
    for (uint32 i = 0; i < n; ++i)
    {
        auto mp = outputs[i];
        split_mat_ptrs(i, 0) = mp->get().get();
        if (outputs[i]->shape != out_shape)
            throw_rte_with_backtrace("Matrix ", *mp, " shaped ", mp->shape,
                                     " has incorrect size for split");
    }

    Shape exp_input_shape = out_shape.set(Dim, out_shape[Dim] * n);
    if (exp_input_shape != input.shape)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for split operation: ", input.shape, " -> ",
            n, exp_input_shape);
        throw_rte_with_backtrace("Dimension incorrect for split");
    }

    dim3 blockDim(24, 24, 1);
    dim3 gridDim = outputs[0]->grid(blockDim);
    split_kernel<T, Dim, Op>
        <<<gridDim, blockDim>>>(split_mat_ptrs, input, out_shape, outputs.size(), op);
    cudaErrCheck(cudaGetLastError());
}

static std::shared_ptr<curandState> dropout_states{};
static constexpr uint32 Ks = 16;
static constexpr uint32 DROPOUT_MAX_SIZE = Ks * 1024;

template <typename T>
__global__ void dropout_kernel(Matrix<T> res, const Matrix<T> in, Matrix<float32> mask,
                               float32 drop_prob, curandState* states)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 b = blockIdx.z * blockDim.z + threadIdx.z;

    if (res.is_oob(b, y, x)) return;

    uint32 offset = res.shape.offset(b, y, x);

    float32 mask_val = 0.f;
    if (drop_prob > 0 and drop_prob < 1)  // valid dropout probability, generate
    {
        bool keep = (curand_uniform(&states[offset % DROPOUT_MAX_SIZE]) > drop_prob);
        mask_val = keep ? 1.f / (1 - drop_prob) : 0.f;
        mask.template broadcasting_fetch<0b111>(b, y, x) = mask_val;
    }
    else  // use existing mask value at offset
    {
        mask_val = mask.template broadcasting_fetch<0b111>(b, y, x);
    }
    res(b, y, x) = in.template broadcasting_fetch<0b111>(b, y, x) * mask_val;
}

__global__ void init_curand_states(curandState* states, uint32 size, uint32 seed)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 idx = y * blockDim.x + x;
    if (idx < size)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ```
//    if  0<p<=1
//      res[x,y] = (mask[x, y] = (rand() < drop_prob)) ? 0 : res[x, y];
//    else:
//      res[x,y] = res[x,y] * mask[x,y]
// ```
template <typename T>
void dropout(Matrix<T>& res, const Matrix<T>& in, Matrix<float32>& mask, float32 drop_prob)
{
    if (dropout_states == nullptr)
    {
        curandState* _states = nullptr;
        cudaErrCheck(cudaMallocManaged(&_states, DROPOUT_MAX_SIZE * sizeof(curandState)));
        dropout_states =
            std::shared_ptr<curandState>(_states, [](curandState* ptr) { cudaFree(ptr); });

        init_curand_states<<<1024, Ks>>>(_states, DROPOUT_MAX_SIZE,
                                         static_cast<uint32>(time(NULL)));
        cudaErrCheck(cudaDeviceSynchronize());
    }
    if (broadcastable<0>(in, res) and broadcastable<1>(in, res) and broadcastable<1>(in, res) and
        broadcastable<0>(mask, res) and broadcastable<2>(mask, res) and broadcastable<2>(mask, res))
    // all good
    {
    }
    else
    {
        throw_rte_with_backtrace("Dimension mismatch for dropout: ", in.shape, " -> ", res.shape,
                                 " with mask: ", mask.shape);
    }

    dim3 block(24, 24);
    dropout_kernel<T><<<res.grid(block), block>>>(res, in, mask, drop_prob, dropout_states.get());
    cudaErrCheck(cudaGetLastError());
}

template <typename T, typename Tb, typename Tr, typename Op>
__global__ void binary_apply_kernel(Matrix<Tr> res, const Matrix<T> A, const Matrix<Tb> B, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 b = blockIdx.z;

    if (res.is_oob(b, y, x)) return;

    static constexpr uint32 ALL = BATCH_BIT | HEIGHT_BIT | WIDTH_BIT;
    auto a_val = A.template broadcasting_fetch<ALL>(b, y, x);
    auto b_val = B.template broadcasting_fetch<ALL>(b, y, x);

    res(b, y, x) = op(a_val, b_val);
}

template <typename Tr, typename T, typename Tb, typename Op>
void binary_apply(Matrix<Tr>& res, const Matrix<T>& A, const Matrix<Tb>& B, Op op)
{
    check_broadcast_sizes<T>(res, A, B);
    dim3 block(std::min(24u, res.width()), std::min(24u, res.height()));  // slightly inefficient
    auto grid = res.grid(block);
    binary_apply_kernel<Tr, T, Tb, Op><<<grid, block>>>(res, A, B, op);
    cudaErrCheck(cudaGetLastError());
}

template <typename Tr, typename T, typename Op>
__global__ void unary_apply_kernel(Matrix<Tr> res, const Matrix<T> A, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 b = blockIdx.z;

    if (res.is_oob(b, y, x)) return;

    uint32 Axy[3] = {b, y, x};

    if (A.batch() == 1) Axy[0] = 0;
    if (A.height() == 1) Axy[1] = 0;
    if (A.width() == 1) Axy[2] = 0;

    auto a_val = A(Axy[0], Axy[1], Axy[2]);
    res(b, y, x) = op(a_val);
}

template <typename T, typename Tr, typename Op>
void unary_apply(Matrix<Tr>& res, const Matrix<T>& A, Op op)
{
    check_broadcast_sizes(res, A);
    dim3 block(32, 32, 1);
    unary_apply_kernel<<<res.grid(block), block>>>(res, A, op);
    cudaErrCheck(cudaGetLastError());
}

template <typename Tr, typename T, typename Op>
__global__ void ternary_apply_kernel(Matrix<Tr> res, const Matrix<T> A, const Matrix<T> B,
                                     const Matrix<T> C, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 b = blockIdx.z * blockDim.z + threadIdx.z;

    if (res.is_oob(b, y, x)) return;
    static constexpr uint32 ALL = BATCH_BIT | HEIGHT_BIT | WIDTH_BIT;

    auto a_val = A.template broadcasting_fetch<ALL>(b, y, x);
    auto b_val = B.template broadcasting_fetch<ALL>(b, y, x);
    auto c_val = C.template broadcasting_fetch<ALL>(b, y, x);

    res(b, y, x) = op(a_val, b_val, c_val);
}

template <typename T, typename Op>
void ternary_apply(Matrix<T>& res, const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C,
                   Op op)
{
    check_broadcast_sizes<T>(res, A, B, C);
    dim3 block(16, 16, 1);
    ternary_apply_kernel<T, T, Op><<<res.grid(block), block>>>(res, A, B, C, op);
    cudaErrCheck(cudaGetLastError());
}

// TODO: This is getting out of hand, need to passin Op as pointer-to-base-class for unary, binary
// and ternary ops
template void transpose(Matrix<FloatT>& res, const Matrix<FloatT>& A, Identity<FloatT>);

template void transpose<FloatT, Exp<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&, Exp<FloatT>);

template void transpose<FloatT, Neg<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&, Neg<FloatT>);

template void concat<FloatT, 0, Identity<FloatT>>(
    Matrix<FloatT>&, std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*>> const&,
    Identity<FloatT>);

template void concat<FloatT, 1, Identity<FloatT>>(
    Matrix<FloatT>&, std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*>> const&,
    Identity<FloatT>);

template void concat<FloatT, 2, Identity<FloatT>>(
    Matrix<FloatT>&, std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*>> const&,
    Identity<FloatT>);

template void split<FloatT, 0, Identity<FloatT>>(
    std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*>>&, Matrix<FloatT> const&,
    Identity<FloatT>);

template void split<FloatT, 1, Identity<FloatT>>(
    std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*>>&, Matrix<FloatT> const&,
    Identity<FloatT>);

template void split<FloatT, 2, Identity<FloatT>>(
    std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*>>&, Matrix<FloatT> const&,
    Identity<FloatT>);

template void dropout<FloatT>(Matrix<FloatT>& res, const Matrix<FloatT>& in, Matrix<float32>& mask,
                              float32 drop_prob);

template void transpose<FloatT, TanH<FloatT>::TanhB>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                     TanH<FloatT>::TanhB);

template void transpose<FloatT, Sigmoid<FloatT>::SigmoidB>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                           Sigmoid<FloatT>::SigmoidB);

template void transpose<FloatT, Relu<FloatT>::ReluB>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                     Relu<FloatT>::ReluB);

template void transpose<FloatT, LeakyRelu<FloatT>::LeakyReluB>(Matrix<FloatT>&,
                                                               Matrix<FloatT> const&,
                                                               LeakyRelu<FloatT>::LeakyReluB);

template void binary_apply<FloatT, FloatT, FloatT, Plus<FloatT, FloatT>>(Matrix<FloatT>&,
                                                                         Matrix<FloatT> const&,
                                                                         Matrix<FloatT> const&,
                                                                         Plus<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, Neg<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                       Neg<FloatT>);

template void
binary_apply<FloatT, FloatT, FloatT, Composition<FloatT, Sub<FloatT, FloatT>, Square<FloatT>>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&,
    Composition<FloatT, Sub<FloatT, FloatT>, Square<FloatT>>);

template void unary_apply<FloatT, FloatT, DividedBy<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                             DividedBy<FloatT>);
template void binary_apply<FloatT, FloatT, FloatT, MomentUpdate<FloatT>>(Matrix<FloatT>&,
                                                                         Matrix<FloatT> const&,
                                                                         Matrix<FloatT> const&,
                                                                         MomentUpdate<FloatT>);
template void binary_apply<FloatT, FloatT, FloatT, SecondMomentUpdate<FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, SecondMomentUpdate<FloatT>);
template void binary_apply<FloatT, FloatT, FloatT, AdamWeightUpdate<FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, AdamWeightUpdate<FloatT>);
template void binary_apply<FloatT, FloatT, FloatT, WeightUpdate<FloatT, FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, WeightUpdate<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, Exp<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                       Exp<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, Div<FloatT, FloatT>>(Matrix<FloatT>&,
                                                                        Matrix<FloatT> const&,
                                                                        Matrix<FloatT> const&,
                                                                        Div<FloatT, FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, NegLogLossFwd<FloatT, FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, NegLogLossFwd<FloatT, FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, NegLogLossBckwd<FloatT, FloatT>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&, NegLogLossBckwd<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, MultiplyBy<FloatT>>(Matrix<FloatT>&,
                                                              Matrix<FloatT> const&,
                                                              MultiplyBy<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, Sub<FloatT, FloatT>>(Matrix<FloatT>&,
                                                                        Matrix<FloatT> const&,
                                                                        Matrix<FloatT> const&,
                                                                        Sub<FloatT, FloatT>);

template void unary_apply<FloatT, FloatT, Square<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                          Square<FloatT>);

template void unary_apply<FloatT, FloatT, TanH<FloatT>::TanhB>(Matrix<FloatT>&,
                                                               Matrix<FloatT> const&,
                                                               TanH<FloatT>::TanhB);
template void unary_apply<FloatT, FloatT, Sigmoid<FloatT>::SigmoidB>(Matrix<FloatT>&,
                                                                     Matrix<FloatT> const&,
                                                                     Sigmoid<FloatT>::SigmoidB);
template void unary_apply<FloatT, FloatT, Identity<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                            Identity<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, Mul<FloatT, FloatT>>(Matrix<FloatT>&,
                                                                        Matrix<FloatT> const&,
                                                                        Matrix<FloatT> const&,
                                                                        Mul<FloatT, FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, ActBackwardMul<FloatT, Sigmoid<FloatT>>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&,
    ActBackwardMul<FloatT, Sigmoid<FloatT>>);

template void binary_apply<FloatT, FloatT, FloatT, ActBackwardMul<FloatT, TanH<FloatT>>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&,
    ActBackwardMul<FloatT, TanH<FloatT>>);

template void binary_apply<FloatT, FloatT, FloatT, ActBackwardMul<FloatT, IActivation<FloatT>>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&,
    ActBackwardMul<FloatT, IActivation<FloatT>>);

template void unary_apply<FloatT, FloatT, NLSToSoftmax<FloatT>>(Matrix<FloatT>&,
                                                                Matrix<FloatT> const&,
                                                                NLSToSoftmax<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, LSMCEBkwd<FloatT>>(Matrix<FloatT>&,
                                                                      Matrix<FloatT> const&,
                                                                      Matrix<FloatT> const&,
                                                                      LSMCEBkwd<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, ActBackwardMul<FloatT, Relu<FloatT>>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&,
    ActBackwardMul<FloatT, Relu<FloatT>>);

template void unary_apply<FloatT, FloatT, Abs<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                       Abs<FloatT>);

template void unary_apply<FloatT, FloatT, Sign<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                        Sign<FloatT>);

template void ternary_apply<FloatT, Norm<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                  Matrix<FloatT> const&, Matrix<FloatT> const&,
                                                  Norm<FloatT>);

template void unary_apply<FloatT, FloatT, Pow<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                       Pow<FloatT>);

template void ternary_apply<FloatT, DivDiff<FloatT>>(Matrix<FloatT>&, Matrix<FloatT> const&,
                                                     Matrix<FloatT> const&, Matrix<FloatT> const&,
                                                     DivDiff<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, PowDiff<FloatT>>(Matrix<FloatT>&,
                                                                    Matrix<FloatT> const&,
                                                                    Matrix<FloatT> const&,
                                                                    PowDiff<FloatT>);

template void binary_apply<FloatT, FloatT, FloatT, ActBackwardMul<FloatT, LeakyRelu<FloatT>>>(
    Matrix<FloatT>&, Matrix<FloatT> const&, Matrix<FloatT> const&,
    ActBackwardMul<FloatT, LeakyRelu<FloatT>>);

// template void unary_apply<FloatT, FloatT, Sqrt<FloatT> >(Matrix<FloatT>&, Matrix<FloatT> const&,
// Sqrt<FloatT>);

// template void binary_apply<FloatT, FloatT, FloatT, SqrtDiff<FloatT> >(Matrix<FloatT>&,
// Matrix<FloatT> const&, Matrix<FloatT> const&, SqrtDiff<FloatT>);

// template void unary_apply<FloatT, FloatT, PowDiff<FloatT> >(Matrix<FloatT>&, Matrix<FloatT>
// const&, PowDiff<FloatT>);