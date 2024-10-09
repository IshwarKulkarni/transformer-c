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

    if (aW == 1) Axy[0] = 0; // broadcast along x axis
    if (aH == 1) Axy[1] = 0;
    if (bW == 1) Bxy[0] = 0;
    if (bH == 1) Bxy[1] = 0;

    result[y * resW + x] = op(A[Axy[0] + aW * Axy[1]], B[Bxy[0] + bW * Bxy[1]]);
}

template <typename Ta, typename Tb, typename Tr, typename Op>
void binary_apply(Matrix<Tr> &res, const Matrix<Ta> &A, const Matrix<Tb> &B, Op op)
{
    // a and b's dimensions should match result dimensions either on height or
    // width or numels == 1

    auto match_atleast_one_dim = [](uint32 rh, uint32 rw, uint32 ah, uint32 aw) -> bool {
        if (rh != ah) return ah == 1 and rw == aw;
        if (rw != aw) return aw == 1 and rh == ah;
        return true;
    };

    if (B.numels() != 1 and
        match_atleast_one_dim(res.height, res.width, B.height, B.width) == false)
    {
        LOG(RED, "Dimension mismatch: R, B: ", res.shape_string(), " != ", B.shape_string());
        throw std::runtime_error("Dimension mismatch");
    }

    if (A.numels() != 1 and
        match_atleast_one_dim(res.height, res.width, A.height, A.width) == false)
    {
        LOG(RED, "Dimension mismatch: R, A: ", res.shape_string(), " != ", A.shape_string());
        throw std::runtime_error("Dimension mismatch");
    }

    if (!match_atleast_one_dim(A.height, A.width, B.height, B.width))
    {
        LOG(RED, "Dimension mismatch: A, B: ", A.shape_string(), " != ", B.shape_string());
        throw std::runtime_error("Dimension mismatch");
    }

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

    if (aW == 1) Axy[0] = 0; // broadcast along x axis
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