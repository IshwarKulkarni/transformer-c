#include "../headers/matrix_ops.cuh"
#include "curand_kernel.h"

template <typename T, uint32 BLOCK_SIZE, typename Op>
__global__ void transpose_kernel(T *__restrict__ result, const T *__restrict__ A, uint32 height,
                                 uint32 width, Op op)
{
    __shared__ float32 tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    uint32 x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint32 y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height) tile[threadIdx.y][threadIdx.x] = A[y * width + x];

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    T out = op(tile[threadIdx.x][threadIdx.y]);

    if (y < width && x < height) result[y * height + x] = out;
}

template <typename T, typename Op>
void transpose(Matrix<T> &res, const Matrix<T> &A, Op op)
{
    if (A.height != res.width || A.width != res.height)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for transpose operation: ", A.shape_str,
            " -> ", res.shape_str);
        throw_rte_with_backtrace("Dimension mismatch for transpose");
    }

    if (A.width == 1 and std::is_same<Op, Identity<T>>::value)
    {
        fill(res, A.begin());
        return;
    }

    constexpr uint32 BLOCK_SIZE = 32;
    uint32 max_dim = std::max(A.width, A.height);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(iDivUp(max_dim, BLOCK_SIZE), iDivUp(max_dim, BLOCK_SIZE));
    transpose_kernel<T, BLOCK_SIZE, Op>
        <<<gridDim, blockDim>>>(res.begin(), A.begin(), A.height, A.width, op);
}

template <typename T, typename Op = Identity<T>>
__global__ void concat_kernel(T *__restrict__ result, T **__restrict__ inputs, uint32 height,
                              uint32 width, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 z = blockIdx.z;
    uint32 num_inputs = gridDim.z;

    if (x >= width || y >= height) return;
    uint32 out_offset =
        (width * num_inputs) * y + z * width + x;  // (larger width) * y + z * width + x
    result[out_offset] = op(inputs[z][y * width + x]);
}

static FloatT **concat_matrix_ptrs = nullptr;
static constexpr uint32 CONCAT_MAX = 128;
std::mutex concat_mtx;

template <typename T, typename Op>
void concat(Matrix<T> &res, const std::vector<Matrix<T> *> &inputs, Op op)
{
    std::lock_guard<std::mutex> lock(concat_mtx);
    if (concat_matrix_ptrs == nullptr)
    {
        cudaErrCheck(cudaMallocManaged(&concat_matrix_ptrs, CONCAT_MAX * sizeof(FloatT *)));
    }

    if (inputs.size() > CONCAT_MAX)
    {
        throw_rte_with_backtrace("Number of matrices to concatenate exceeds ", CONCAT_MAX);
    }

    auto shape = inputs[0]->shape();
    for (uint32 i = 0; i < inputs.size(); i++)
    {
        if (inputs[i]->shape() != shape)
        {
            LOG(BOLD, RED,
                "Matrix shapes do not match for concatenation operation: ", inputs[i]->shape_str,
                " -> ", inputs[0]->shape_str);
            throw_rte_with_backtrace("Dimension mismatch for concatenation");
        }
        concat_matrix_ptrs[i] = inputs[i]->begin();
    }

    if (shape.first != res.height || shape.second * inputs.size() != res.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for concatenation operation: ", shape.first,
            "x", shape.second, " -> ", res.shape_str);
        throw_rte_with_backtrace("Dimension mismatch for concatenation");
    }

    dim3 blockDim(32, 32);
    dim3 gridDim(iDivUp(shape.second, 32), iDivUp(shape.first, 32), inputs.size());
    concat_kernel<T, Op>
        <<<gridDim, blockDim>>>(res.begin(), concat_matrix_ptrs, shape.first, shape.second, op);
}

template <typename T, typename Op>
__global__ void split_kernel(T **__restrict__ outputs, const T *__restrict__ input, uint32 height,
                             uint32 width, uint32 num_outputs, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= num_outputs) return;
    uint32 in_offset =
        (width * num_outputs) * y + z * width + x;  // (larger width) * y + z * width + x

    outputs[z][y * width + x] = op(input[in_offset]);
}

static FloatT **split_matrix_ptrs = nullptr;
static constexpr uint32 SPLIT_MAX = 128;
std::mutex split_mtx;

template <typename T, typename Op>
void split(std::vector<Matrix<T> *> &outputs, const Matrix<T> &input, Op op)
{
    // std::lock_guard<std::mutex> lock(split_mtx);
    bool need_alloc = split_matrix_ptrs == nullptr;
    if (need_alloc)
    {
        cudaErrCheck(cudaMallocManaged(&split_matrix_ptrs, SPLIT_MAX * sizeof(FloatT *)));
    }

    if (outputs.size() > SPLIT_MAX)
    {
        throw_rte_with_backtrace("Number of matrices to split exceeds ", SPLIT_MAX);
    }

    auto shape = outputs[0]->shape();
    for (uint32 i = 0; i < outputs.size(); i++)
    {
        if (outputs[i]->shape() != shape)
        {
            LOG(BOLD, RED,
                "Matrix shapes do not match for split operation: ", outputs[i]->shape_str, " -> ",
                outputs[0]->shape_str);
            throw_rte_with_backtrace("Dimension mismatch for split");
        }
        split_matrix_ptrs[i] = outputs[i]->begin();
    }

    if (shape.first != input.height || shape.second * outputs.size() != input.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for split operation: ", input.shape_str,
            " -> ", shape.first, "x", shape.second);
        throw_rte_with_backtrace("Dimension mismatch for split");
    }

    dim3 blockDim(128, 2, 2);
    dim3 gridDim(iDivUp(shape.second, blockDim.x), iDivUp(shape.first, blockDim.y),
                 iDivUp(outputs.size(), 2));
    split_kernel<T, Op><<<gridDim, blockDim>>>(split_matrix_ptrs, input.begin(), shape.first,
                                               shape.second, outputs.size(), op);
}

static std::shared_ptr<curandState> dropout_states{};
static constexpr uint32 Ks = 16;
static constexpr uint32 DROPOUT_MAX_SIZE = Ks * 1024;

template <typename T>
__global__ void dropout_kernel(T *__restrict__ res, const T *__restrict__ in,
                               float32 *__restrict__ mask, uint32 height, uint32 width,
                               float32 drop_prob, curandState *states)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 offset = y * width + x;

    if (x >= width || y >= height) return;

    float32 mask_val = 0.f;
    if (drop_prob > 0 and drop_prob < 1)  // valid dropout probability, generate
    {
        bool keep = (curand_uniform(&states[offset % DROPOUT_MAX_SIZE]) > drop_prob);
        mask_val = keep ? 1.f / (1 - drop_prob) : 0.f;
        mask[offset] = mask_val;
    }
    else  // use existing mask value at offset
    {
        mask_val = mask[offset];
    }
    res[offset] = in[offset] * mask_val;
}

__global__ void init_curand_states(curandState *states, uint32 size, uint32 seed)
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
void dropout(Matrix<T> &res, const Matrix<T> &in, Matrix<float32> &mask, float32 drop_prob)
{
    if (dropout_states == nullptr)
    {
        curandState *_states = nullptr;
        cudaErrCheck(cudaMallocManaged(&_states, DROPOUT_MAX_SIZE * sizeof(curandState)));
        dropout_states =
            std::shared_ptr<curandState>(_states, [](curandState *ptr) { cudaFree(ptr); });

        init_curand_states<<<1024, Ks>>>(_states, DROPOUT_MAX_SIZE, time(NULL));
        cudaErrCheck(cudaDeviceSynchronize());
    }
    auto shape = res.shape();
    if (shape != mask.shape())
    {
        LOG(BOLD, RED, "Matrix shapes do not match for dropout operation: ", res.shape_str, " -> ",
            mask.shape_str);
        throw_rte_with_backtrace("Dimension mismatch for dropout");
    }

    dim3 blockDim(32, 32);
    dim3 gridDim(iDivUp(shape.second, 32), iDivUp(shape.first, 32));
    dropout_kernel<T><<<gridDim, blockDim>>>(res.begin(), in.begin(), mask.begin(), in.height,
                                             in.width, drop_prob, dropout_states.get());
}

template void transpose(Matrix<FloatT> &res, const Matrix<FloatT> &A, Identity<FloatT>);

template void transpose<FloatT, Exp<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &, Exp<FloatT>);

template void transpose<FloatT, Neg<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &, Neg<FloatT>);

template void concat<FloatT, Identity<FloatT>>(
    Matrix<FloatT> &, std::vector<Matrix<FloatT> *, std::allocator<Matrix<FloatT> *>> const &,
    Identity<FloatT>);

template void split<FloatT, Identity<FloatT>>(
    std::vector<Matrix<FloatT> *, std::allocator<Matrix<FloatT> *>> &, Matrix<FloatT> const &,
    Identity<FloatT>);

template void dropout<FloatT>(Matrix<FloatT> &res, const Matrix<FloatT> &in, Matrix<float32> &mask,
                              float32 drop_prob);

template void transpose<FloatT, TanH<FloatT>::TanhB>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                     TanH<FloatT>::TanhB);

template void transpose<FloatT, Sigmoid<FloatT>::SigmoidB>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                           Sigmoid<FloatT>::SigmoidB);

template void transpose<FloatT, Relu<FloatT>::ReluB>(Matrix<FloatT> &, Matrix<FloatT> const &,
                                                     Relu<FloatT>::ReluB);

template void transpose<FloatT, LeakyRelu<FloatT>::LeakyReluB>(Matrix<FloatT> &,
                                                               Matrix<FloatT> const &,
                                                               LeakyRelu<FloatT>::LeakyReluB);