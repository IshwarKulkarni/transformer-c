#include "../headers/matrix_ops.cuh"

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
    cudaErrCheck(cudaGetLastError());
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
__global__ void split_kernel(T **__restrict__ outputs, const T *__restrict__ input, uint32 height, uint32 width, Op op)
{
    uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32 y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32 z = blockIdx.z;
    uint32 num_outputs = gridDim.z;

    if (x >= width || y >= height) return;
    uint32 in_offset =
        (width * num_outputs) * y + z * width + x;  // (larger width) * y + z * width + x
    outputs[z][y * width + x] = input[in_offset];
}

static FloatT **split_matrix_ptrs = nullptr;
static constexpr uint32 SPLIT_MAX = 128;
std::mutex split_mtx;

template <typename T, typename Op>
void split(std::vector<Matrix<T> *> &outputs, const Matrix<T> &res, Op op)
{
    std::lock_guard<std::mutex> lock(split_mtx);
    if (split_matrix_ptrs == nullptr)
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

    if (shape.first != res.height || shape.second * outputs.size() != res.width)
    {
        LOG(BOLD, RED, "Matrix dimensions do not match for split operation: ", res.shape_str,
            " -> ", shape.first, "x", shape.second);
        throw_rte_with_backtrace("Dimension mismatch for split");
    }

    dim3 blockDim(32, 32);
    dim3 gridDim(iDivUp(shape.second, 32), iDivUp(shape.first, 32), outputs.size());
    split_kernel<T, Op>
        <<<gridDim, blockDim>>>(split_matrix_ptrs, res.begin(), shape.first, shape.second, op);
}

template void transpose(Matrix<FloatT> &res, const Matrix<FloatT> &A, Identity<FloatT>);

template void transpose<FloatT, Exp<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &, Exp<FloatT>);

template void transpose<FloatT, Neg<FloatT>>(Matrix<FloatT> &, Matrix<FloatT> const &, Neg<FloatT>);

template void concat<FloatT, Identity<FloatT>>(
    Matrix<FloatT> &, std::vector<Matrix<FloatT> *, std::allocator<Matrix<FloatT> *>> const &,
    Identity<FloatT>);


template void split<FloatT, Identity<FloatT> >(std::vector<Matrix<FloatT>*, std::allocator<Matrix<FloatT>*> >&, Matrix<FloatT> const&, Identity<FloatT>);