/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <limits>
#include <map>
#include <memory>
#include <vector>
#include "errors.hpp"
#include "logger.hpp"
#include "types"
#include "utils.hpp"

static constexpr uint32 WIDTH_IDX = 0;
static constexpr uint32 HEIGHT_IDX = 1;
static constexpr uint32 BATCH_IDX = 2;

static_assert(WIDTH_IDX < HEIGHT_IDX && HEIGHT_IDX < BATCH_IDX, "Invalid dimension order");

static constexpr uint32 WIDTH_BIT = 0x1 << WIDTH_IDX;
static constexpr uint32 HEIGHT_BIT = 0x1 << HEIGHT_IDX;
static constexpr uint32 BATCH_BIT = 0x1 << BATCH_IDX;

inline __device__ __host__ uint32 iDivUp(uint32 a, uint32 b) { return (a + b - 1) / b; }

template <uint32 Dim>  // i is the index in the dimension Dim, other two are the other dimensions in
                       // order of b, y, x
constexpr inline __host__ __device__ std::tuple<uint32, uint32, uint32> get_indices(uint32 i,
                                                                                    uint32 i1,
                                                                                    uint32 i2)
{
    static_assert(Dim <= BATCH_IDX, "Invalid dimension for get_indices");
    if constexpr (Dim == WIDTH_IDX) return std::make_tuple(i1, i2, i);
    if constexpr (Dim == HEIGHT_IDX) return std::make_tuple(i1, i, i2);
    if constexpr (Dim == BATCH_IDX) return std::make_tuple(i, i1, i2);
}

struct Shape
{
    const uint32 width, height, batch;
    const uint64 numels = width * height * batch;
    const uint64 size2d = width * height;
    Shape(uint32 batch, uint32 height, uint32 width) : width(width), height(height), batch(batch) {}
    Shape(uint32 height, uint32 width) : width(width), height(height), batch(1) {}
    // there should be a `std::vector<uint32> heigher_dims` such that their product is batch
    inline __host__ __device__ bool operator==(const Shape& other) const
    {
        return batch == other.batch && height == other.height && width == other.width;
    }

    inline uint32 __device__ __host__ operator[](uint64 i) const
    {
        if (i == WIDTH_IDX) return width;
        if (i == HEIGHT_IDX) return height;
        if (i == BATCH_IDX) return batch;
        throw_rte_with_backtrace("Index out of range: ", i);
        return static_cast<uint32>(-1);
    }

    inline bool __device__ __host__ is_oob(uint32 b, uint32 y, uint32 x) const
    {
        return b >= batch || y >= height || x >= width;
    }

    inline bool operator!=(const Shape& other) const { return !(*this == other); }

    template <typename T>
    inline uint64 bytes() const
    {
        return numels * sizeof(T);
    }

    inline __host__ __device__ uint64 offset(uint32 b, uint32 y, uint32 x) const
    {
#ifndef DISABLE_SIZE_CHECK
        if (b >= batch || y >= height || x >= width)
        {
            throw_oob3_with_backtrace(b, y, x, batch, height, width);
        }
#endif
        return b * height * width + y * width + x;
    }

    template <uint32 Dim>
    inline __host__ __device__ uint64 offset_in_dim(uint32 i, uint32 i1, uint32 i2) const
    {
        auto [b, y, x] = get_indices<Dim>(i, i1, i2);
        return offset(b, y, x);
    }

    template <unsigned int bits>
    inline __host__ __device__ uint64 broadcasting_offset(uint32 b, uint32 y, uint32 x) const
    {
        static_assert(bits <= 0b111, "dim bits must in [0b00, 0b111]");
        // clang-format off
        if (bits & WIDTH_BIT and width == 1)   x = 0;
        if (bits & HEIGHT_BIT and height == 1) y = 0;
        if (bits & BATCH_BIT and batch == 1)   b = 0;
        // clang-format on

        return offset(b, y, x);
    }

    Shape t() const { return {batch, width, height}; }

    Shape shape2d() const { return {1, height, width}; }

    operator std::string() const
    {
        return '[' + std::to_string(batch) + " x " + std::to_string(height) + " x " +
               std::to_string(width) + ']';
    }

    Shape set(uint32 dim, uint32 val) const
    {
        if (dim == BATCH_IDX) return {val, height, width};
        if (dim == HEIGHT_IDX) return {batch, val, width};
        if (dim == WIDTH_IDX) return {batch, height, val};
        throw_rte_with_backtrace("Invalid dimension: ", dim);
        return Shape(0, 0, 0);
    }

    std::string str() const { return (std::string) * this; }
};

inline std::ostream& operator<<(std::ostream& os, const Shape& s) { return os << s.str(); }

typedef struct MatrixInitUitls
{
    static uint32 get_id() { return ++id; }
    static uint32 peek_id() { return id; }
    static uint64 get_alloced_bytes() { return alloced_bytes; }

    template <typename T>
    static T* allocManaged(const Shape& shape, uint32 id)
    {
        T* ptr = nullptr;
        if (shape.numels == 0) throw_rte_with_backtrace("Cannot allocate a matrix with 0 elements");
        LOG_ALLOC("Allocating matrix ", id);
        (void)id;
        cudaErrCheck(cudaMallocManaged((void**)&ptr, shape.bytes<T>()));
        alloced_bytes += shape.numels * sizeof(T);
        id_to_alloced_bytes[id] = shape.numels * sizeof(T);
        return ptr;
    }

    static uint32* allocManagedExtent(uint32 length, uint32 id)
    {
        uint32* ptr = nullptr;
        LOG_ALLOC("Allocating extent for matrix ", id, " len: ", length);
        (void)id;
        cudaErrCheck(cudaMallocManaged((void**)&ptr, length * sizeof(uint32)));
        alloced_bytes += length * sizeof(uint32);
        return ptr;
    }

    static void free(uint32* ptr, uint32 length, uint32 id)
    {
        LOG_FREE("Freeing extent for matrix ", id);
        (void)id;
        cudaErrCheck(cudaFree(ptr));
        freed_bytes += length * sizeof(uint32);
    }

    template <typename T>
    static void free(T* ptr, uint32 id)
    {
        LOG_FREE("Freeing matrix ", id);
        (void)id;
        cudaErrCheck(cudaFree(ptr));
        freed_bytes += id_to_alloced_bytes[id];
    }

 private:
    MatrixInitUitls() = delete;
    static uint32 id;
    static uint64 alloced_bytes;
    static uint64 freed_bytes;
    static std::map<uint32, uint64> id_to_alloced_bytes;
} MatrixInitUitls;

struct MatrixBase
{
    MatrixBase() { all_matrices.push_back(this); }
    virtual ~MatrixBase() = default;
    static std::vector<const MatrixBase*> all_matrices;
};

// This class embodies "valid extents" for a batched 2d matrix.
// A Matrix is expected to have a valid element at (b, y, x) if
// y < y_extent[b] and x < x_extent[b], for a batch b.
// I.e. valid values span from (b, 0, 0) to (b, y_extent[b], x_extent[b]).
// Can also set x_extent[b] and y_extent[b] to 0 to make the batch invalid.
struct Extents2d
{
    const uint32 matrixid;
    const Shape shape;

    // all rows and cols are valid
    Extents2d(uint32 matrixid, const Shape& shape) : matrixid(matrixid), shape(shape)
    {
        // set all row values to shape.height and all col values to shape.width
        for (uint32 i = 0; i < shape.batch; i++) set(i, shape.height, shape.width);
    }

    template <uint32 Dim>
    inline __host__ __device__ uint32 set(uint32 batch, uint32 val)
    {
        if constexpr (Dim == WIDTH_IDX)
            set(batch, shape.height, val);
        else if constexpr (Dim == HEIGHT_IDX)
            set(batch, val, shape.width);
        throw_rte_with_backtrace("Invalid dimension: ", Dim);
    }

    void set(uint32 batch, uint32 y, uint32 x)
    {
        if (batch >= shape.batch)
        {
            throw_rte_with_backtrace("Invalid batch: ", batch, " for matrix ", matrixid);
        }

        if (y > shape.height)
        {
            throw_rte_with_backtrace("Invalid row y extent: ", y, " for batch ", batch,
                                     " and matrix ", matrixid);
        }
        if (x > shape.width)
        {
            throw_rte_with_backtrace("Invalid col x extent: ", x, " for batch ", batch,
                                     " and matrix ", matrixid);
        }
        y_extent[batch] = y;
        x_extent[batch] = x;
    }

    __host__ __device__ uint32 operator()(uint32 batch, uint32 dim) const
    {
        if (dim == WIDTH_IDX) return x_extent[batch];
        if (dim == HEIGHT_IDX) return y_extent[batch];
        if (dim == BATCH_IDX) return shape.batch;  // for completeness, not used
        throw_rte_with_backtrace("Invalid dimension: ", dim);
        return 0;
    }

    __host__ __device__ std::tuple<uint32, uint32> operator()(uint32 batch) const
    {
        return std::make_tuple(y_extent[batch], x_extent[batch]);
    }

    template <uint32 Dim>  // same logic as Shape::offset_in_dim
    __host__ __device__ bool in_bounds(uint32 i, uint32 i1, uint32 i2) const
    {
        auto [b, y, x] = get_indices<Dim>(i, i1, i2);
        return in(b, y, x);
    }

    __host__ __device__ bool in(uint32 batch, uint32 y, uint32 x) const
    {
        return batch < shape.batch && y < y_extent[batch] && x < x_extent[batch];
    }

    __host__ __device__ uint32 width(uint32 batch) const { return x_extent[batch]; }

    __host__ __device__ uint32 height(uint32 batch) const { return y_extent[batch]; }

    bool all_valid() const
    {
        // all x_extent_ptr are width and all y_extent_ptr are height
        for (uint32 i = 0; i < shape.batch; i++)
        {
            if (x_extent[i] != shape.width || y_extent[i] != shape.height) return false;
        }
        return true;
    }

 private:
    Extents2d() = delete;
    std::shared_ptr<uint32[]> extent = std::shared_ptr<uint32[]>(
        MatrixInitUitls::allocManagedExtent(shape.batch * 2, matrixid), [this](uint32* ptr) {
            MatrixInitUitls::free(ptr, shape.batch, matrixid);
            y_extent = nullptr;
            x_extent = nullptr;
        });

    uint32* x_extent = extent.get();
    uint32* y_extent = extent.get() + shape.batch;
};

inline std::ostream& operator<<(std::ostream& os, const Extents2d& extents)
{
    for (uint32 i = 0; i < extents.shape.batch; i++)
    {
        os << i << ": [" << extents(i, HEIGHT_IDX) << ", " << extents(i, WIDTH_IDX) << "]\t";
    }
    return os;
}

/*
Matrix class for 3d tensors (batch, height, width) with managed memory allocation
and automatic deallocation on destruction. The data is stored in a shared pointer
that is returned by the get() method. Matrices cannot be copied, only moved.
Allows for creation with `shape`, and vector of matrices to concatenate along the batch dimension.

Data is stored in row-major order, i.e. 0th dimension is width, 1st is height, and 2nd is batch
and increment of pointer from get() or begin() is along the width dimension.

Access is done with
    3-element () operator: batchIdx, heightIdx, widthIdx
    1-element [] operator: linear offset
    or index method: index<0>(i, m, n) is equivalent to operator()(m, n, i), i is width dim index
                     index<1>(i, m, n) is equivalent to operator()(m, i, n), i is height dim index
                     index<2>(i, m, n) is equivalent to operator()(i, m, n), i is batch dim index
*/
template <typename T>
struct Matrix
{
    const uint32 id = MatrixInitUitls::get_id();
    const std::string name;
    const Shape shape;
    Extents2d extents = Extents2d(id, shape);

    typedef std::shared_ptr<T[]> CudaPtr;

    Matrix() : id(0), name("Empty"), shape(0, 0, 0), data() {}

    Matrix(Shape shape, const std::string& name_ = "Matrix")
        : name(name_ + '{' + std::to_string(id) + '}'), shape(shape)
    {
        LOG_MATRIX_CREATE(this->name, " : ", this->shape, " size: ", this->shape.numels,
                          " bytes: ", this->shape.bytes<T>());
    }

    inline uint32 sum_batches(std::vector<const Matrix<T>*> mats)
    {
        uint32 sum = 0;
        for (auto m : mats) sum += m->batch();
        return sum;
    }

    // concat matrices along the batch dimension
    Matrix(std::vector<const Matrix<T>*> mats, const std::string& name_ = "Matrix")
        : Matrix({sum_batches(mats), mats[0]->height(), mats[0]->width()}, name_)
    {
        auto shape = mats[0]->shape;
        uint64 offset = 0;
        for (auto m : mats)
        {
            if (m->shape.shape2d() != shape.shape2d())
                throw_rte_with_backtrace("All matrices must have the same height and width");
            offset += memcpy(m->begin(), offset, m->shape.numels);
        }
    }

    Matrix<T>& operator=(const Matrix<T>&) = delete;

    CudaPtr get() { return data; }

    inline __host__ __device__ uint32 height() const { return shape.height; }

    inline __host__ __device__ uint32 width() const { return shape.width; }

    inline __host__ __device__ uint32 batch() const { return shape.batch; }

    inline __host__ __device__ const T& operator()(uint32 y, uint32 x) const
    {
        // if (rawData == nullptr) throw_rte_with_backtrace("Matrix data is null");
        if (batch() != 1)
            throw_rte_with_backtrace(
                "Matrix is not 2D, use operator()(uint32 b, uint32 y, uint32 x)");
        return rawData[shape.offset(0, y, x)];
    }

    inline __host__ __device__ T& operator()(uint32 y, uint32 x)
    {
        // if (rawData == nullptr) throw_rte_with_backtrace("Matrix data is null");
        if (batch() != 1)
            throw_rte_with_backtrace(
                "Matrix is not 2D, use operator()(uint32 b, uint32 y, uint32 x)");
        return rawData[shape.offset(0, y, x)];
    }

    inline __host__ __device__ const T& operator()(uint32 b, uint32 y, uint32 x) const
    {
        // if (rawData == nullptr) throw_rte_with_backtrace("Matrix data is null");
        return rawData[shape.offset(b, y, x)];
    }

    inline __device__ __host__ T& operator()(uint32 b, uint32 y, uint32 x)
    {
        // if (rawData == nullptr) throw_rte_with_backtrace("Matrix data is null");
        return rawData[shape.offset(b, y, x)];
    }

    // return element at b, y, x , by zeroing out any dimendion that is 1.
    // and having corresponding dim set in `bits`, e.g. 0b001 for width, 0b010 for height, 0b100 for
    // batch. Error out if the corresponding dimension is not 1.
    template <unsigned int bits>
    inline __device__ __host__ const T& broadcasting_fetch(uint32 b, uint32 y, uint32 x) const
    {
        return rawData[shape.template broadcasting_offset<bits>(b, y, x)];
    }

    template <unsigned int bits>
    inline __device__ __host__ T& broadcasting_fetch(uint32 b, uint32 y, uint32 x)
    {
        return rawData[shape.template broadcasting_offset<bits>(b, y, x)];
    }

    // return element at index i in dimension dim, using i1 and i2 as the other indices
    // in order of batch, height, width. e.g. index<1>(5, 1, 2) is equivalent to operator()(1, 5, 2)
    template <unsigned int Dim>
    inline __device__ __host__ T& index(uint32 i, uint32 i1, uint32 i2)
    {
        return rawData[shape.template offset_in_dim<Dim>(i, i1, i2)];
    }

    template <unsigned int Dim>
    inline __device__ __host__ const T& index(uint32 i, uint32 i1, uint32 i2) const
    {
        return rawData[shape.template offset_in_dim<Dim>(i, i1, i2)];
    }

    // grid size for given block to have a thread for each element in matrix
    dim3 grid(dim3 block) const
    {
        return dim3(iDivUp(width(), block.x), iDivUp(height(), block.y), iDivUp(batch(), block.z));
    }

    template <typename U>  // copy to `batch`th batch if batch is valid, else copy all
    inline uint64 copy(const U* src, Optional<uint32> batch = {})
    {
        if (!src) throw_rte_with_backtrace("Cannot copy from null pointer");
        bool all_batches = !batch.is_valid();
        uint64 copy_count = (all_batches ? shape.numels : shape.size2d);
        uint64 offset = (all_batches ? 0 : shape.offset(*batch, 0, 0));
        return memcpy(src, offset, copy_count);
    }

    inline uint64 reset()
    {
        cudaErrCheck(cudaMemset(rawData, 0, shape.bytes<T>()));
        return shape.numels;
    }

    template <typename U>
    inline void set_val(const U& val)
    {
        for (uint64 i = 0; i < shape.numels; i++) rawData[i] = val;
    }

    inline __host__ __device__ const T& operator[](uint64 i) const
    {
        if (i >= shape.numels) throw_oob1_with_backtrace(i, shape.numels);
        return rawData[i];
    }

    inline __host__ __device__ T& operator[](uint64 i)
    {
        if (i >= shape.numels) throw_oob1_with_backtrace(i, shape.numels);
        return rawData[i];
    }

    inline __host__ __device__ const T* begin() const { return rawData; }

    inline __host__ __device__ const T* end() const { return rawData + shape.numels; }

    inline __host__ __device__ uint64 numels() const { return shape.numels; }

    inline __host__ __device__ bool is_oob(uint32 b, uint32 y, uint32 x) const
    {
        return shape.is_oob(b, y, x);
    }

    bool is_on_device() const
    {
        cudaPointerAttributes attr;
        cudaErrCheck(cudaPointerGetAttributes(&attr, data.get()));
        return attr.type == cudaMemoryTypeDevice;
    }

    virtual ~Matrix<T>() = default;

    CudaPtr get_data() const { return data; }

    CudaPtr get_data() { return data; }

 protected:
    void set_data(CudaPtr in)
    {
        data = in;
        rawData = data.get();
    }

 private:
    CudaPtr data = CudaPtr(MatrixInitUitls::allocManaged<T>(shape, id), [this](T* ptr) {
        MatrixInitUitls::free<T>(ptr, id);
        this->rawData = nullptr;
    });
    T* rawData = data.get();

    template <typename U>
    uint64 memcpy(const U* src, uint64 offset, uint64 numels)
    {
        if (offset + numels > shape.numels)
            throw_rte_with_backtrace("Offset (", offset, ") + numels (", numels, ") invalid for ",
                                     shape);
        auto* dst = rawData + offset;
        for (uint64 i = 0; i < numels; i++) dst[i] = static_cast<T>(src[i]);
        return numels;
    }

    uint64 memcpy(const T* src, uint64 offset, uint64 numels)
    {
        if (offset + numels > shape.numels)
            throw_rte_with_backtrace("Offset (", offset, ") + numels (", numels, ") invalid for ",
                                     shape);
        cudaErrCheck(cudaMemcpy(rawData + offset, src, numels * sizeof(T), cudaMemcpyDefault));
        return numels;
    }

    void moveToDevice(int32_t device = 0)
    {
        cudaErrCheck(
            cudaMemAdvise(data.get(), shape.bytes<T>(), cudaMemAdviseSetPreferredLocation, device));
        cudaErrCheck(
            cudaMemAdvise(data.get(), shape.bytes<T>(), cudaMemAdviseSetAccessedBy, device));
    }
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const Matrix<T>& m)  // usable to paste in torch ()
{
    std::setiosflags(std::ios::fixed);
    uint32 precision = 6;
    os << ' ' << m.name << m.shape;
    if (!m.extents.all_valid()) os << "\nExtents: " << m.extents;
    os << "\t([" << std::fixed << std::setfill(' ');

    for (uint32 b = 0; b < m.batch(); b++)
    {
        os << "\n[";
        for (uint32 y = 0; y < m.height(); y++)
        {
            os << (m.height() > 1 ? "\n[" : "[");
            for (uint32 x = 0; x < m.width(); x++)
            {
                os << std::setw(precision + 5) << std::setfill(' ') << std::setprecision(precision)
                   << m(b, y, x) << (x == m.width() - 1 ? " " : ",  ");
            }
            os << ']' << (y == m.height() - 1 ? "" : ",");
        }
        os << ']' << (b == m.batch() - 1 ? "" : ",");
    }
    os << "])\n";
    return os;
}

#endif  // MATRIX_CUH
