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

    template <uint32 Dim>  //
    inline __host__ __device__ uint64 offset_in_dim(uint32 i, uint32 i1, uint32 i2) const
    {
        auto [b, y, x] = get_indices<Dim>(i, i1, i2);
        return offset(b, y, x);
    }

    template <unsigned int bits>
    inline __host__ __device__ uint64 broadcasting_offset(uint32 b, uint32 y, uint32 x) const
    {
        static_assert(bits <= 0b111, "dim bits must in [0b000, 0b111]");
        // clang-format off
        if (bits & WIDTH_BIT &&width == 1)   x = 0;
        if (bits & HEIGHT_BIT &&height == 1) y = 0;
        if (bits & BATCH_BIT &&batch == 1)   b = 0;
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

    template <typename T>  // alloc for matrix data
    static T* allocManaged(const Shape& shape, uint32 id)
    {
        T* ptr = nullptr;
        if (shape.numels == 0) throw_rte_with_backtrace("Cannot allocate a matrix with 0 elements");
        LOG_ALLOC("Allocating matrix ", id);
        (void)id;
        cudaErrCheck(cudaMallocManaged((void**)&ptr, shape.bytes<T>()));
        alloced_bytes += shape.numels * sizeof(T);
        if (auto it = id_to_alloced_bytes.find(id); it != id_to_alloced_bytes.end())
        {
            it->second += shape.numels * sizeof(T);
        }
        else
        {
            id_to_alloced_bytes[id] = shape.numels * sizeof(T);
        }
        return ptr;
    }

    template <typename T>  // alloc for everything else
    static T* allocManaged(uint32 length, uint32 id)
    {
        T* ptr = nullptr;
        LOG_ALLOC("Allocating memory for matrix ", id, " len: ", length);
        (void)id;
        cudaErrCheck(cudaMallocManaged((void**)&ptr, length * sizeof(T)));
        alloced_bytes += length * sizeof(T);

        if (auto it = id_to_alloced_bytes.find(id); it != id_to_alloced_bytes.end())
        {
            it->second += length * sizeof(T);
        }
        else
        {
            id_to_alloced_bytes[id] = length * sizeof(T);
        }
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

    static void print_stats()
    {
        LOG(YELLOW, "MatrixInitUitls: ", alloced_bytes, " bytes allocated, ", freed_bytes,
            " bytes freed");
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

/*
Matrix class for 3d tensors (batch, height, width) with managed memory allocation
and automatic deallocation on destruction. The data is stored in a shared pointer
that is returned by the get() method. Matrices cannot be copied, only moved.
Allows for creation with `shape`, &&vector of matrices to concatenate along the batch dimension.

Data is stored in row-major order, i.e. 0th dimension is width, 1st is height, &&2nd is batch
and increment of pointer from get() || begin() is along the width dimension.

Access is done with
    3-element () operator: batchIdx, heightIdx, widthIdx
    1-element [] operator: linear offset
    || index method: index<0>(i, m, n) is equivalent to operator()(m, n, i), i is width dim index
                     index<1>(i, m, n) is equivalent to operator()(m, i, n), i is height dim index
                     index<2>(i, m, n) is equivalent to operator()(i, m, n), i is batch dim index
*/
template <typename T>
struct Matrix
{
    const uint32 id = MatrixInitUitls::get_id();
    const std::string name;
    const Shape shape;

    typedef std::shared_ptr<T[]> CudaPtr;

    Matrix() : id(0), name("Empty"), shape(0, 0, 0), data() {}

    Matrix(Shape shape, const std::string& name_ = "Matrix")
        : name(name_ + '{' + std::to_string(id) + '}'), shape(shape)
    {
        LOG_MATRIX_CREATE(this->name, " : ", this->shape, " size: ", this->shape.numels,
                          " bytes: ", this->shape.bytes<T>());
        for (uint32 b = 0; b < shape.batch; b++)
        {
            extent<HEIGHT_IDX>(b) = shape.height;
            extent<WIDTH_IDX>(b) = shape.width;
            extent<BATCH_IDX>(b) = shape.batch;
        }
        set_val(std::numeric_limits<T>::quiet_NaN());
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
                throw_rte_with_backtrace("All matrices must have the same height &&width");
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
    // &&having corresponding dim set in `bits`, e.g. 0b001 for width, 0b010 for height, 0b100 for
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

    // return element at index i in dimension dim, using i1 &&i2 as the other indices
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

    template <uint32 Dim>
    inline __device__ __host__ void set_extent(uint32 b, uint32 val)
    {
        static_assert(Dim < BATCH_IDX, "Invalid dimension for set_extents");
        if (b >= batch()) throw_rte_with_backtrace("Batch OOB: ", b, " >= ", batch());
        if (val > shape[Dim])
            throw_rte_with_backtrace("Extent OOB: ", val, " > ", shape[Dim], " for ", this->name,
                                     this->shape);
        extent<Dim>(b) = val;
    }

    template <uint32 Dim = BATCH_IDX>  // is Dim == BATCH_IDX, then (i, i1, i2) same as (b, y, x)
    inline __device__ __host__ void set_extents(uint32 i, uint32 i1, uint32 i2)
    {
        static_assert(Dim <= BATCH_IDX, "Invalid dimension for set_extents");
        auto [b, y, x] = get_indices<Dim>(i, i1, i2);
        set_extent<WIDTH_IDX>(b, x);
        set_extent<HEIGHT_IDX>(b, y);
    }

    // return y &&x extents for batch b
    inline __device__ __host__ std::pair<uint32, uint32> get_extents(uint32 b) const
    {
        return std::make_pair(extent<HEIGHT_IDX>(b), extent<WIDTH_IDX>(b));
    }

    inline __device__ __host__ void set_extents(uint32 b, std::pair<uint32, uint32> extents)
    {
        extent<HEIGHT_IDX>(b) = extents.first;
        extent<WIDTH_IDX>(b) = extents.second;
    }

    template <uint32 Dim>
    inline __device__ __host__ uint32 get_extent(uint32 b) const
    {
        return extent<Dim>(b);
    }

    template <uint32 Dim>
    inline __device__ __host__ bool in_extent(uint32 b, uint32 i) const
    {
        return i < extent<Dim>(b);
    }

    template <uint32 Dim = BATCH_IDX>  // default Dim => interpret (i, i1, i2) as (b, y, x)
    inline __device__ __host__ bool in_extents(uint32 i, uint32 i1, uint32 i2) const
    {
        auto [b, y, x] = get_indices<Dim>(i, i1, i2);
        return in_extent<HEIGHT_IDX>(b, y) && in_extent<WIDTH_IDX>(b, x);
    }

    inline __device__ __host__ bool extents_same_as_shape() const
    {
        for (uint32 b = 0; b < batch(); b++)
        {
            if (extent<HEIGHT_IDX>(b) != height() || extent<WIDTH_IDX>(b) != width()) return false;
        }
        return true;
    }

    inline void copy_extents(const Matrix<T>& src)
    {
        if (src.shape != shape)
            throw_rte_with_backtrace(
                "Cannot copy extents from matrix with different shape: ", src.shape, " to ", shape);
        cudaMemcpy(extents_ptr, src.extents_ptr, shape.batch * 3 * sizeof(uint32),
                   cudaMemcpyDeviceToDevice);
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

    inline uint64 copy(const Matrix<T>& src, Optional<uint32> batch = {})
    {
        if (src.shape != shape)
            throw_rte_with_backtrace("Cannot copy from matrix with different shape: ", src.shape,
                                     " to ", shape);
        uint64 copied = copy(src.rawData, batch);

        cudaErrCheck(cudaMemcpy(extents_ptr, src.extents_ptr, shape.batch * 3 * sizeof(uint32),
                                cudaMemcpyDefault));

        return copied;
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

    std::shared_ptr<uint32[]> extents = std::shared_ptr<uint32[]>(
        MatrixInitUitls::allocManaged<uint32>(shape.batch * 3, id), [this](uint32* ptr) {
            MatrixInitUitls::free<uint32>(ptr, id);
            this->extents_ptr = nullptr;
        });

    uint32* extents_ptr = extents.get();

    template <uint32 Dim>
    inline __device__ __host__ uint32& extent(uint32 b)
    {
        if (b >= batch()) throw_rte_with_backtrace("Batch OOB: ", b, " >= ", batch());
        return extents_ptr[batch() * Dim + b];
    }

    template <uint32 Dim>
    inline __device__ __host__ const uint32& extent(uint32 b) const
    {
        if (b >= batch()) throw_rte_with_backtrace("Batch OOB: ", b, " >= ", batch());
        return extents_ptr[batch() * Dim + b];
    }

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
inline void print_extents(std::ostream& os, const Matrix<T>& m)
{
    if (m.extents_same_as_shape()) return;
    os << "\nExtents for " << m.name << " : ";
    for (uint32 b = 0; b < m.batch(); b++)
    {
        auto [h, w] = m.get_extents(b);
        os << " [" << h << ", " << w << "]";
    }
    os << "\n";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os,
                                const Matrix<T>& m)  // usable to paste in torch ()
{
    std::setiosflags(std::ios::fixed);
    uint32 precision = 6;
    os << ' ' << m.name << m.shape;
    print_extents(os, m);
    os << "([" << std::fixed << std::setfill(' ');

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

inline void print_param_histogram(std::ostream& os, Matrix<FloatT>* mat, uint32 num_bins = 100)
{
    auto weights = mat->get_data();
    FloatT min_val = std::numeric_limits<FloatT>::max();
    FloatT max_val = std::numeric_limits<FloatT>::min();

    // Find min and max values
    for (uint32 i = 0; i < mat->numels(); ++i)
    {
        min_val = std::min(min_val, weights[i]);
        max_val = std::max(max_val, weights[i]);
    }

    // count small values
    uint32 small_values = 0;
    float32 small_value_epsilon = 0.0001;

    // Create histogram bins
    std::vector<uint32> bin_counts(num_bins, 0);
    std::set<uint32> used_bins;
    for (uint32 i = 0; i < mat->numels(); ++i)
    {
        if (std::abs(weights[i]) < small_value_epsilon) small_values++;
        uint32 bin_idx = (weights[i] - min_val) / (max_val - min_val) * num_bins;
        bin_idx = std::min(bin_idx, num_bins - 1);  // Ensure we don't go out of bounds
        bin_counts[bin_idx]++;
        used_bins.insert(bin_idx);
    }

    if (used_bins.size() == 1)
    {
        os << "All values are in same bin: " << min_val << " .. " << max_val << " for " << mat->name
           << mat->shape << "\n";
        return;
    }

    // Print histogram
    uint32 max_count = *std::max_element(bin_counts.begin(), bin_counts.end());
    uint32 max_width = 50;  // Maximum width of histogram bars
    os << " ---------------------------------------------------------------\n";
    os << "#Small values: " << small_values << " :" << small_values * 100.0 / mat->numels() << "% ["
       << -small_value_epsilon << " , " << small_value_epsilon << "]\n";
    os << "Histogram for " << mat->name << mat->shape << ":\n";

    for (uint32 i = 0; i < num_bins; ++i)
    {
        uint32 count = bin_counts[i];
        uint32 bar_length = max_count > 0 ? (count * max_width / max_count) : 0;

        // Add at least one character if there are values in this bin
        if (count > 0 && bar_length == 0) bar_length = 1;

        FloatT bin_min = min_val + i * (max_val - min_val) / num_bins;
        FloatT bin_max = min_val + (i + 1) * (max_val - min_val) / num_bins;

        char line[120];
        snprintf(line, sizeof(line), "%2d|  %8.4f - %8.4f: %6d : %s\n", i, bin_min, bin_max, count,
                 std::string(bar_length, '>').c_str());
        os << line;
    }
}

#endif  // MATRIX_CUH
