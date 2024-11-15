#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include "errors.hpp"
#include "logger.hpp"
#include "types"

#define cudaErrCheck(err) cudaErrCheck_((err), __FILE__, __LINE__)

inline void cudaErrCheck_(cudaError_t code, const char* file, uint32 line, bool abort = true)
{
    if (code == cudaSuccess) return;
    LOG(BOLD, RED, "CUDA ERROR: ", code, ", `", cudaGetErrorString(code), "` at ",
        Log::Location{file, line});
    if (abort) throw_rte_with_backtrace("CUDA ERROR")
}

typedef struct MatrixInitUitls
{
    template <typename T>
    static uint32 get(uint32 height, uint32 width)
    {
        alloced_byes += height * width * sizeof(T);
        return id++;
    }
    static uint32 get_alloced_bytes() { return alloced_byes; }

 private:
    MatrixInitUitls() = delete;
    static uint32 id;
    static uint32 alloced_byes;
} MatrixInitUitls;

template <typename T>
struct Matrix
{
    const uint32 id;
    const uint32 height;
    const uint32 width;
    const std::string shape_str;
    const std::string name;

    T* CudaAllocator(size_t numElems)
    {
        T* ptr = nullptr;
        if (numElems == 0) throw_rte_with_backtrace("Matrix size is 0");

        cudaErrCheck(cudaMallocManaged((void**)&ptr, sizeof(T) * numElems));
        return ptr;
    }

    typedef std::shared_ptr<T[]> CudaPtr;

    CudaPtr data;

    Matrix(uint32 height, uint32 width, const std::string& name_ = "Matrix")
        : id(MatrixInitUitls::get<T>(height, width)),
          height(height),
          width(width),
          shape_str('[' + std::to_string(height) + "x" + std::to_string(width) + ']'),
          name(name_ + '{' + std::to_string(id) + '}'),
          data(CudaAllocator(height * width), [](T* ptr) { cudaErrCheck(cudaFree(ptr)); })
    {
        moveToDevice(0);
    }

    Matrix(std::pair<uint32, uint32> shape, const std::string& name_ = "Matrix")
        : Matrix(shape.first, shape.second, name_)
    {
    }

    const T& operator()(uint32 y, uint32 x) const
    {
        bounds_and_ptr(y, x);
        return data[y * width + x];
    }

    T& operator()(uint32 y, uint32 x)
    {
        bounds_and_ptr(y, x);
        return data[y * width + x];
    }

    inline T* begin() { return data.get(); }

    inline const T* begin() const { return data.get(); }

    inline T* end() { return data.get() + height * width; }

    inline const T* end() const { return data.get() + height * width; }

    uint32 numels() const { return height * width; }

    bool is_on_device() const
    {
        cudaPointerAttributes attr;
        cudaErrCheck(cudaPointerGetAttributes(&attr, data.get()));
        return attr.type == cudaMemoryTypeDevice;
    }

    std::pair<uint32, uint32> shape() const { return {height, width}; }

    std::pair<uint32, uint32> t_shape() const { return {width, height}; }

 private:
    inline void bounds_and_ptr(uint32 y, uint32 x) const
    {
        if (data == nullptr)
        {
            throw_rte_with_backtrace("Matrix data is null");
        }
        if (x >= width || y >= height)
        {
            throw_rte_with_backtrace("Matrix index out of range: x:", x, ", y:", y, " for ",
                                     shape_str);
        }
    }
    void moveToDevice(int32_t device = 0)
    {
        cudaErrCheck(cudaMemAdvise(data.get(), height * width * sizeof(T),
                                   cudaMemAdviseSetPreferredLocation, device));
        cudaErrCheck(cudaMemAdvise(data.get(), height * width * sizeof(T),
                                   cudaMemAdviseSetAccessedBy, device));
        cudaErrCheck(cudaMemPrefetchAsync(data.get(), height * width * sizeof(T), device));
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)  // usable to paste in torch ()
{
    os << m.name << m.shape_str << std::setprecision(12) << std::fixed << std::setfill(' ') << '[';
    for (uint32 y = 0; y < m.height; y++)
    {
        os << "\n[";
        for (uint32 x = 0; x < m.width; x++)
        {
            os << std::setw(15) << m(y, x) << (x == m.width - 1 ? "" : ", ");
        }
        os << ']' << (y == m.height - 1 ? "" : ",");
    }
    os << "]\n";
    return os;
}

typedef Matrix<FloatT> Matrixf;

#endif  // MATRIX_CUH
