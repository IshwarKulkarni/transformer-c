#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <ios>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>
#include "errors.hpp"
#include "logger.hpp"
#include "types"

uint32 getMatrixId();

#define cudaErrCheck(err)                         \
    {                                             \
        cudaErrCheck_((err), __FILE__, __LINE__); \
    }
inline void cudaErrCheck_(cudaError_t code, const char* file, uint32 line, bool abort = true)
{
    if (code == cudaSuccess) return;
    LOG(BOLD, RED, "CUDA ERROR: ", code, ", `", cudaGetErrorString(code), "` at ",
        Log::Location{file, line});
    if (abort) throw runtime_error_with_backtrace("CUDA ERROR");
}

template <typename T>
struct Matrix
{
    const uint32 id;
    const uint32 height;
    const uint32 width;
    const std::string shape_str;
    const std::string name;

    struct CudaDeleter
    {
        void operator()(T* ptr) { cudaErrCheck(cudaFree(ptr)); }
    };

    T* CudaAllocator(size_t numElems)
    {
        T* ptr = nullptr;
        cudaErrCheck(cudaMallocManaged((void**)&ptr, sizeof(T) * numElems));
        return ptr;
    }

    typedef std::unique_ptr<T[], CudaDeleter> CudaPtr;

    CudaPtr data;

    Matrix(uint32 height, uint32 width, const T* values = nullptr)
        : id(getMatrixId()),
          height(height),
          width(width),
          shape_str('[' + std::to_string(height) + "x" + std::to_string(width) + ']'),
          name("Matrix-" + std::string(typeid(T).name()) + '{' + std::to_string(id) + '}' +
               shape_str),
          data(CudaAllocator(height * width))
    {
        if (values)
        {
            cudaErrCheck(
                cudaMemcpy(data.get(), values, height * width * sizeof(T), cudaMemcpyDefault));
        }
        moveToDevice(0);
    }

    Matrix(Matrix<T>&& m) : id(m.id), height(m.height), width(m.width), data(std::move(m.data))
    {
        m.data = nullptr;
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

 private:
    inline void bounds_and_ptr(uint32 y, uint32 x) const
    {
        if (data == nullptr)
        {
            throw runtime_error_with_backtrace("Matrix data is null");
        }
        if (x >= width || y >= height)
        {
            throw std::out_of_range("Matrix index out of range: " + std::to_string(x) + " " +
                                    std::to_string(y));
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
    os << m.name << std::setprecision(12) << std::fixed << std::setfill(' ') << '[';
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
