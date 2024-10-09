#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "logger.hpp"
#include "types"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <ios>
#include <string>

uint32 getMatrixId();

#define cudaErrCheck(err)                                                                          \
    {                                                                                              \
        cudaErrCheck_((err), __FILE__, __LINE__);                                                  \
    }
inline void cudaErrCheck_(cudaError_t code, const char* file, uint32 line, bool abort = true)
{
    if (code == cudaSuccess) return;
    LOG(BOLD, RED, "CUDA ERROR: ", code, ", `", cudaGetErrorString(code), "` at",
        Log::Location{file, line});
    throw std::runtime_error("CUDA ERROR");
}

template <typename T> struct Matrix
{
    const uint32 id;
    const uint32 height;
    const uint32 width;

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
        : height(height), width(width), id(getMatrixId()), data(CudaAllocator(height * width))
    {
        if (values)
        {
            cudaMemcpy(data.get(), values, height * width * sizeof(T), cudaMemcpyDefault);
        }
        moveToDevice(0);
    }

    Matrix(Matrix<T>&& m) : id(m.id), height(m.height), width(m.width), data(std::move(m.data))
    {
        m.data = nullptr;
    }

    T& operator()(uint32 x, uint32 y) // opposite of convention, but I like it
    {
        bounds_and_ptr(x, y);
        return data.get()[x + y * width];
    }

    T operator()(uint32 x, uint32 y) const
    {
        bounds_and_ptr(x, y);
        return data.get()[x + y * width];
    }

    std::string get_name() const
    {
        char name[64];
        snprintf(name, 64, "Matrix{%d}[%dx%d]@0x%lx", id, height, width, uint64_t(data.get()));
        return name;
    }

    inline T* begin() { return data.get(); }

    inline const T* begin() const { return data.get(); }

    inline T* end() { return data.get() + height * width; }

    inline const T* end() const { return data.get() + height * width; }

    uint32 numels() const { return height * width; }

  private:
    inline void bounds_and_ptr(uint32 x, uint32 y) const
    {
        if (data == nullptr)
        {
            throw std::runtime_error("Matrix data is null");
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

template <typename T> std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
{
    os << m.get_name() << std::setprecision(6) << std::fixed << std::setfill(' ');
    for (uint32 y = 0; y < m.height; y++)
    {
        os << std::endl;
        for (uint32 x = 0; x < m.width; x++)
        {
            os << std::setw(10) << m(x, y) << " ";
        }
    }
    return os;
}

typedef Matrix<float32> Matrixf;
typedef Matrix<float16> Matrixf16;

#endif // MATRIX_CUH
