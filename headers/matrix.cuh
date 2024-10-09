#ifndef MATRIX_CUH
#define MATRIX_CUH

#include "types"
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <memory>
#include <string>
#include "logger.hpp"

uint32_t getMatrixId();

#define cudaErrCheck(err)                                                                          \
    {                                                                                              \
        cudaErrCheck_((err), __FILE__, __LINE__);                                                  \
    }
inline void cudaErrCheck_(cudaError_t code, const char* file, uint32_t line, bool abort = true)
{
    if (code == cudaSuccess)
        return;
    LOG(BOLD, RED, "CUDA ERROR: ", code, ", `", cudaGetErrorString(code), "` at", Log::Location{file, line});
    throw std::runtime_error("CUDA ERROR");
}

template <typename T> struct Matrix
{
    const uint32_t id;
    const uint32_t height;
    const uint32_t width;

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

    Matrix(uint32_t height, uint32_t width, const T* values = nullptr)
        : height(height), width(width), id(getMatrixId()), data(CudaAllocator(height * width))
    {
        if (values)
        {
            std::copy(values, values + height * width, data.get());
        }
    }

    Matrix(Matrix<T>&& m) : id(m.id), height(m.height), width(m.width), data(std::move(m.data)) {}

    T& operator()(uint32_t x, uint32_t y)
    {
        bounds_and_ptr(x, y);
        return data.get()[x + y * width];
    }

    T operator()(uint32_t x, uint32_t y) const
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

    uint32_t numels() const { return height * width; }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m)
    {
        os << m.get_name() << "\n" << std::setfill(' ');
        for (uint32_t i = 0; i < m.height; i++)
        {
            for (uint32_t j = 0; j < m.width; j++)
            {
                typename AccumT<T>::type val = m(i, j);
                os << std::setw(10) << std::setprecision(7) << std::fixed << std::setfill(' ')
                   << val << ' ';
            }
            os << "\n";
        }
        os << std::endl;
        return os;
    }

  private:
    inline void bounds_and_ptr(uint32_t x, uint32_t y) const
    {
        if (data == nullptr)
        {
            throw std::runtime_error("Matrix data is null");
        }
        if (x >= height || y >= width)
        {
            throw std::out_of_range("Matrix index out of range: " + std::to_string(x) + " " +
                                    std::to_string(y));
        }
    }
};

typedef Matrix<float32> Matrixf;
typedef Matrix<float16> Matrixf16;

#endif // MATRIX_CUH
