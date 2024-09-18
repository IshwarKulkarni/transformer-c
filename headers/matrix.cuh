#ifndef MATRIX_CUH
#define MATRIX_CUH

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#define DEBUG 1

uint32_t getMatrixId();

#define cudaErrCheck(err) { cudaErrCheck_((err), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t code, const char *file, uint32_t line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        std::cerr << "CUDA ERROR: " << code << ", " << cudaGetErrorString(code) << " at " << file <<  ":" <<  line << std::endl;
        throw std::runtime_error("CUDA ERROR");
    } 
}

template<typename T>
struct Matrix {

    const uint32_t id;
    const uint32_t rows;
    const uint32_t cols;

    struct CudaDeleter 
    {
        void operator()(T* ptr) 
        {
            cudaErrCheck(cudaFree(ptr));
        }
    };
    
    T* CudaAllocator(size_t numElems) {
        T* ptr;
        cudaErrCheck(cudaMallocManaged(&ptr, sizeof(T) * numElems));
        return ptr;
    }

    typedef std::unique_ptr<T, CudaDeleter> CudaPtr;

    CudaPtr data;

    Matrix(uint32_t rows, uint32_t cols, const T* values=nullptr): 
        rows(rows),
        cols(cols),
        id(getMatrixId()),
        data(CudaAllocator(rows * cols))
    {
        if(values) {
            std::copy(values, values + rows * cols, data.get());
        }
    }

    T& operator()(uint32_t i, uint32_t j) {
        if (data == nullptr){
            throw std::runtime_error("Matrix data is null");
        }
        if (i >= rows || j >= cols ) {
            throw std::out_of_range("Matrix index out of range: " + std::to_string(i) + " " + std::to_string(j));
        }
        return data.get()[j * cols + i];
    }

    T operator()(uint32_t i, uint32_t j) const {
        if (data == nullptr){
            throw std::runtime_error("Matrix data is null");
        }
        if (i >= rows || j >= cols ) {
            throw std::out_of_range("Matrix index out of range: " + std::to_string(i) + " " + std::to_string(j));
        }
        return data.get()[j * cols + i];
    }

    std::string get_name() const {
        char name[64];
        snprintf(name, 64, "Matrix{%d}[%dx%d]@0x%lx", id, rows, cols, uint64_t(data.get()));
        return name;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        os << m.get_name() << "\n" << std::fixed  << std::setprecision(8) << "\t";
        for(uint32_t i = 0; i < m.rows; i++) {
            for(uint32_t j = 0; j < m.cols; j++) {
                os << m(i, j) << "  ";
            }
            os << "\n\t";
        }
        os << std::endl;
        return os;
    }
};

typedef Matrix<float> Matrixf;

#endif // MATRIX_CUH
