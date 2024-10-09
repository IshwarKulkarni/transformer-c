#ifndef LINEAR_HPP
#define LINEAR_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"


template<typename T>
struct Layer
{
    virtual void forward(const Matrix<T>& x) = 0;
    virtual void backward(const Matrix<T>& x, const Matrix<T>& dy) = 0;
    virtual Matrix<T>& get_output() { return output; }
    protected:
    Layer(uint32_t height, uint32_t width) : output(height, width) {}
    Matrix<T> output;
};

template <typename T> struct Linear: public Layer<T>
{
    Matrix<T> W;
    Matrix<T> b;

    Linear(uint32 in, uint32 out) :
        Layer<T>(out, 1),
        W(xavier_init<T>(out, in)), 
        b(xavier_init<T>(out, 1))
        {}

    void forward(const Matrix<T>& x)
    {
        mmadd<T>(this->output, W, x, b);;
    }

    void backward(const Matrix<T>& x, const Matrix<T>& dy)
    {
        Matrix<T> dx = mmadd<T>(W, dy, nullptr);
        Matrix<T> dW = mmadd<T>(dy, x, nullptr);
        Matrix<T> db = sum<T>(dy, 1);
    }
};

template <typename T> 
struct Summation
{
    Matrix<T> Sum;
    Summation(uint32 inheight): Sum(inheight, 1) {}

    void forward(const Matrix<T>& x)
    {
        return sum<T>(x, 0);
    }

    void backward(const Matrix<T>& x, const Matrix<T>& dy)
    {
        
    }
};



#endif // LINEAR_HPP
