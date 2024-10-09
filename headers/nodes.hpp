#ifndef NODES_HPP
#define NODES_HPP

#include "matrix.cuh"
#include "matrix_ops.cuh"
#include <cstddef>

template <typename T> struct Layer
{
    virtual void forward(const Matrix<T>& x) = 0;
    virtual void backward(const Matrix<T>& x, const Matrix<T>& dy) = 0;
    virtual const Matrix<T>& get_output() { return output; }

  protected:
    Layer(uint32_t height, uint32_t width) : output(height, width)
    {
        cudaErrCheck(cudaMemset(output.begin(), 0, output.numels() * sizeof(T)));
    }
    Matrix<T> output;
};

template <typename TW, typename TG = TW> // weight and gradient
struct Parameter
{
    void update(float32 lr)
    {
        // W -= lr * G;
    }
    Matrix<TW> W; // weight
    Matrix<TG> G; // gradient

    Parameter(uint32_t height, uint32_t width, TW* wValues = nullptr)
        : W(xavier_init<TW>(height, width)), G(Matrix<TG>(height, width))
    {
        reset_grad();
        if (wValues)
        {
            fill(W, wValues);
        }
    }

    inline void reset_grad() { cudaErrCheck(cudaMemset(G.begin(), 0, G.numels() * sizeof(TG))); }
};

template <typename T, typename Tg = T> struct Linear : public Layer<T>
{
    Parameter<T, T> W;
    Parameter<T, T> b;

    Linear(uint32 in, uint32 out, T* wValues = nullptr, T* bValues = nullptr)
        : Layer<T>(out, 1), W(out, in, wValues), b(out, 1, bValues)
    {
    }

    void forward(const Matrix<T>& x) { mmadd(this->output, W.W, x, &b.W); }

    // x is input, ey is error from the next layer
    void backward(const Matrix<T>& x, const Matrix<T>& dy)
    {
        mmadd<T, T, T, T>(W.G, dy, x, nullptr);
        fill(b.G, dy.begin());
    }
};

template <typename T, typename Tg = T> struct MSE : public Layer<T>
{
    Matrix<T> difference;

    MSE(uint32_t height, uint32_t width) : Layer<T>(height, width), difference(height, width) {}

    void forward(const Matrix<T>& y, const Matrix<T>& t)
    {
        // binary_apply<T>(difference, y, t, [](T a, T b) { return (a - b) *(a - b); });
        // Matrix<T> sq(y.height, y.width);
        // square(sq, diff);
        // sum(this->output, sq);
    }

    void backward(const Matrix<T>& y, const Matrix<T>& t)
    {
        // Matrix<T> diff(y.height, y.width);
        // sub(diff, y, t);
        // mmadd<T, T, T, T>(this->grad, diff, diff, nullptr);
    }

    Matrix<T> output;
    Matrix<Tg> grad;
};

#endif // NODES_HPP
