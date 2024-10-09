#ifndef NODES_HPP
#define NODES_HPP

#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.cuh"

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

template <typename T, typename ActivationT = IdentityActivation<T>> struct Linear
{
    Parameter<T, T> W;
    Parameter<T, T> b;
    using Forward = typename ActivationT::forward;
    using Backward = typename ActivationT::backward;

    Linear(uint32 in, uint32 out, T* wValues = nullptr, T* bValues = nullptr)
        : output(out, 1), W(out, in, wValues), b(out, 1, bValues)
    {
    }

    const Matrix<T>& forward(const Matrix<T>& x)
    {
        mmadd<T, Forward>(this->output, W.W, x, &b.W);
        return this->output;
    }

    void backward(const Matrix<T>& x, const Matrix<T>& dy)
    {
        mmadd<T, Backward>(W.G, dy, x, nullptr);
        fill(b.G, dy.begin());
    }

    virtual const Matrix<T>& get_output() { return output; }

  private:
    Matrix<T> output;
};

template <typename T> struct MSE
{
    Matrix<T> squared_diff;
    Matrix<T> output_vec; // output of reduction to 1D
    Matrix<T> output_scalar;
    bool reduceTo1D;

    MSE(uint32_t inHeight, uint32_t inWidth, bool reduce = true)
        : squared_diff(inHeight, inWidth), output_vec(inHeight, 1), output_scalar(1, 1),
          reduceTo1D(reduce)
    {
    }

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        binary_apply(squared_diff, y, target, DiffSq<T>());
        reduce_mean(output_vec, squared_diff);
        if (reduceTo1D)
        {
            reduce_mean(output_scalar, output_vec);
        }
        return get_output();
    }

    void backward(const Matrix<T>& y, const Matrix<T>& t)
    {
        binary_apply(squared_diff, y, t, IntegerMultiplier<T, -2>()); // -2(y - t)
    }

    const Matrix<T>& get_output() { return reduceTo1D ? output_scalar : output_vec; }
};

#endif // NODES_HPP
