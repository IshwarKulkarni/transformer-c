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
    Matrix<TW> Weights;
    Matrix<TG> Grads;

    Parameter(uint32_t height, uint32_t width, TW* wValues = nullptr)
        : Weights(xavier_init<TW>(height, width)), Grads(Matrix<TG>(height, width))
    {
        reset_grad();
        if (wValues)
        {
            fill(Weights, wValues);
        }
    }

    inline void reset_grad()
    {
        cudaErrCheck(cudaMemset(Grads.begin(), 0, Grads.numels() * sizeof(TG)));
    }
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
        mmadd<T, Forward>(this->output, W.Weights, x, &b.Weights);
        return this->output;
    }

    void backward(const Matrix<T>& x, const Matrix<T>& dy)
    {
        mmadd<T, Backward>(W.Grads, dy, x, nullptr);
        fill(b.Grads, dy.begin());
    }

    virtual const Matrix<T>& get_output() { return output; }

  private:
    Matrix<T> output;
};

template <typename T> struct MSE
{
    Matrix<T> diff;          // holds (x - y)^2 in forward and -2(y - x) in backward
    Matrix<T> output_vec;    // output of reduction to 1D
    Matrix<T> output_scalar; // output of reduction to scalar
    bool reduceToScalar;

    MSE(uint32_t inHeight, uint32_t inWidth, bool reduceToScalar = true)
        : diff(inHeight, inWidth), output_vec(inHeight, 1), output_scalar(1, 1),
          reduceToScalar(reduceToScalar)
    {
    }

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        if (diff.width == 1)
        {
            binary_apply(output_vec, y, target, DiffSq<T>());
        }
        else
        {
            binary_apply(diff, y, target, DiffSq<T>());
            reduce_mean(output_vec, diff);
        }
        cudaErrCheck(cudaDeviceSynchronize());
        std::cout << "output_vec: " << output_vec << std::endl;
        if (reduceToScalar)
        {
            reduce_mean(output_scalar, output_vec);
            cudaErrCheck(cudaDeviceSynchronize());
            std::cout << "output_scalar: " << output_scalar << std::endl;
        }
        return get_output();
    }

    void backward(const Matrix<T>& y, const Matrix<T>& t)
    {
        binary_apply(diff, y, t, MultNeg2<T>());
    }

    const Matrix<T>& get_output() { return reduceToScalar ? output_scalar : output_vec; }
};

#endif // NODES_HPP
