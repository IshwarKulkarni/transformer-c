#ifndef ERROR_NODES_HPP
#define ERROR_NODES_HPP

#include "nodes.hpp"

///////////////////////////// Error Nodes ////////////////////////////////
template <typename T>
struct Loss : Node<T>
{
    Loss(uint32 height, uint32 width, Node<T>* prev = nullptr, const char* name = "Loss")
        : Node<T>(height, width, prev, name)
    {
    }

    virtual const Matrix<T>* forward(const Matrix<T>* y) { return y; }

    virtual void backward(const Matrix<T>*)
    {
        throw std::runtime_error(
            "This is loss class, no-input version of backward should be called");
    }

    virtual const Matrix<T>* forward(const Matrix<T>* y, const Matrix<T>* target) = 0;
    virtual void backward() = 0;
};

template <typename T>
struct L2Loss : Loss<T>
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^2
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L2Loss(uint32_t inHeight, uint32_t inWidth, Node<T>* prev = nullptr,
           const char* name = "L2Loss")
        : Loss<T>(1, 1, prev, name),
          diff(inHeight, inWidth),
          nDiff(inHeight, inWidth),
          temp1d(inHeight, 1),
          gradientOut(inHeight, inWidth),
          times2ByNumels(FloatT(2.) / (diff.numels()))
    {
    }

    const Matrix<T>* forward(const Matrix<T>* y, const Matrix<T>* target)
    {
        binary_apply(diff, *y, *target, Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        if (nDiff.width > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(this->output, temp1d);
        }
        else
            reduce_mean(this->output, nDiff);
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward()
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, times2ByNumels);
        if (this->prev) this->prev->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params()
    {
        return diff.numels() + nDiff.numels() + temp1d.numels() + gradientOut.numels();
    }
};

template <typename T>
struct L1Loss : Loss<T>  // L-n loss computes (Y^ - Y)^N
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^N
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> timesNByNumels;

    L1Loss(uint32_t inHeight, uint32_t inWidth, Node<T>* prev = nullptr,
           const char* name = "L1Loss")
        : Loss<T>(1, 1, prev, name),
          diff(inHeight, inWidth),
          nDiff(inHeight, inWidth),
          temp1d(inHeight, 1),
          gradientOut(inHeight, inWidth),
          timesNByNumels(FloatT(1.) / (diff.numels()))
    {
    }

    const Matrix<T>* forward(const Matrix<T>* y, const Matrix<T>* target)
    {
        binary_apply(diff, *y, *target, Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        if (nDiff.width > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(this->output, temp1d);
        }
        else
            reduce_mean(this->output, nDiff);
        if (this->next) this->next->forward(&this->output);
        return &this->output;
    }

    void backward()
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, Sign<T>{FloatT(1) / diff.numels()});
        if (this->prev) this->prev->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params()
    {
        return diff.numels() + nDiff.numels() + temp1d.numels() + gradientOut.numels();
    }
};

template <typename T>
struct CrossEntropyLoss : Loss<T>
{
    Matrix<T> tOverY;
    Matrix<T> gradientOut;
    Matrix<T> ce;  // per element cross entropy
    Matrix<T> temp1d;

    CrossEntropyLoss(uint32 height, uint32 width, Node<T>* prev = nullptr)
        : Loss<T>(1, 1, prev, "CrossEntropyLoss"),
          tOverY(height, width),
          gradientOut(height, width),
          ce(height, width),
          temp1d(height, 1)
    {
    }

    const Matrix<T>* forward(const Matrix<T>* y, const Matrix<T>* target)
    {
        binary_apply(ce, *target, *y, CrossEntropy<T>());
        if (ce.width > 1)
        {
            reduce_sum(temp1d, ce);
            reduce_sum(this->output, temp1d);
        }
        else
            reduce_sum(this->output, ce);
        binary_apply(gradientOut, *target, *y, NegDiv<T>());  // for backward
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward()
    {
        if (this->prev) this->prev->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params()
    {
        return tOverY.numels() + gradientOut.numels() + ce.numels() + temp1d.numels();
    }
};

typedef L2Loss<FloatT> L2ErrorF;

#endif  // ERROR_NODES_HPP