#ifndef NODES_HPP
#define NODES_HPP

#include <cstdlib>
#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "parameter.hpp"

using FloatT = float32;

template <typename T>
struct Node
{
    virtual const Matrix<T>& forward(const Matrix<T>& x) = 0;
    virtual const Matrix<T>& backward(const Matrix<T>& e) = 0;
    virtual bool update_weights(FloatT lr)
    {
        throw std::runtime_error("update_weights not implemented for this node");
    }

    Node<T>* next = nullptr;  // for now singly linked list, not multiple fan outs.
    Node<T>* prev = nullptr;

    Node(uint32_t height, uint32_t width, Node<T>* prev) : output(height, width)
    {
        if (prev)
            {
                prev->next = this;
                this->prev = prev;
            }
    }
    Node<T>* prev_node() { return prev; }

 protected:
    Matrix<T> output;
    Matrix<T>& get_output() { return output; }
};

template <typename T, typename ActivationT = IdentityActivation<T>>
struct Linear : public Node<T>
{
    Parameter<T, T> W;
    Parameter<T, T> b;
    using Forward = typename ActivationT::forward;
    using Backward = typename ActivationT::backward;

    Linear(uint32 in, uint32 out, Node<T>* prev = nullptr)
        : Node<T>(out, 1, prev),
          W(out, in),
          b(out, 1),
          actGrad(out, 1),
          WtTranspose(in, out),
          retGradient(in, 1),
          inputT(1, in),
          actGxGradIn(out, 1)
    {
    }

    const Matrix<T>& forward(const Matrix<T>& x)
    {
        mmadd<T, Forward>(this->output, W.Weights, x, &b.Weights);
        transpose(inputT, x);  // disbale this in inference only mode.
        if (this->next) return this->next->forward(this->output);
        return this->output;
    }

    // gradientIn: gradient from "next" node, input: input to this node if "prev" node is null
    // returns gradient from "prev" node if it exists, otherwise returns gradient of this node
    const Matrix<T>& backward(const Matrix<T>& gradientIn)
    {
        //  for W gradient is:  gradientIn * backward(output) * inputT

        unary_apply(actGrad, this->get_output(), Backward());  // backward(output)

        binary_apply(actGxGradIn, gradientIn, actGrad, Mul<T>());  // gradientIn * backward(output)
        mmadd<T>(W.Grads, actGxGradIn, inputT, nullptr);  // gradientIn * backward(output) * input

        // for b gradient is: gradientIn * backward(output)
        fill(b.Grads, actGxGradIn);

        // return gradient is transpose(W) * actGxGradIn
        transpose(WtTranspose, W.Weights);
        mmadd<T>(retGradient, WtTranspose, actGxGradIn, nullptr);
        if (this->prev_node()) return this->prev_node()->backward(retGradient);
        return retGradient;
    }

    bool update_weights(FloatT lr)
    {
        W.update(lr);
        b.update(lr);
        if (this->prev_node()) return this->prev_node()->update_weights(lr);
        return true;
    }

 private:
    Matrix<T> actGrad;      // gradient of activation ( i.e. backward(output))
    Matrix<T> inputT;       // temp for x transpose
    Matrix<T> actGxGradIn;  // temp for dEdy * dydz
    Matrix<T> WtTranspose;  // temp for W transpose
    Matrix<T> retGradient;  // temp for W transpose * dEdy
};

///////////////////////////// Error Nodes ////////////////////////////////

template <typename T>
struct L2Error : Node<T>  // Mean Squared Error reduced to scalar
{
    Matrix<T> diff;         // (y - y_tartget)
    Matrix<T> sqDiff;       //  (y - y_tartget)^2
    Matrix<T> temp1d;       // output of reduction to 1d
    Matrix<T> gradientOut;  // output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L2Error(uint32_t inHeight, uint32_t inWidth, Node<T>* prev = nullptr)
        : Node<T>(1, 1, prev),
          diff(inHeight, inWidth),
          sqDiff(inHeight, inWidth),
          temp1d(inHeight, 1),
          gradientOut(inHeight, inWidth),
          times2ByNumels(2.f / diff.numels())
    {
    }

    const Matrix<T>& forward(const Matrix<T>& y)
    {
        // empty implem to avoid throw from Node<T> , call forward with target
        return y;
    }

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        binary_apply(diff, y, target, Sub<T>());
        unary_apply(sqDiff, diff, Square<T>());
        if (sqDiff.width > 1)
            {
                reduce_mean(temp1d, sqDiff);
                reduce_mean(this->get_output(), temp1d);
            }
        else
            reduce_mean(this->get_output(), sqDiff);
        if (this->next)
            return this->next->forward(this->get_output());
        return this->get_output();
    }

    const Matrix<T>& backward(const Matrix<T>&)
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, times2ByNumels);
        if (this->prev) return this->prev->backward(gradientOut);
        return gradientOut;
    }
};

template <typename T>
struct L1Error : Node<T>  // Mean Squared Error reduced to scalar
{
    Matrix<T> diff;         // (y - y_tartget)
    Matrix<T> absDiff;      // |y - y_tartget|
    Matrix<T> temp1d;       // output of reduction to 1d
    Matrix<T> gradientOut;  // output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L1Error(uint32_t inHeight, uint32_t inWidth, Node<T>* prev = nullptr)
        : Node<T>(1, 1, prev),
          diff(inHeight, inWidth),
          absDiff(inHeight, inWidth),
          temp1d(inHeight, 1),
          gradientOut(inHeight, inWidth),
          times2ByNumels(2.f / diff.numels())
    {
    }

    const Matrix<T>& forward(const Matrix<T>& y)
    {
        // empty implem to avoid throw from Node<T> , call forward with target
        return y;
    }

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        binary_apply(diff, y, target, Sub<T>());
        unary_apply(absDiff, diff, Square<T>());
        if (absDiff.width > 1)
            {
                reduce_mean(temp1d, absDiff);
                reduce_mean(this->get_output(), temp1d);
            }
        else
            reduce_mean(this->get_output(), absDiff);
        if (this->next) return this->next->forward(this->get_output());
        return this->get_output();
    }

    const Matrix<T>& backward(const Matrix<T>&)
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, times2ByNumels);
        if (this->prev) return this->prev->backward(gradientOut);
        return gradientOut;
    }
};

#endif  // NODES_HPP
