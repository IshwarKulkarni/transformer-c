#ifndef NODES_HPP
#define NODES_HPP

#include <cstdlib>
#include <sstream>
#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "parameter.hpp"

using FloatT = float64;

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
    const std::string name;

    Node(uint32_t height, uint32_t width, Node<T>* prev, std::string name = "")
        : output(height, width), name(name + "_" + get_layer_num(prev))
    {
        if (prev)
        {
            prev->next = this;
            this->prev = prev;
        }
    }
    Node<T>* prev_node() { return prev; }

    Matrix<T> output;
    Matrix<T>& get_output() { return output; }

    static std::string get_layer_num(Node<T>* node)
    {
        uint32 layer_num = 0;
        while (node)
        {
            node = node->prev_node();
            layer_num++;
        }
        return std::to_string(layer_num);
    }

    virtual std::string graph_rep(bool traverse_to_init = true)
    {
        std::stringstream ss;
        ss << "Graph " << (prev and !traverse_to_init ? "does not start from here\n" : ":\n");
        ss << std::right << std::setw(15) << "Layer"
           << " -> "
           << " Output "
           << " | "
           << "Params\n\t" << std::string(40, '-') << '\n';
        Node<T>* node = this;
        uint32 params_count = 0;
        while (node->prev and traverse_to_init)
        {
            node = node->prev;
        }
        while (node)
        {
            uint32 count = node->n_trainable_params();
            params_count += count;
            ss << std::right << std::setw(15) << node->name << " -> " << std::setw(8)
               << node->get_output().shape_str << " | "
               << (count ? std::to_string(count) + " : " + node->params_string() : "") << "\n";
            node = node->next;
        }
        ss << std::string(20, '-') << "\nTotal params: " << params_count << '\n'
           << std::string(20, '-') << std::endl;
        return ss.str();
    }

    virtual uint32 n_trainable_params() { return 0; }

    virtual std::string params_string() { return " null "; }
};

template <typename T, typename ActivationT = IdentityActivation<T>>
struct Linear : public Node<T>
{
    Parameter<T, T> W;
    Parameter<T, T> b;
    using Forward = typename ActivationT::forward;
    using Backward = typename ActivationT::backward;
    const std::string name;

    Linear(uint32 in, uint32 out, Node<T>* prev = nullptr, const char* name = "Linear")
        : Node<T>(out, 1, prev, name),
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

    // gradientIn: gradient from "next" node,
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

    virtual uint32 n_trainable_params() { return W.Weights.numels() + b.Weights.numels(); }

    virtual std::string params_string()
    {
        return W.Weights.shape_str + " + " + b.Weights.shape_str;
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
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> sqDiff;       // storage for (y - y_tartget)^2
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L2Error(uint32_t inHeight, uint32_t inWidth, Node<T>* prev = nullptr,
            const char* name = "L2Error")
        : Node<T>(1, 1, prev, name),
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

template <typename T>
struct L1Error : Node<T>  // Mean Squared Error reduced to scalar
{
    Matrix<T> diff;         // (y - y_tartget)
    Matrix<T> absDiff;      // |y - y_tartget|
    Matrix<T> temp1d;       // output of reduction to 1d
    Matrix<T> gradientOut;  // output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L1Error(uint32_t inHeight, uint32_t inWidth, Node<T>* prev = nullptr)
        : Node<T>(1, 1, prev, "L1Error"),
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

template <typename T>
struct Softmax : Node<T>
{
    Matrix<T> exps;    // storage for exps
    Matrix<T> temp1d;  // storage to reduce sum of exps to 1d
    Matrix<T> temp0d;  // storage for sum of temp1d

    Softmax(uint32 height, uint32 width, Node<T>* prev = nullptr, const char* name = "Softmax")
        : Node<T>(height, width, prev, name),
          exps(height, width),
          temp1d(height, 1),
          temp0d(1, 1),
          outputT(width, height),
          outputToutput(height, height),
          softmaxGrad(height, height)
    {
    }

    const Matrix<T>& forward(const Matrix<T>& x)
    {
        unary_apply(exps, x, Exp<T>());
        if (exps.width > 1)
        {
            reduce_sum(temp1d, exps);
            reduce_sum(temp0d, temp1d);
        }
        else
            reduce_sum(temp0d, exps);
        binary_apply(this->output, exps, temp0d, Div<T>());
        if (this->next) return this->next->forward(this->output);
        return this->output;
    }

    const Matrix<T>& backward(const Matrix<T>& gradientIn)
    {
        // softmax gradient is: softmax * (1 - softmax) * gradientIn
        cudaErrCheck(cudaDeviceSynchronize());
        std::cout << "softmax grad in: " << gradientIn << std::endl;
        transpose(outputT, this->output);
        mmadd<T>(outputToutput, this->output, outputT, nullptr);
        binary_apply(softmaxGrad, outputToutput, this->output, SoftmaxGrad<T>());

        if (this->prev) return this->prev->backward(softmaxGrad);
        return softmaxGrad;
    }

    Matrix<T> outputToutput;
    Matrix<T> outputT;
    Matrix<T> softmaxGrad;
};

using FloatT = float64;
typedef Node<FloatT> NodeF;
typedef Linear<FloatT, Sigmoid<FloatT>> LinearSigmoidF;
typedef Linear<FloatT, IdentityActivation<FloatT>> LinearIdentityF;
typedef L2Error<FloatT> L2ErrorF;
typedef L1Error<FloatT> L1ErrorF;
typedef Softmax<FloatT> SoftmaxF;
#endif  // NODES_HPP
