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
        if (prev) return prev->update_weights(lr);
        return false;
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
        char buffer[100];
        std::stringstream ss;
        ss << "Graph " << (prev and !traverse_to_init ? " (does not start from here)\n" : ":\n");
        sprintf(buffer, "Layer        |  Output | Param #  |        Shape       | Other params\n");
        ss << buffer << std::string(70, '-') << '\n';
        Node<T>* node = this;
        uint32 n_params = 0;
        uint32 nt_params = 0;
        while (node->prev and traverse_to_init)
        {
            node = node->prev;
        }
        while (node)
        {
            uint32 count = node->n_trainable_params();
            n_params += count;
            uint32 other_params = node->n_untrainable_params();
            nt_params += other_params;
            snprintf(buffer, 100, "%-12s | %-8s| % 8d | %18s | %5d\n", node->name.c_str(),
                     node->get_output().shape_str.c_str(), count, node->params_string().c_str(),
                     node->n_untrainable_params());
            ss << buffer;
            node = node->next;
        }
        ss << std::string(30, '-') << "\n  Total trainable params: " << n_params << '\n'
           << "Total untrainable params: " << nt_params << '\n'
           << std::string(30, '-') << std::endl;
        return ss.str();
    }

    virtual uint32 n_trainable_params() { return 0; }

    virtual uint32 n_untrainable_params() { return 0; }

    virtual std::string params_string() { return ""; }
};

template <typename T, typename ActivationT = IdentityActivation<T>>
struct Linear : public Node<T>
{
    Parameter<T, T> W;
    Parameter<T, T> b;
    using Forward = typename ActivationT::forward;
    using Backward = typename ActivationT::backward;

    Linear(uint32 in, uint32 out, Node<T>* prev = nullptr, const char* name = "Linear")
        : Node<T>(out, 1, prev, name),
          W(out, in, nullptr, Node<T>::name + std::string("_W")),
          b(out, 1, nullptr, Node<T>::name + std::string("_b")),
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

    virtual std::string params_string() { return W.Weights.shape_str + "+" + b.Weights.shape_str; }

    virtual uint32 n_untrainable_params()
    {
        return actGrad.numels() + WtTranspose.numels() + retGradient.numels() +
               actGxGradIn.numels() + inputT.numels();
    }

 private:
    Matrix<T> actGrad;      // gradient of activation ( i.e. backward(output))
    Matrix<T> inputT;       // temp for x transpose
    Matrix<T> actGxGradIn;  // temp for dEdy * dydz
    Matrix<T> WtTranspose;  // temp for W transpose
    Matrix<T> retGradient;  // temp for W transpose * dEdy
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
          softmaxGrad(height, height),
          gradOut(height, 1)
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
        transpose(outputT, this->output);
        mmadd(outputToutput, this->output, outputT, (const Matrix<T>*)(nullptr));
        binary_apply(softmaxGrad, outputToutput, this->output, SoftmaxGrad<T>());
        mmadd<T, Identity<T>>(gradOut, softmaxGrad, gradientIn, (const Matrix<T>*)(nullptr));

        if (this->prev) return this->prev->backward(gradOut);
        return gradOut;
    }

    virtual uint32 n_untrainable_params()
    {
        return outputT.numels() + outputToutput.numels() + softmaxGrad.numels() + gradOut.numels();
    }

    Matrix<T> outputT;        // transpose of output
    Matrix<T> outputToutput;  // outputT * output
    Matrix<T> softmaxGrad;    // gradient of softmax function
    Matrix<T> gradOut;
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

    virtual uint32 n_untrainable_params()
    {
        return diff.numels() + sqDiff.numels() + temp1d.numels() + gradientOut.numels();
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

using FloatT = float64;
typedef Node<FloatT> NodeF;
typedef Linear<FloatT, Sigmoid<FloatT>> LinearSigmoidF;
typedef Linear<FloatT, IdentityActivation<FloatT>> LinearIdentityF;
typedef L2Error<FloatT> L2ErrorF;
typedef L1Error<FloatT> L1ErrorF;
typedef Softmax<FloatT> SoftmaxF;
#endif  // NODES_HPP
