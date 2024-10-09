#ifndef NODES_HPP
#define NODES_HPP

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "parameter.hpp"

using FloatT = float64;

#define print(x, name) print_ln(x, name, __FILE__, __LINE__)

template <typename T>
inline void print_ln(const Matrix<T>& m, const std::string& name, const char* file,
                     const int line_num)
{
    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << "  " << file << ":" << line_num << " | " << name << " :\n" << m << std::endl;
}

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
            if (prev->next)
                throw std::runtime_error(
                    "Node already has a next node, only singly linked nodes are supported");
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
        sprintf(buffer,
                "Layer                |  Output | Param #  |        Shape       | Other params\n");
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
            snprintf(buffer, 100, "%-s | %-8s| % 8d | %18s | %5d\n", node->name.c_str(),
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

 private:
    Matrix<T> actGrad;        // gradient of activation ( i.e. backward(output))
    Matrix<T> inputT;         // temp for x transpose
    Matrix<T> actGxGradIn;    // temp for dEdy * dydz
    Matrix<T> WtTranspose;    // temp for W transpose
    Matrix<T> retGradient1D;  // temp for W transpose * dEdy

 public:
    // outW is default 1 => Vector transform
    Linear(uint32 in, uint32 outH, uint32 outW = 1, Node<T>* prev = nullptr,
           const char* name_ = "Linear")
        : Node<T>(outH, outW, prev, name_),
          W(outH, in, nullptr, this->name + "_W"),
          b(outH, 1, nullptr, this->name + "_b"),
          actGrad(outH, outW),
          inputT(outW, in),
          actGxGradIn(outH, outW),
          WtTranspose(in, outH),
          retGradient1D(in, 1)  // gradient sum from all input Columns
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
        if (actGxGradIn.width > 1)
        {
            reduce_sum(b.Grads, actGxGradIn);
        }
        else
            fill(b.Grads, actGxGradIn);

        // return gradient is transpose(W) * actGxGradIn
        if (this->prev_node())
        {
            transpose(WtTranspose, W.Weights);
            mmadd<T>(retGradient1D, WtTranspose, b.Grads, nullptr);
            return this->prev_node()->backward(retGradient1D);
        }
        return retGradient1D;
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
        return actGrad.numels() + WtTranspose.numels() + retGradient1D.numels() +
               actGxGradIn.numels() + inputT.numels();
    }
};

template <
    typename T>  // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
struct Softmax : Node<T>
{
    Matrix<T> exps;           // storage for exps
    Matrix<T> temp1d;         // storage to reduce sum of exps to 1d
    Matrix<T> temp0d;         // storage for sum of temp1d
    Matrix<T> outputT;        // transpose of output
    Matrix<T> outputToutput;  // outputT * output
    Matrix<T> softmaxGrad;    // gradient of softmax function
    Matrix<T> gradOut;
    Matrix<T> gradOut1D;
    Matrix<T> output1d;
    bool reduce_to_0d;

    Softmax(uint32 height, uint32 width, bool reduce_to_scalar, Node<T>* prev = nullptr,
            const char* name = "Softmax")
        : Node<T>(height, width, prev, name),
          output1d(height, 1),
          exps(height, width),
          temp1d(height, 1),
          temp0d(1, 1),
          outputT(1, height),
          outputToutput(height, height),
          softmaxGrad(height, height),
          gradOut(height, width),
          gradOut1D(height, 1),
          reduce_to_0d(reduce_to_scalar)
    {
    }

    const Matrix<T>& forward(const Matrix<T>& x)
    {
        unary_apply(exps, x, Exp<T>());
        if (reduce_to_0d)
        {
            if (exps.width > 1)
            {
                reduce_sum(temp1d, exps);
                reduce_sum(temp0d, temp1d);
            }
            else
            {
                reduce_sum(temp0d, exps);
            }
            binary_apply(this->output, exps, temp0d, Div<T>());
            if (this->next) return this->next->forward(this->output);
            return this->output;
        }
        else
        {
            if (exps.width > 1)
            {
                reduce_sum(temp1d, exps);
                binary_apply(this->output, exps, temp1d, Div<T>());
                if (this->next) return this->next->forward(this->output);
                return this->output;
            }
            else
            {
                fill(this->output, exps);
                if (this->next) return this->next->forward(this->output);
                return exps;
            }
        }
    }

    const Matrix<T>& backward(const Matrix<T>& gradientIn)
    {
        if (this->output.width > 1)
            reduce_sum(output1d, this->output);
        else
            fill(output1d, this->output);

        transpose(outputT, output1d);
        mmadd<T>(outputToutput, output1d, outputT, nullptr);
        binary_apply(softmaxGrad, outputToutput, output1d, SoftmaxGrad<T>());
        mmadd<T>(gradOut, softmaxGrad, gradientIn, nullptr);
        if (gradOut.width > 1)
        {
            // reduce_mean(gradOut1D, gradOut);
            reduce(gradOut1D, gradOut, Plus<T>(), T(0),
                   DividebBy<T>(gradOut.width * gradOut.width));
            if (this->prev) return this->prev->backward(gradOut1D);
            return gradOut1D;
        }
        if (this->prev) return this->prev->backward(gradOut);
        return gradOut;
    }

    virtual uint32 n_untrainable_params()
    {
        return outputT.numels() + outputToutput.numels() + softmaxGrad.numels() + gradOut.numels();
    }
};

///////////////////////////// Error Nodes ////////////////////////////////
template <typename T>
struct Loss : Node<T>
{
    Loss(uint32 height, uint32 width, Node<T>* prev = nullptr, const char* name = "Loss")
        : Node<T>(height, width, prev, name)
    {
    }

    virtual const Matrix<T>& forward(const Matrix<T>& y) { return y; }

    virtual const Matrix<T>& backward(const Matrix<T>& e)
    {
        throw std::runtime_error(
            "This is loss class, no-input version of backward should be called");
    }

    virtual const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target) = 0;
    virtual const Matrix<T>& backward() = 0;
};

template <typename T>
struct L2Loss : Loss<T>  // L-n loss computes (Y^ - Y)^N
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^N
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

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        binary_apply(diff, y, target, Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        if (nDiff.width > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(this->get_output(), temp1d);
        }
        else
            reduce_mean(this->get_output(), nDiff);
        if (this->next) return this->next->forward(this->get_output());
        return this->get_output();
    }

    const Matrix<T>& backward()
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, times2ByNumels);
        if (this->prev) return this->prev->backward(gradientOut);
        return gradientOut;
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

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        binary_apply(diff, y, target, Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        if (nDiff.width > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(this->get_output(), temp1d);
        }
        else
            reduce_mean(this->get_output(), nDiff);
        if (this->next) return this->next->forward(this->get_output());
        return this->get_output();
    }

    const Matrix<T>& backward()
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, Sign<T>{1. / diff.numels()});
        if (this->prev) return this->prev->backward(gradientOut);
        return gradientOut;
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

    const Matrix<T>& forward(const Matrix<T>& y, const Matrix<T>& target)
    {
        binary_apply(ce, target, y, CrossEntropy<T>());
        if (ce.width > 1)
        {
            reduce_sum(temp1d, ce);
            reduce_sum(this->output, temp1d);
        }
        else
            reduce_sum(this->output, ce);
        binary_apply(gradientOut, target, y, NegDiv<T>());  // for backward
        if (this->next) return this->next->forward(this->output);
        return this->output;
    }

    const Matrix<T>& backward()
    {
        if (this->prev) return this->prev->backward(gradientOut);
        return gradientOut;
    }

    virtual uint32 n_untrainable_params()
    {
        return tOverY.numels() + gradientOut.numels() + ce.numels() + temp1d.numels();
    }
};

using FloatT = float64;
typedef Node<FloatT> NodeF;
typedef Linear<FloatT, Sigmoid<FloatT>> LinearSigmoidF;
typedef Linear<FloatT, IdentityActivation<FloatT>> LinearIdentityF;
typedef L2Loss<FloatT> L2ErrorF;
typedef Softmax<FloatT> SoftmaxF;
#endif  // NODES_HPP
