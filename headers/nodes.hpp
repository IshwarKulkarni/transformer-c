#ifndef NODES_HPP
#define NODES_HPP

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cmath>
#include <cstdlib>
#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "node_parameter.hpp"

using FloatT = float64;

#define print(x, name) print_ln(x, name, __FILE__, __LINE__)

template <typename T>
inline void print_ln(const Matrix<T>& m, const std::string& name, const char* file,
                     const int line_num)
{
    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << "  " << file << ":" << line_num << " | " << name << " :\n" << m << std::endl;
}

template <typename T, typename ActivationT = IdentityActivation<T>>
struct Linear : public Node<T>
{
    Parameter<T, T> W;
    Parameter<T, T> b;
    using Forward = typename ActivationT::forward;
    using Backward = typename ActivationT::backward;

 private:
    Matrix<T> actGrad;        // gradient of activation ( i.e. backward(output))
    Matrix<T> actGxGradIn;    // temp for dEdy * dydz
    Matrix<T> actGxGradIn1D;  // 1D version of actGxGradIn
    Matrix<T> inputT;         // temp for x transpose
    Matrix<T> retGradient1D;  // temp for W transpose * dEdy
    Matrix<T> WtTranspose;    // temp for W transpose
    bool useBias;

 public:
    // outW is default 1 => Vector transform
    Linear(uint32 in, uint32 outH, uint32 outW = 1, bool bias = true, Node<T>* prev = nullptr,
           const std::string& name_ = "Linear")
        : Node<T>(outH, outW, prev, name_),
          W(outH, in, nullptr, this->name + "_W"),
          b(outH, 1, nullptr, this->name + "_b"),
          actGrad(outH, outW),
          actGxGradIn(outH, outW),
          actGxGradIn1D(outH, 1),
          inputT(outW, in),
          retGradient1D(in, 1),  // gradient sum from all input Columns
          WtTranspose(in, outH),
          useBias(bias)
    {
    }

    const Matrix<T>* forward(const Matrix<T>* x)
    {
        const auto* bias = useBias ? &b.Weights : nullptr;
        mmadd<T, Forward>(this->output, W.Weights, *x, bias);
        transpose(inputT, *x);  // disbale this in inference only mode.
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    // gradientIn: gradient from "next" node,
    // returns gradient from "prev" node if it exists, otherwise returns gradient of this node
    void backward(const Matrix<T>* gradientIn)
    {
        //  for W gradient is:  gradientIn * backward(output) * inputT
        unary_apply(actGrad, this->output, Backward());  // backward(output)

        binary_apply(actGxGradIn, *gradientIn, actGrad, Mul<T>());  // gradientIn * backward(output)
        mmadd<T>(W.Grads, actGxGradIn, inputT, nullptr);  // gradientIn * backward(output) * input

        Matrix<T>* temp = actGxGradIn.width > 1 ? &actGxGradIn1D : &actGxGradIn;

        // for b gradient is: gradientIn * backward(output)
        if (actGxGradIn.width > 1)
        {
            reduce_mean(*temp, actGxGradIn);
            if (useBias) unary_apply(b.Grads, *temp, MultiplyBy<T>(actGxGradIn.width));
        }
        else if (useBias)
        {
            fill(b.Grads, *temp);
        }

        // return gradient is transpose(W) * actGxGradIn
        if (this->prev)
        {
            transpose(WtTranspose, W.Weights);
            mmadd<T>(retGradient1D, WtTranspose, *temp, nullptr);
            this->prev->backward(&retGradient1D);
        }
    }

    virtual uint32 n_trainable_params() { return W.Weights.numels() + b.Weights.numels(); }

    virtual std::string params_string() { return W.Weights.shape_str + "+" + b.Weights.shape_str; }

    virtual uint32 n_untrainable_params()
    {
        return actGrad.numels() + actGxGradIn.numels() + actGxGradIn1D.numels() + inputT.numels() +
               retGradient1D.numels() + WtTranspose.numels() + this->output.numels();
    }

    // std::string repr()
    //{
    //     std::stringstream ss;
    //     ss << "ActGrad " << actGrad << "\nActGxGradIn " << actGxGradIn << "\nActGxGradIn1D "
    //        << actGxGradIn1D << "\nInputT " << inputT << "\nRetGradient1D " << retGradient1D
    //        << "\nWtTranspose " << WtTranspose;
    // }
};

// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
template <typename T>
struct Softmax : Node<T>
{
    Matrix<T> gradientOut;
    Matrix<T> expT;
    Matrix<T> sumExpsT;
    Matrix<T> softmaxT;

    Softmax(uint32 height, uint32 width, Node<T>* prev = nullptr,
            const std::string& name = "Softmax")
        : Node<T>(height, width, prev, name),
          gradientOut(height, width),
          expT(width, height),
          sumExpsT(width, 1),
          softmaxT(width, height)
    {
    }

    const Matrix<T>* forward(const Matrix<T>* x)
    {
        transpose<FloatT, Exp<T>>(expT, *x);
        reduce_sum(sumExpsT, expT);
        binary_apply(softmaxT, expT, sumExpsT, Div<T>());
        transpose(this->output, softmaxT);
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward(const Matrix<T>* gradientIn)
    {
        softmax_gradient(gradientOut, softmaxT, *gradientIn);
        if (this->prev) this->prev->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params()
    {
        return gradientOut.numels() + expT.numels() + sumExpsT.numels() + softmaxT.numels() +
               this->output.numels();
    }

    std::string repr()
    {
        std::stringstream ss;
        ss << "ExpT " << expT << "\nSumExpsT " << sumExpsT << "\nSoftmaxT " << softmaxT
           << "\nGradientOut " << gradientOut;
        return ss.str();
    }
};

template <typename T>
struct Attention : Node<T>
{
    Linear<T> Wq;
    Linear<T> Wk;
    Linear<T> Wv;
    Softmax<T> softmax;

    Matrix<T> kT;
    Matrix<T> qk;
    const DividebBy<T> scaler;

    Attention(uint32 emb_size, uint32 seq_len, Node<T>* prev = nullptr,
              const std::string& name = "Attention")
        : Node<T>(emb_size, seq_len, prev, name),
          Wq(emb_size, emb_size, emb_size, false, prev, this->name + "_Wq"),
          Wk(emb_size, emb_size, emb_size, false, prev, this->name + "_Wk"),
          Wv(emb_size, emb_size, emb_size, false, prev, this->name + "_Wv"),
          qk(emb_size, seq_len),
          kT(seq_len, emb_size),
          softmax(seq_len, seq_len, prev, this->name + "_Softmax"),
          scaler(1.0 / sqrt(emb_size))
    {
    }

    const Matrix<T>* forward(const Matrix<T>* x)
    {
        const Matrix<T>* q = Wq.forward(x++);
        const Matrix<T>* k = Wk.forward(x++);
        const Matrix<T>* v = Wv.forward(x);

        transpose(kT, *k);
        mmadd<T>(qk, *q, kT, nullptr, scaler);

        const Matrix<T>* attn = softmax.forward(&qk);
        mmadd<T>(this->output, *attn, *v, nullptr);
        return &this->output;
    }

    void backward(const Matrix<T>* gradientIn) {}

    virtual uint32 n_untrainable_params()
    {
        return qk.numels() + kT.numels() + Wq.n_untrainable_params() + Wk.n_untrainable_params() +
               Wv.n_untrainable_params() + softmax.n_untrainable_params();
        +this->output.numels();
    }

    virtual uint32 n_trainable_params()
    {
        return Wq.n_trainable_params() + Wk.n_trainable_params() + Wv.n_trainable_params() +
               qk.numels() + kT.numels() + softmax.n_trainable_params() + this->output.numels();
    }
};

using FloatT = float64;
typedef Node<FloatT> NodeF;
typedef Linear<FloatT, Sigmoid<FloatT>> LinearSigmoidF;
typedef Linear<FloatT, IdentityActivation<FloatT>> LinearIdentityF;
typedef Softmax<FloatT> SoftmaxF;
#endif  // NODES_HPP
