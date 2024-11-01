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

#define print(x, name) print_ln(x, name, __FILE__, __LINE__)

template <typename T>
inline void print_ln(const Matrix<T>& m, const std::string& name, const char* file,
                     const int line_num)
{
    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << "  " << file << ":" << line_num << " | " << name << " :\n" << m << std::endl;
}

template <typename T, typename ActivationT = IdentityActivation<T>>
struct FullyConnected : public Node<T>
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
    FullyConnected(uint32 inH, uint32 outH, uint32 outW = 1, bool bias = true,
                   Node<T>* prev = nullptr, const std::string& name_ = "Linear")
        : Node<T>(outH, outW, prev, name_),
          W(outH, inH, nullptr, this->name + "_W"),
          b(outH, 1, nullptr, this->name + "_b"),
          actGrad(outH, outW),
          actGxGradIn(outH, outW),
          actGxGradIn1D(outH, 1),
          inputT(outW, inH),
          retGradient1D(inH, 1),  // gradient sum from all input Columns
          WtTranspose(inH, outH),
          useBias(bias)
    {
    }

    const Matrix<T>* forward(const Matrix<T>* x)
    {
        const auto* bias = useBias ? &b : nullptr;
        mmadd<T, Forward>(this->output, W, *x, bias);
        transpose(inputT, *x);  // disbale this in inference only mode.
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    // gradientIn: gradient from "next" node,
    // returns gradient from "prev" node if it exists, otherwise returns gradient of this node
    void backward(const Matrix<T>* gradientIn)
    {
        //  for W gradient is:  gradientIn * backward(output) * inputT

        Matrix<T>* temp = &this->output;
        if (!std::is_same<Backward, Identity<T>>::value)
        {
            unary_apply(actGrad, this->output, Backward());  // backward(output)
            temp = &actGrad;
        }

        binary_apply(actGxGradIn, *gradientIn, *temp, Mul<T>());  // gradientIn * backward(output)
        mmadd<T>(W.grads, actGxGradIn, inputT, nullptr);  // gradientIn * backward(output) * input

        temp = actGxGradIn.width > 1 ? &actGxGradIn1D : &actGxGradIn;

        // for b gradient is: gradientIn * backward(output)
        if (actGxGradIn.width > 1)
        {
            reduce_mean(*temp, actGxGradIn);
            if (useBias) unary_apply(b.grads, *temp, MultiplyBy<T>(actGxGradIn.width));
        }
        else if (useBias)
        {
            fill(b.grads, *temp);
        }

        // return gradient is transpose(W) * actGxGradIn
        if (this->prev)
        {
            transpose(WtTranspose, W);
            mmadd<T>(retGradient1D, WtTranspose, *temp, nullptr);
            this->prev->backward(&retGradient1D);
        }
    }

    virtual uint32 n_trainable_params() { return W.numels() + b.numels(); }

    virtual std::string params_string() { return W.shape_str + "+" + b.shape_str; }

    virtual uint32 n_untrainable_params()
    {
        return actGrad.numels() + actGxGradIn.numels() + actGxGradIn1D.numels() + inputT.numels() +
               retGradient1D.numels() + WtTranspose.numels() + this->output.numels();
    }

    std::string repr()
    {
        std::stringstream ss;
        ss << "ActGrad " << actGrad << "\nActGxGradIn " << actGxGradIn << "\nActGxGradIn1D "
           << actGxGradIn1D << "\nInputT " << inputT << "\nRetGradient1D " << retGradient1D
           << "\nWtTranspose " << WtTranspose;
        return ss.str();
    }
};

// https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
template <typename T>
struct SoftmaxDim0 : Node<T>
{
    Matrix<T> gradientOut;
    Matrix<T> exp;
    Matrix<T> sumExps;
    Matrix<T> softmax;

    SoftmaxDim0(uint32 height, uint32 width, Node<T>* prev = nullptr,
                const std::string& name = "SoftmaxDim0")
        : Node<T>(height, width, prev, name),
          gradientOut(height, width),
          exp(width, height),
          sumExps(width, 1),
          softmax(width, height)
    {
    }

    // computes softmax along width of x => output rows sum to 1
    const Matrix<T>* forward(const Matrix<T>* x)
    {
        transpose<FloatT, Exp<T>>(exp, *x);
        reduce_sum(sumExps, exp);
        binary_apply(softmax, exp, sumExps, Div<T>());
        transpose(this->output, softmax);
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward(const Matrix<T>* gradientIn)
    {
        softmax_gradient(gradientOut, softmax, *gradientIn);
        if (this->prev) this->prev->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params()
    {
        return gradientOut.numels() + exp.numels() + sumExps.numels() + softmax.numels() +
               this->output.numels();
    }

    std::string repr()
    {
        std::stringstream ss;
        ss << "ExpT " << exp << "\nSumExps " << sumExps << "\nSoftmaxT " << softmax
           << "\nGradientOut " << gradientOut;
        return ss.str();
    }
};

template <typename T>
struct SoftmaxDim1 : Node<T>
{
    Matrix<T> gradientOut;
    Matrix<T> exp;
    Matrix<T> sumExps;
    Matrix<T> xT;

    SoftmaxDim1(uint32 height, uint32 width, Node<T>* prev = nullptr,
                const std::string& name = "SoftmaxDim0")
        : Node<T>(height, width, prev, name),
          gradientOut(height, width),
          exp(width, height),
          sumExps(width, 1),
          xT(width, height)
    {
    }

    // computes softmax along height of x => output columns sum to 1
    const Matrix<T>* forward(const Matrix<T>* x)
    {
        unary_apply(exp, *x, Exp<T>());
        reduce_sum(sumExps, exp);
        binary_apply(this->output, exp, sumExps, Div<T>());
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward(const Matrix<T>* gradientIn)
    {
        softmax_gradient(gradientOut, this->output, *gradientIn);
        if (this->prev) this->prev->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params()
    {
        return gradientOut.numels() + exp.numels() + sumExps.numels() + xT.numels() +
               this->output.numels();
    }

    std::string repr()
    {
        std::stringstream ss;
        ss << "Exp " << exp << "\nSumExps " << sumExps << "\nSoftmax Output" << this->output
           << "\nGradientOut " << gradientOut;
        return ss.str();
    }
};

template <typename T>
struct Linear : Node<T>
{
    Parameter<T, T> W;
    Matrix<T> xT;

    // out_width is number of output columns --> seq length
    Linear(uint32 emb_size, uint32 seq_len, Node<T>* prev = nullptr,
           const std::string& name = "MMProduct")
        : Node<T>(seq_len, seq_len, prev, name), W(seq_len, emb_size), xT(seq_len, emb_size)
    {
    }

    const Matrix<T>* forward(const Matrix<T>* x)
    {
        multiply(this->output, W, *x);
        transpose(xT, *x);  // disable this in inference only mode.
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward(const Matrix<T>* gradientIn)
    {
        multiply(W.grads, *gradientIn, xT);
        if (this->prev) this->prev->backward(&W.grads);
    }

    bool update_weights(FloatT lr)
    {
        W.update(lr);
        if (this->prev) return this->prev->update_weights(lr);
        return true;
    }

    virtual uint32 n_trainable_params() { return W.numels(); }

    virtual std::string params_string() { return W.shape_str; }

    virtual uint32 n_untrainable_params() { return xT.numels(); }
};

template <typename T, typename PostProcess>
struct Product : Node<T>
{
    Matrix<T> aT, bT;
    Node<T>*prevA, prevB;
    Matrix<T> a_grad_in, b_grad_in;
    PostProcess pProcess;

    Product(uint32 heightA, uint32 width, uint32 heightB, PostProcess pProcess,
            Node<T>** prev = nullptr, const std::string& name = "Product")
        : Node<T>(heightA, heightB, prev, name),
          aT(width, heightA),
          bT(width, heightB),
          prevA(prev[0]),
          prevB(prev[1]),
          a_grad_in(heightA, width),
          b_grad_in(heightB, width),
          pProcess(pProcess)
    {
    }

    const Matrix<T>* forward(const Matrix<T>* x)
    {
        auto a = x;
        auto b = x + 1;
        transpose(aT, *a);  // disable this in inference only mode.
        transpose(bT, *b);
        mmadd(this->output, *a, bT, (Matrix<T>*)nullptr, pProcess);
    }

    void backward(const Matrix<T>* gradientIn)
    {
        mmadd(a_grad_in, *gradientIn, bT, (Matrix<T>*)nullptr, pProcess);
        mmadd(b_grad_in, aT, *gradientIn, (Matrix<T>*)nullptr, pProcess);
        if (prevA) prevA->backward(&a_grad_in);
        if (prevB) prevB->backward(&b_grad_in);
    }

    virtual uint32 n_untrainable_params() { return aT.numels() + bT.numels(); }

    std::string repr() { return ""; }
};

template <typename T>
struct Attention : Node<T>
{
    Linear<T> Q, K, V;
    SoftmaxDim1<T> softmax;
    Matrix<T> s_grad_out;  // gradientT of softmax as we use the transpose of softmax, see `forward`
    Matrix<T> kT, qkT, vT;
    Matrix<T> v_grad_in, s_grad_in, q_grad_in, k_grad_in, k_grad_inT, softmaxT;
    DividebBy<T> scalerF;

    Attention(uint32 emb_size, uint32 seq_len, Node<T>** qkv_prev = nullptr,
              const std::string& name = "Attention")
        : Node<T>(seq_len, seq_len, nullptr, name),
          Q(emb_size, seq_len, qkv_prev[0], name + "_Q"),
          K(emb_size, seq_len, qkv_prev[1], name + "_K"),
          V(emb_size, seq_len, qkv_prev[2], name + "_V"),
          softmax(seq_len, seq_len, nullptr, name + "_Softmax"),
          s_grad_out(seq_len, seq_len),
          kT(seq_len, seq_len),
          qkT(seq_len, seq_len),
          vT(seq_len, seq_len),
          v_grad_in(seq_len, seq_len),
          s_grad_in(seq_len, seq_len),
          q_grad_in(seq_len, seq_len),
          k_grad_in(seq_len, seq_len),
          k_grad_inT(seq_len, seq_len),
          softmaxT(seq_len, seq_len),
          scalerF(sqrt(emb_size))
    {
    }

    const Matrix<T>* forward(const Matrix<T>* qkv)
    {
        auto q = Q.forward(&qkv[0]);
        auto k = K.forward(&qkv[1]);
        auto v = V.forward(&qkv[2]);

        transpose(kT, *k);
        mmadd(qkT, *q, kT, (Matrix<T>*)nullptr, scalerF);
        softmax.forward(&qkT);
        multiply(this->output, softmax.output, *v);

        transpose(vT, *v);  // disable this in inference only mode.
        if (this->next) return this->next->forward(&this->output);
        return &this->output;
    }

    void backward(const Matrix<T>* gradientIn)
    {
        transpose(softmaxT, softmax.output);
        multiply(v_grad_in, softmaxT, *gradientIn);
        multiply(s_grad_in, *gradientIn, vT);

        softmax.backward(&s_grad_in);
        transpose(s_grad_out, softmax.gradientOut);
        V.backward(&v_grad_in);

        mmadd(k_grad_in, softmax.gradientOut, Q.output, (Matrix<T>*)(nullptr), scalerF);
        K.backward(&k_grad_in);

        mmadd(q_grad_in, s_grad_out, K.output, (Matrix<T>*)(nullptr), scalerF);
        Q.backward(&q_grad_in);
    }
};
typedef Node<FloatT> NodeF;
typedef FullyConnected<FloatT, Sigmoid<FloatT>> LinearSigmoidF;
typedef FullyConnected<FloatT, IdentityActivation<FloatT>> LinearIdentityF;
typedef SoftmaxDim0<FloatT> SoftmaxDim0F;
#endif  // NODES_HPP
