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

template <typename T = FloatT>
inline void print_ln(const Matrix<T>& m, const std::string& name, const char* file,
                     const int line_num)
{
    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << "  " << file << ":" << line_num << " | " << name << " :\n" << m << std::endl;
}

/*
Implements a Fully connected layer with activation, that takes `outW` column vectors of size
as inputs and produces `outW` column vectors as output, each of size `outH`;
Here's an equivalent python code:
def Linear(x, W, b, Activation_):
    assert x.shape[1] == W.shape[0]
    return Activatation_(torch.mm(x, W) + b)
*/
template <typename T = FloatT, typename ActivationT = IdentityActivation<T>>
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
    // outW is default 1 => Vector transform, > 1 is batch of input vectors
    FullyConnected(uint32 inH, uint32 outH, uint32 outW, NodePtrs<T> inputs, bool bias = true,
                   const std::string& name_ = "Linear")
        : Node<T>(outH, outW, inputs, name_, 1),
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
        this->params.push_back(&W);
        this->params.push_back(&b);
    }

    void compute() override
    {
        Matrix<T>& x = this->prev(0);
        const auto* bias = useBias ? &b : nullptr;
        mmadd<T, Forward>(*this, W, x, bias);
        transpose(inputT, x);  // disbale this in inference only mode.
    }

    // gradientIn: gradient from "next" node,
    // returns gradient from "prev" node if it exists, otherwise returns gradient of this node
    void backward(const Matrix<T>* gradientIn) override
    {
        //  for W gradient is:  gradientIn * backward(output) * inputT
        Matrix<T>* temp = this;
        if (!std::is_same<Backward, Identity<T>>::value)
        {
            unary_apply(actGrad, *this, Backward());  // backward(output)
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

        transpose(WtTranspose, W);
        mmadd<T>(retGradient1D, WtTranspose, *temp, nullptr);
        for (auto& p : this->prev_nodes) p->backward(&retGradient1D);
    }
};

/*
Implements softmax along width of x => output rows sum to 1
inp = this_prev_0_
assert inp.dim == 2
s = inp.exp() / inp.exp().sum(dim=0, keepdim=True)
*/
template <typename T = FloatT>
struct SoftmaxDim0 : Node<T>
{
    Matrix<T> exp;
    Matrix<T> sumExps;
    Matrix<T> softmax;
    Matrix<T> gradientOut;

    SoftmaxDim0(NodePtrs<T>& prev, const std::string& name = "SoftmaxDim0")
        : Node<T>(prev[0]->shape(), prev, name, 1),
          exp(prev[0]->t_shape()),
          sumExps(prev[0]->width, 1),
          softmax(prev[0]->t_shape()),
          gradientOut(prev[0]->shape())
    {
    }

    // computes softmax along width of x => output rows sum to 1
    void compute() override
    {
        transpose<FloatT, Exp<T>>(exp, this->prev(0));
        reduce_sum(sumExps, exp);
        binary_apply(softmax, exp, sumExps, Div<T>());
        transpose(*this, softmax);
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        softmax_gradient(gradientOut, softmax, *gradientIn);
        this->prev_nodes[0]->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params() override
    {
        return gradientOut.numels() + exp.numels() + sumExps.numels() + softmax.numels() +
               this->numels();
    }
};

/*
Implements softmax along height of x => output columns sum to 1
https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
inp = this_prev_0_
assert inp.dim == 2
s = inp.exp() / inp.exp().sum(dim=1, keepdim=True)
*/
template <typename T = FloatT>
struct SoftmaxDim1 : Node<T>
{
    Matrix<T> exp;
    Matrix<T> sumExps;
    Matrix<T> xT;
    Matrix<T> gradientOut;
    Matrix<T> gradientOutT;

    SoftmaxDim1(NodePtrs<T>& prev, const std::string& name)
        : Node<T>(prev[0]->shape(), prev, name, 1),
          exp(this->t_shape()),
          sumExps(this->width, 1),
          xT(this->t_shape()),
          gradientOut(this->shape()),
          gradientOutT(this->t_shape())
    {
    }

    // computes softmax along height of x => output columns sum to 1
    void compute() override
    {
        unary_apply(exp, this->prev(0), Exp<T>());
        reduce_sum(sumExps, exp);
        binary_apply(*this, exp, sumExps, Div<T>());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        softmax_gradient(gradientOut, *this, *gradientIn);
        transpose(gradientOutT, gradientOut, Neg<T>());
        this->prev_node(0)->backward(&gradientOutT);
    }

    virtual uint32 n_untrainable_params() override
    {
        return gradientOut.numels() + exp.numels() + sumExps.numels() + xT.numels() +
               this->numels();
    }
};

/*
Implements a Linear Transform.
Here's an equivalent python code:
def Linear(x, W):
    assert x.shape[1] == W.shape[0]
    return torch.mm(x, W)
*/
template <typename T = FloatT>
struct Linear : Node<T>
{
    Parameter<T, T> W;
    Matrix<T> xT;

    // out_width is number of output columns --> seq length
    Linear(uint32 emb_size, uint32 seq_len, NodePtrs<T>& prev,
           const std::string& name = "MMProduct")
        : Node<T>(seq_len, seq_len, prev, name, 1), W(seq_len, emb_size), xT(seq_len, emb_size)
    {
        this->params.push_back(&W);
    }

    void compute() override
    {
        multiply(*this, W, this->prev(0));
        transpose(xT, this->prev(0));  // disable this in inference only mode.
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        multiply(W.grads, *gradientIn, xT);
        this->prev_node(0)->backward(&W.grads);
    }
    virtual uint32 n_untrainable_params() override { return xT.numels(); }
};

/* Implements matrix product two matrices
 * Here's an equivalent python code:
 def Product(a, b):
    assert(a.shape[1] == b.shape[0])
    return torch.mm(a, b)
*/
template <typename T, typename PostProcess>
struct Product : Node<T>
{
    Matrix<T> aT, bT;
    Matrix<T> a_grad_in, b_grad_in;
    PostProcess pProcess;

    Product(NodePtrs<T> prevs, PostProcess pProcess, const std::string& name)
        : Node<T>(prevs[0]->height, prevs[1]->width, prevs, name, 2),
          aT(this->prev(0).t_shape()),
          bT(this->prev(1).t_shape()),
          a_grad_in(this->prev(0).shape()),
          b_grad_in(this->prev(1).shape()),
          pProcess(pProcess)
    {
        if (this->prev(0).width != this->prev(1).height)
            throw_rte_with_backtrace("Matrix dimensions do not match for product between ",
                                     this->prev(0).shape_str, " and ", this->prev(1).shape_str);
    }

    void compute() override
    {
        mmadd(*this, this->prev(0), this->prev(1), (Matrix<T>*)nullptr, pProcess);
        transpose(aT, this->prev(0), Neg<T>());  // disable this in inference only mode.
        transpose(bT, this->prev(1), Neg<T>());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        mmadd(a_grad_in, *gradientIn, bT, (Matrix<T>*)nullptr, pProcess);
        mmadd(b_grad_in, aT, *gradientIn, (Matrix<T>*)nullptr, pProcess);
        this->prev_node(0)->backward(&a_grad_in);
        this->prev_node(1)->backward(&b_grad_in);
    }
    virtual uint32 n_untrainable_params() override { return aT.numels() + bT.numels(); }
};

template <typename T = FloatT>
struct Transpose : Node<T>
{
    Matrix<T> gradientOut;

    Transpose(NodePtrs<T> prev, const std::string& name)
        : Node<T>(prev[0]->t_shape(), prev, name, 1), gradientOut(this->t_shape())
    {
    }

    void compute() override { transpose(*this, this->prev(0)); }

    void backward(const Matrix<T>* gradientIn) override
    {
        transpose(gradientOut, *gradientIn);
        this->prev_node(0)->backward(&gradientOut);
    }
};

/* Implementes the scaled dot product attention mechanism
 * https://arxiv.org/pdf/1706.03762.pdf with single head
 * Here's an equivalent python code:
def Atten(q_, k_, v_):
    assert(q_.shape == k_.shape == v_.shape == (embed_size, seq_len))
    Q = torch.nn.Parameter(torch.randn(seq_len, embed_size))
    K = torch.nn.Parameter(torch.randn(seq_len, embed_size))
    V = torch.nn.Parameter(torch.randn(seq_len, embed_size))
    q = Q @ q_  # q_ is input query
    k = K @ k_  # k_ is input key
    v = V @ v_  # v_ is input value
    qkt = q @ k.t() / (embed_size ** (1 / 2))
    s = torch.softmax(qkt, dim=-1)
    return s @ v
 */
template <typename T = FloatT>
struct Attention : Node<T>
{
    Linear<T> Q, K;
    Transpose<T> kT;
    DividebBy<T> denom;
    Product<T, DividebBy<T>> qkT;
    SoftmaxDim1<T> softmax;  // attention weights
    Linear<T> V;
    Product<T, Identity<T>> softmaxV;

    Attention(uint32 emb_size, uint32 seq_len, NodePtrs<T>& prev_qkt,
              const std::string& name = "Attention")
        : Node<T>(seq_len, seq_len, prev_qkt, name, 3),
          Q(emb_size, seq_len, {prev_qkt[0]}, name + "_Q"),
          K(emb_size, seq_len, {prev_qkt[1]}, name + "_K"),
          kT({&K}, name + "_kT"),
          denom(sqrt(emb_size)),
          qkT({&Q, &kT}, denom, name + "_QkT"),
          softmax({&qkT}, name + "_Softmax"),
          V(emb_size, seq_len, {prev_qkt[2]}, name + "_V"),
          softmaxV({&softmax, &V}, Identity<T>(), name)
    {
    }

    void forward(uint32 depth) override
    {
        softmaxV.forward(depth + 1);
        fill(*this, softmaxV);
    }

    void compute() override { softmaxV.compute(); }

    void backward(const Matrix<T>* gradientIn) override { softmaxV.backward(gradientIn); }
};

#endif  // NODES_HPP
