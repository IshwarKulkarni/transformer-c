#ifndef LEARNING_NODES_HPP
#define LEARNING_NODES_HPP

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cmath>
#include <cstdlib>
#include <memory>
#include "functors.cuh"
#include "matrix.cuh"
#include "matrix_ops.cuh"
#include "node_parameter.hpp"
#include "nodes.hpp"

#define print(x, name) print_ln(x, name, __FILE__, __LINE__)

/*
Implements a Fully connected layer with activation, that takes `outW` column vectors of size
as inputs and produces `outW` column vectors as output, each of size `outH`;
Here's an equivalent python code:
def Linear(x, W, b, Activation_):
    assert x.shape[1] == W.shape[0]
    return Activatation_(torch.mm(W, x) + b)
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
          retGradient1D(inH, 1),  // gradient sum from all input Columns
          WtTranspose(inH, outH),
          useBias(bias)
    {
        this->params.push_back(&W);
        this->params.push_back(&b);
    }

    void forward() override
    {
        Matrix<T>& x = this->prev(0);
        const auto* bias = useBias ? &b : nullptr;
        mmadd<T, Forward>(*this, W, x, bias);
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

        binary_apply(actGxGradIn, *gradientIn, *temp,
                     Mul<T>());  // a = gradientIn x backward(output)
        mmTadd<T>(W.grads, actGxGradIn, this->prev(0), nullptr);  // a @ input

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
        this->prev_nodes[0]->backward(&retGradient1D);
    }
};

/*
Implements a torch.Linear, y = x @ W^T (no Bias).
def Linear(x, W):
    assert x.shape[1] == W.shape[1]
    return torch.mm(x, W.t())
expectation is that x is a (height-wise)batch of row vectors, each size Ei, Size:(batch_size, Ei)
and produce a matrix of size (batch_size, out_size)
because W is a matrix of size (out_size, Ei)
*/
template <typename T = FloatT>
struct Linear : Node<T>
{
    Parameter<T, T> W;
    Matrix<T> gradInT;
    Matrix<T> gradOut;

    // out_size is width of each output row.
    Linear(uint32 out_size, NodePtrs<T>& prev, const std::string& name)
        : Node<T>(prev[0]->height, out_size, prev, name, 1),
          W(out_size, prev[0]->width, nullptr, name + "_W"),
          gradInT(this->t_shape()),
          gradOut(prev[0]->shape())
    {
        this->params.push_back(&W);
    }

    void forward() override { mmTadd<T>(*this, this->prev(0), W, nullptr); }

    void backward(const Matrix<T>* gradientIn) override
    {
        transpose(gradInT, *gradientIn);
        mmadd<T>(W.grads, gradInT, this->prev(0), nullptr);
        mmadd<T>(gradOut, *gradientIn, W, nullptr);
        this->prev_nodes[0]->backward(&gradOut);
    }
};

/* Implementes the scaled dot product attention mechanism
 * https://arxiv.org/pdf/1706.03762.pdf with single head
 * Here's an equivalent python code:
def Atten(q_, k_, v_, q_size, v_size):  #q_ `emb_size`d rows vectors
    Q = torch.nn.Parameter(torch.randn(q_size, embed_size))
    K = torch.nn.Parameter(torch.randn(q_size, embed_size))
    V = torch.nn.Parameter(torch.randn(v_size, embed_size))
    q = torch.mul(Q, q_.t())  # q_ is input query
    k = torch.mul(K, k_.t())  # k_ is input key
    v = torch.mul(V, v_.t())  # v_ is input value
    qkt = torch.mul(q, k.t()) / (q_size ** (1 / 2))
    s = torch.softmax(qkt, dim=-1)
    return s @ v
 */
template <typename T = FloatT>
struct Attention : Node<T>
{
    Linear<T> Q, K, V;                  // The projection matrices
    DividebBy<T> denom;                 // The denominator for scaling, sqrt(emb_size)
    ProductT<T, DividebBy<T>> qkT;      // The product of Q and K^T
    SoftmaxDim1<T> attention_weights;   // The softmax of qkT (along the dim=-1)
    Product<T, Identity<T>> attention;  // Product of Attention Weights and V

    Attention(uint32 q_size, uint32 v_size, NodePtrs<T>& _qkt,
              const std::string& name = "Attention")
        : Node<T>(_qkt[0]->height, v_size, _qkt, name, 3),
          Q(q_size, {_qkt[0]}, name + "_Q"),
          K(q_size, {_qkt[1]}, name + "_K"),
          V(v_size, {_qkt[2]}, name + "_V"),
          denom(sqrt(q_size)),
          qkT({&Q, &K}, denom, name + "_Q*K^T"),
          attention_weights({&qkT}, name + "_Softmax"),
          attention({&attention_weights, &V}, Identity<T>(), name + "_Softmax*V")
    {
        this->data = attention.data;
        LOG(BLUE, "Attention output size: ", this->shape_str, " for Q: ", Q.shape_str,
            " K: ", K.shape_str, " V: ", V.shape_str, " Q.W.shape: ", Q.W.shape_str,
            " K.W.shape: ", K.W.shape_str, " V.W.shape: ", V.W.shape_str);
    }

    void compute() override
    {
        attention.compute();
        cudaErrCheck(cudaDeviceSynchronize());
    }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override { attention.backward(gradientIn); }

    uint32 n_trainable_params() override { return attention.n_trainable_params(); }
};

/*
MultiHeadAttention:
Input is a std::vector of 3 matrices, each of size `S x Ei`, where S is the sequence length.
With `n_heads`, each head projects querys and keys to `S x q_size`
to generate attention and, values are projected to `S x v_size` to generate each output,
that are concatenated to `S x n_heads * v_size`, which are then linearly transformed to
`S x out_size`.
*/
template <typename T = FloatT>
struct MultiHeadAttention : Node<T>
{
    std::vector<std::unique_ptr<Attention<T>>> heads;
    std::unique_ptr<Concat<T>> concat;
    std::unique_ptr<Linear<T>> linear;

    MultiHeadAttention(uint32 n_heads, uint32 q_size, uint32 v_size, uint32 out_size,
                       NodePtrs<T>& _qkt, const std::string& name = "MultiHeadAttention")
        : Node<T>(_qkt[0]->height, out_size, _qkt, name, 3)
    {
        std::vector<Node<T>*> head_ptrs;
        for (uint32 i = 0; i < n_heads; ++i)
        {
            auto att = new Attention<T>(q_size, v_size, _qkt, name + "_Head_" + std::to_string(i));
            heads.emplace_back(att);
            head_ptrs.push_back(att);
        }
        concat = std::make_unique<Concat<T>>(head_ptrs, name + "_Concat");
        NodePtrs<T> concat_ptr = {concat.get()};
        linear = std::make_unique<Linear<T>>(out_size, concat_ptr, name + "_Linear");

        this->data = linear->data;

        LOG(BLUE, "MultiHeadAttention with ", n_heads,
            " heads, Each Attentions Head's output size is ", this->shape_str,
            " Linear projection matrix shape: ", linear->W.shape_str,
            " to output: ", this->shape_str);
    }

    void compute() override { linear->compute(); }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override { linear->backward(gradientIn); }
};

#endif  // LEARNING_NODES_HPP
