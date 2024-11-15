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

template <typename T = FloatT>
void print_ln(const Matrix<T>& x, const std::string& name, const char* file, int line)
{
    cudaErrCheck(cudaDeviceSynchronize());
    std::cout << name << " @ " << file << ":" << line << " " << x << std::endl;
}

/*
Implements torch.Linear with Bias, y = Act(X @ W^T + b)
X: stack of row vectors, W: weight matrix, b: bias vector, Act: activation function
*/
template <typename T = FloatT, typename Act = IdentityActivation<T>>
struct Linear : Node<T>
{
    Parameter<T, T> W;
    Parameter<T, T> b;
    Matrix<T> gradInT;
    Matrix<T> gradOut;
    bool useBias;
    typename Act::forward ActForward;
    typename Act::backward ActBackward;

    Linear(uint32 out_size, Node<T>* prev, bool useBias, const std::string& name)
        : Node<T>(prev->height, out_size, {prev}, name, 1),
          W(out_size, prev->width, name + "_W"),
          b(1, out_size, name + "_b"),
          gradInT(this->t_shape()),
          gradOut(prev->shape()),
          useBias(useBias)
    {
        this->params.push_back(&W);
        if (useBias) this->params.push_back(&b);
    }

    void forward() override { mmTadd(*this, this->prev(0), W, useBias ? &b : nullptr, ActForward); }

    void backward(const Matrix<T>* gradientIn) override
    {
        transpose(gradInT, *gradientIn, ActBackward);
        mmadd<T>(W.grads, gradInT, this->prev(0), nullptr);
        if (useBias) reduce_sum(b.grads, gradInT);
        mmadd<T>(gradOut, *gradientIn, W, nullptr);
        this->prev_nodes[0]->backward(&gradOut);
    }
};

/* Implementes the scaled dot product attention mechanism
https://arxiv.org/pdf/1706.03762.pdf with single head
Here's an equivalent python code:
def Atten(q_, k_, v_):  #q_ `emb_size`d rows vectors
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
          Q(q_size, {_qkt[0]}, false, name + "_Q"),
          K(q_size, {_qkt[1]}, false, name + "_K"),
          V(v_size, {_qkt[2]}, false, name + "_V"),
          denom(sqrt(q_size)),
          qkT({&Q, &K}, denom, name + "_Q*K^T"),
          attention_weights({&qkT}, name + "_Softmax"),
          attention({&attention_weights, &V}, Identity<T>(), name + "_Softmax*V")
    {
        this->data = attention.data;
    }

    void compute() override
    {
        attention.compute();
        cudaErrCheck(cudaDeviceSynchronize());
    }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override { attention.backward(gradientIn); }

    void print_desc()
    {
        LOG(BLUE, "Attention output size: ", this->shape_str, " for Q: ", Q.shape_str,
            " K: ", K.shape_str, " V: ", V.shape_str, " Q.W.shape: ", Q.W.shape_str,
            " K.W.shape: ", K.W.shape_str, " V.W.shape: ", V.W.shape_str);
    }
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
        linear = std::make_unique<Linear<T>>(out_size, concat.get(), false, name + "_Linear");
        this->data = linear->data;
    }

    void compute() override { linear->compute(); }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override { linear->backward(gradientIn); }

    void print_desc()
    {
        LOG(BLUE, "MultiHeadAttention with ", heads.size(),
            " heads, Each Attentions Head's output size is ", this->shape_str,
            " Linear projection matrix shape: ", linear->W.shape_str,
            " to output: ", this->shape_str);
    }
};

#endif  // LEARNING_NODES_HPP
