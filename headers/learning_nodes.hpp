#ifndef LEARNING_NODES_HPP
#define LEARNING_NODES_HPP

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cmath>
#include <cstdlib>
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

template <typename T = FloatT>
struct LinearInputT  // a consolidated input arguments for Linear.
{
    uint32 out_size;
    NodePtr<T> prev;
    bool useBias;
    std::string name;
};
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

    typedef LinearInputT<T> LinearInput;

    Linear(uint32 out_size, NodePtr<T> prev, bool useBias, const std::string& name)
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

    Linear(const LinearInput& inp) : Linear(inp.out_size, inp.prev, inp.useBias, inp.name) {}

    void forward() override { mmTadd(*this, this->prev(0), W, useBias ? &b : nullptr, ActForward); }

    void backward(const Matrix<T>* gradientIn) override
    {
        transpose(gradInT, *gradientIn, ActBackward);
        mmadd<T>(W.grads, gradInT, this->prev(0), nullptr);
        if (useBias) reduce_sum(b.grads, gradInT);
        mmadd<T>(gradOut, *gradientIn, W, nullptr);
        this->prev_nodes[0]->backward(&gradOut);
    }

    std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", shape=octagon, style=filled, fillcolor=lightblue]";
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
template <typename T = FloatT, typename ActQ = IdentityActivation<T>, typename ActK = ActQ,
          typename ActV = ActQ, typename ActOut = IdentityActivation<T>>
struct Attention : Node<T>
{
    using LinQ = Linear<T, ActQ>;
    using LinK = Linear<T, ActK>;
    using LinV = Linear<T, ActV>;
    using LinQi = typename LinQ::LinearInput;
    using LinKi = typename LinK::LinearInput;
    using LinVi = typename LinV::LinearInput;

    Linear<T, ActQ> Q;
    Linear<T, ActK> K;
    Linear<T, ActV> V;                  // The projection nodes.
    DividebBy<T> denom;                 // The denominator for scaling, sqrt(emb_size)
    ProductT<T, DividebBy<T>> qkT;      // The product of Q and K^T
    SoftmaxDim1<T> attention_weights;   // The softmax of qkT (along the dim=-1)
    Product<T, Identity<T>> attention;  // Product of Attention Weights and V

    Attention(const LinQi& Qinp, const LinKi& Kinp, const LinVi& Vinp,
              std::string name = "Attention")
        : Node<T>(Qinp.prev->height, Vinp.out_size, {Qinp.prev, Kinp.prev, Vinp.prev}, name, 3),
          Q(Qinp),
          K(Kinp),
          V(Vinp),
          denom(sqrt(Qinp.out_size)),
          qkT({&Q, &K}, denom, name + "_Q*K^T"),
          attention_weights({&qkT}, name + "_Softmax"),
          attention({&attention_weights, &V}, Identity<T>(), name + "_Softmax*V")
    {
        this->data = attention.data;
        if (Qinp.out_size != Kinp.out_size)
            throw_rte_with_backtrace("Q and V output sizes do not match for Attention ",
                                     Qinp.out_size, " != ", Kinp.out_size);
        this->prev_nodes = attention.prev_nodes;
    }

    void compute() override { attention.compute(); }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override { attention.backward(gradientIn); }

    void print_desc()
    {
        LOG(BLUE, "Attention output size: ", this->shape_str, " for Q, K: ", Q.shape_str,
            " V: ", V.shape_str, " Q.W and K.W: ", Q.W.shape_str, " V.W.shape: ", V.W.shape_str);
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;
        ss << " [label=\"" << this->name << "\", shape=box3d]\n";
        ss << "subgraph cluster_" << this->id
           << " {\n"
              "label = \""
           << this->name << "\"\n"
           << Q.id << "\n"
           << K.id << "\n"
           << V.id << "\n"
           << qkT.id << "\n"
           << attention_weights.id << "\n"
           << "}\n";

        // return " [label=\"" + this->name + "\", shape=rect]";
        return ss.str();
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
template <typename T = FloatT, typename ActQ = IdentityActivation<T>, typename ActK = ActQ,
          typename ActV = ActQ, typename OutAct = IdentityActivation<T>>
struct MultiHeadAttention : Node<T>
{
    using Att = Attention<T, ActQ, ActK, ActV>;
    using LinO = Linear<T, OutAct>;
    using LinOi = typename LinO::LinearInput;
    using LinQi = typename Att::LinQi;
    using LinKi = typename Att::LinKi;
    using LinVi = typename Att::LinVi;

    std::vector<std::unique_ptr<Att>> heads;
    std::unique_ptr<Concat<T>> concat;
    std::unique_ptr<LinO> linear;

    MultiHeadAttention(uint32 num_heads, LinQi Qinp, LinKi Kinp, LinVi Vinp, LinOi Oinp,
                       std::string name = "MHA")
        : Node<T>(Qinp.prev->height, Oinp.out_size, {Qinp.prev, Kinp.prev, Vinp.prev}, name, 3)
    {
        NodePtrs<T> head_ptrs;
        for (uint32 i = 0; i < num_heads; ++i)
        {
            auto att = new Att(Qinp, Kinp, Vinp, name + "_Head_" + std::to_string(i));
            heads.emplace_back(att);
            head_ptrs.push_back(att);
        }
        concat = std::make_unique<Concat<T>>(head_ptrs, name + "_Concat");
        Oinp.prev = concat.get();
        Oinp.name = name + "_Linear";
        linear = std::make_unique<LinO>(Oinp);
        this->data = linear->data;
        this->prev_nodes = linear->prev_nodes;
    }

    MultiHeadAttention(uint32 num_heads, uint32 out_size, LinQi Qinp, std::string name = "SelfMHA")
        : MultiHeadAttention(num_heads, {Qinp.out_size, Qinp.prev, Qinp.useBias, "Q_" + Qinp.name},
                             {Qinp.out_size, Qinp.prev, Qinp.useBias, "K_" + Qinp.name},
                             {Qinp.out_size, Qinp.prev, Qinp.useBias, "V_" + Qinp.name},
                             {out_size, nullptr, true, name + "_Linear"})
    {
    }

    void compute() override { linear->compute(); }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override { linear->backward(gradientIn); }

    void print_desc()
    {
        LOG(BLUE, "MultiHeadAttention with ", heads.size(),
            " heads; Linear projection matrix shape: ", linear->W.shape_str,
            " to output: ", this->shape_str, " each attention looks like: ");
        heads[0]->print_desc();
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;
        ss << " [label=\"" << this->name << "_linear\", shape=box3d]\n";
        ss << "subgraph cluster_" << this->id << " {\nlabel = \"" << this->name << "\"\n";
        for (auto& h : heads) ss << h->id << "\n";
        ss << concat->id << "\n" << this->id << "\n}\n";
        return ss.str();
    }
};

template <typename T = FloatT, typename Act1 = IdentityActivation<T>, typename Act2 = Relu<T>>
struct MLP : Node<T>
{
    using Linear1 = Linear<T, Act1>;
    using Linear2 = Linear<T, Act2>;
    using Lin1i = typename Linear1::LinearInput;
    using Lin2i = typename Linear2::LinearInput;

    std::unique_ptr<Dropout<T>> dropout;
    std::unique_ptr<Linear1> l1;
    std::unique_ptr<Linear2> l2;

    MLP(Lin1i l1i, Lin2i l2i, FloatT dropout_ratio, const std::string& name = "MLP")
        : Node<T>(l1i.prev->height, l2i.out_size, {l1i.prev}, name, 1)
    {
        if (l2i.prev != nullptr)
        {
            throw_rte_with_backtrace(
                "MLP: Linear2 should not have a previous node (it's assigned to as yet "
                "non-existent Linear1)");
        }
        l1 = std::make_unique<Linear1>(l1i);
        dropout = std::make_unique<Dropout<T>>(dropout_ratio, l1.get(), name + "_Dropout");
        l2i.prev = dropout.get();
        l2 = std::make_unique<Linear2>(l2i);

        this->data = l2->data;
        this->prev_nodes = l2->prev_nodes;
    }
    MLP(uint32 out_size, Lin1i l1i, FloatT dropout_ratio, const std::string& name = "MLP")
        : MLP(l1i, {out_size, nullptr, false, "Lin2"}, dropout_ratio, name)
    {
    }

    void forward() override { l2->forward(); }

    void backward(const Matrix<T>* gradientIn) override { l2->backward(gradientIn); }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;
        ss << " [label=\"" << this->name
           << "\", shape=doubleoctagon, style=filled, fillcolor=\"#46bfe8\"]\n"
           << "subgraph cluster_" << this->id << " {\nlabel = \"" << this->name << "\"\n"
           << l1->id << "\n"
           << dropout->id << "\n"
           << this->id << "\n}\n";
        return ss.str();
    }
};

#endif  // LEARNING_NODES_HPP
