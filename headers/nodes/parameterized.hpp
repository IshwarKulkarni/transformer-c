#ifndef NODES_PARAMETERIZED_HPP
#define NODES_PARAMETERIZED_HPP

/*
Various nodes that have learnable parameters, currently they are either Linear<> or use Linear<>
*/

#include <cmath>
#include <cstdlib>
#include "matrix_ops.cuh"
#include "nodes/unparameterized.hpp"

template <typename T = FloatT>
struct LinearInputT  // a consolidated input arguments for Linear.
{
    uint32 out_size;
    NodePtr<T> prev;
    bool useBias;
    std::string name;
};

/*
Implements torch.Linear with Bias and activation, y = Act(X @ W^T + b)
X: stack of row vectors, W: weight matrix, b: bias vector, Act: activation function
*/
template <typename T = FloatT, typename Act = IActivation<T>>
struct Linear : Node<T>
{
    Node<T>* input_node;
    Parameter<T, T> W;
    Parameter<T, T> b;
    Matrix<T> gradientOut;
    Matrix<T> WGradUpdate;
    Matrix<T> bGradUpdate;
    Matrix<T> tempT, temp;
    bool useBias;

    typedef LinearInputT<T> LinearInput;

    typedef ActBackwardMul<T, Act> ActBackwardMulT;

    ActBackwardMulT gradientFunctor;

    Linear(uint32 out_width, NodePtr<T> prev, bool useBias, const std::string& name)
        : Node<T>(prev->shape.set(WIDTH_IDX, out_width), {prev}, name, 1),
          input_node(prev),
          W({out_width, prev->width()}, name + "_W"),
          b({1, out_width}, name + "_b"),
          gradientOut(prev->shape, name + "_gradientOut"),
          WGradUpdate(W.shape.set(BATCH_IDX, prev->batch()), name + "_WGradUpdate"),
          bGradUpdate(b.shape.set(BATCH_IDX, prev->batch()), name + "_bGradUpdate"),
          tempT(this->shape.t(), name + "_tempT2"),
          temp(this->shape, name + "_temp"),
          useBias(useBias)
    {
        this->params.push_back(&W);
        std::string bias_str = "\t\t";
        if (useBias)
        {
            this->params.push_back(&b);
            b.reset();
            bias_str = " + B: (" + std::to_string(b.numels()) + ") ";
        }
        else
            b.set_val(0.f);
        LOG(BLUE, R_JUST(this->name, 18), prev->shape, R_JUST("->", 6), this->shape,
            "\t| W: ", W.shape, " (", num_to_si(W.numels(), false), ")", bias_str,
            "| Activation: ", Act::name);
    }

    void init()
    {
        Act act;
        if (dynamic_cast<Relu<T>*>(&act))  // else xavier is default
            kaiming_init(W);
    }

    Linear(const LinearInput& inp) : Linear(inp.out_size, inp.prev, inp.useBias, inp.name) {}

    void forward() override
    {
        if (useBias)
            mmTadd(*this, *input_node, W, {b}, typename Act::forward());
        else
            mmTadd(*this, *input_node, W, {}, typename Act::forward());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_TRACE(GRAY, "Backward", RESET, " for ", this->name,
                  " with gradientIn: ", gradientIn->name, gradientIn->shape);
        auto const* gradIn = gradientIn;
        if constexpr (not std::is_same<Act, IActivation<T>>::value)
        {
            binary_apply(temp, *this, *gradientIn, gradientFunctor);
            gradIn = &temp;
        }

        if (useBias)
        {
            if (gradIn->height() > 1)
            {
                reduce<T, HEIGHT_IDX, Plus<T>>(bGradUpdate, *gradIn);
                b.accumulate_grad(bGradUpdate);
            }
            else
                b.accumulate_grad(*gradIn);
        }

        transpose(tempT, *gradIn);
        mmadd(WGradUpdate, tempT, *input_node, {});
        W.accumulate_grad(WGradUpdate);

        if (dynamic_cast<Input<T>*>(input_node)) return;

        multiply(gradientOut, *gradientIn, W);
        input_node->backward(&gradientOut);
    }

    std::string dot_repr() override
    {
        char label_sz[256];
        int32 n = snprintf(label_sz, sizeof(label_sz), "%s\n[%dx%dx%d]:%s", this->name.c_str(),
                           W.batch(), W.height(), W.width(), num_to_si(W.numels()).c_str());
        if (useBias)
            snprintf(label_sz + n, sizeof(label_sz) - n, "\n[%dx%dx%d]:%s", b.batch(), b.height(),
                     b.width(), num_to_si(b.numels()).c_str());

        std::string label = label_sz;
        return " [label=\"" + label + "\", shape=rect, style=filled, fillcolor=lightblue]";
    }

    void debug_print()
    {
        LOG("Debug print for ", this->name, "\n", *this, W, WGradUpdate, *this, gradientOut);
        if (useBias) LOG(b, bGradUpdate);
    }

    void save_weights(std::ostream& os) const override
    {
        char activation[16] = {0};
        snprintf(activation, sizeof(activation), "%s", Act::name);
        os.write(activation, sizeof(activation));

        int8 bias[1] = {useBias ? int8(1) : int8(0)};
        os.write(bias, sizeof(bias));

        W.save_weights(os);
        b.save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        char activation[16] = {0};
        is.read(activation, sizeof(activation));
        if (strcmp(activation, Act::name) != 0)
            LOG(RED, "Activation mismatch for ", this->name, " expected ", Act::name, " but got ",
                activation);

        char bias[1] = {0};
        is.read(bias, sizeof(bias));
        if (bias[0] != useBias)
            LOG(RED, "Bias mismatch for ", this->name, " expected ", useBias, " but got ", bias[0]);

        W.load_weights(is);
        b.load_weights(is);
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
template <typename T = FloatT, typename ActQ = IActivation<T>, typename ActK = ActQ,
          typename ActV = ActQ>
struct Attention : Node<T>
{
    using LinQ = Linear<T, ActQ>;
    using LinK = Linear<T, ActK>;
    using LinV = Linear<T, ActV>;
    using LinQi = typename LinQ::LinearInput;
    using LinKi = typename LinK::LinearInput;
    using LinVi = typename LinV::LinearInput;

    LinQ Q;
    LinK K;
    LinV V;                             // The projection nodes.
    DividedBy<T> denom;                 // The denominator for scaling, sqrt(emb_size)
    ProductT<T, DividedBy<T>> qkT;      // The product of Q and K^T
    SoftmaxDim0<T> attention_weights;   // The softmax of qkT (along the dim=-1)
    Product<T, Identity<T>> attention;  // Product of Attention Weights and V

    Attention(const LinQi& Qinp, const LinKi& Kinp, const LinVi& Vinp,
              std::string name = "Attention")
        : Node<T>(Qinp.prev->shape.set(WIDTH_IDX, Vinp.out_size), {}, name, 0),
          Q(Qinp),
          K(Kinp),
          V(Vinp),
          denom(sqrt(Qinp.out_size)),
          qkT({&Q, &K}, denom, name + "_Q*K^T"),
          attention_weights({&qkT}, name + "_Softmax"),
          attention({&attention_weights, &V}, Identity<T>(), name + "_Softmax*V")
    {
        // this->data = attention.data;
        if (Qinp.out_size != Kinp.out_size)
            throw_rte_with_backtrace("Q and V output sizes do not match for Attention ",
                                     Qinp.out_size, " != ", Kinp.out_size);
        this->set_data(attention.get_data());
    }

    void forward() override { attention.compute(); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                  gradientIn->shape);
        attention.backward(gradientIn);
    }

    void print_desc()
    {
        LOG(BLUE, "Attention output size: ", this->shape, " for Q, K: ", Q.shape, " V: ", V.shape,
            " Q.W and K.W: ", Q.W.shape, " V.W.shape: ", V.W.shape);
    }

    virtual std::string dot_repr() override
    {
        uint32 learnable_count = Q.param_count() + K.param_count() + V.param_count();
        std::stringstream ss;
        ss << " [label=\"" << this->name << "\", shape=box3d style=filled fillcolor=\"#4eb0f1\"]\n";

        std::set<uint32> input_ids{Q.prev(0).id, V.prev(0).id, K.prev(0).id};
        NodePtrList<T> nodes = {&Q, &K, &V, &qkT, &attention_weights, &attention};

        ss << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "["
           << num_to_si(learnable_count, true) << "]\"\n";
        for (auto& n : nodes) ss << n->id << ' ';

        if (input_ids.size() == 1) ss << Q.prev(0).id << '\n';
        ss << "\n{rank=same; " << Q.id << ' ' << K.id << ' ' << V.id << " }\n"
           << "\n{rank=same; " << attention_weights.id << ' ' << attention.id << " }\n"
           << this->id << "}\n";  // is attention

        return ss.str();
    }

    NodePtr<T> get_terminal_node() override { return &attention; }

    void save_weights(std::ostream& os) const override
    {
        Q.save_weights(os);
        K.save_weights(os);
        V.save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        Q.load_weights(is);
        K.load_weights(is);
        V.load_weights(is);
    }
};

template <typename T = FloatT, typename Act = IActivation<T>>
struct SelfAttention : Node<T>
{
    NodePtr<T> prev;
    Copy<T> x = Copy<T>(prev, "SA-Input");
    Attention<T, Act> attn;

    SelfAttention(uint32 out_size, const NodePtr<T> prev_, bool bias = false,
                  std::string name = "SelfAttention")
        : Node<T>(prev_->shape.set(WIDTH_IDX, out_size), {prev_}, name, 1),
          prev(prev_),
          attn({out_size, &x, bias, name + "_Q"}, {out_size, &x, bias, name + "_K"},
               {out_size, &x, bias, name + "_V"}, name)
    {
        LOG(BLUE, "SAttn: ", this->name, this->shape, " for input: ", prev->name, prev->shape,
            " attention node: ", attn.name, attn.shape);
        this->set_data(attn.get_data());
    }

    void forward() override { attn.compute(); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                  gradientIn->shape);
        attn.backward(gradientIn);
        prev->backward(&x.gradientOut);
        x.gradientOut.set_val(0.f);
    }

    NodePtr<T> get_terminal_node() override { return &attn; }

    virtual std::string dot_repr() override
    {
        return "[label=\"" + this->name + "\", shape=box3d style=filled fillcolor=\"#4eb0f1\"]\n";
    }

    void save_weights(std::ostream& os) const override { attn.save_weights(os); }

    void load_weights(std::istream& is) override { attn.load_weights(is); }
};

/*
MultiHeadAttention:
Input is a std::vector of 3 matrices, each of size `S x Ei`, where S is the sequence length.
With `n_heads`, each head projects querys and keys to `S x q_size`
to generate attention and, values are projected to `S x v_size` to generate each output,
that are concatenated to `S x n_heads * v_size`, which are then linearly transformed to
`S x out_size`.
*/
template <typename T = FloatT, typename OutAct = Sigmoid<T>, typename ActQ = IActivation<T>,
          typename ActK = ActQ, typename ActV = ActQ>
struct MultiHeadAttention : Node<T>
{
    using Att = Attention<T, ActQ, ActK, ActV>;
    using LinO = Linear<T, OutAct>;
    using LinOi = typename LinO::LinearInput;
    using LinQi = typename Att::LinQi;
    using LinKi = typename Att::LinKi;
    using LinVi = typename Att::LinVi;

    std::vector<std::unique_ptr<Att>> heads;
    std::unique_ptr<Concat0<T>> concat;
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
        concat = std::make_unique<Concat0<T>>(head_ptrs, name + "_Concat");
        Oinp.prev = concat.get();
        Oinp.name = name + "_Linear";
        linear = std::make_unique<LinO>(Oinp);
        this->data = linear->data;
        this->prev_nodes = linear->prev_nodes;
    }

    MultiHeadAttention(uint32 num_heads, uint32 out_size, NodePtr<T> prev, std::string name = "MHA")
        : MultiHeadAttention(
              num_heads, {out_size, prev, false, "Q_" + name}, {out_size, prev, false, "K_" + name},
              {out_size, prev, false, "V_" + name}, {out_size, nullptr, false, name + "_Linear"})
    {
    }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                  gradientIn->shape);
        linear->backward(gradientIn);
    }

    void print_desc()
    {
        LOG(BLUE, "MultiHeadAttention with ", heads.size(),
            " heads; Linear projection matrix shape: ", linear->W.shape,
            " to output: ", this->shape, " each attention looks like: ");
        heads[0]->print_desc();
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;

        ss << " [label=\"" << this->name << '\n'
           << linear->W.shape << ':' << linear->W.numels()
           << " \", shape=box3d,  style=filled, fillcolor=azure ]\n";
        ss << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "\"\n";
        ss << '\t' << concat->id << '\n';
        ss << '\t' << this->id << "\n}\n";
        return ss.str();
    }

    void save_weights(std::ostream& os) const override
    {
        uint32 num_heads = heads.size();
        os.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
        for (auto& head : heads) head->save_weights(os);
        linear->save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        uint32 num_heads = 0;
        is.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
        if (num_heads != heads.size())
        {
            if (num_heads != 1)
                throw_rte_with_backtrace("Number of heads mismatch for MultiHeadAttention ",
                                         num_heads, " != ", heads.size());
            auto pos = is.tellg();
            for (uint32 i = 0; i < num_heads; ++i)  // replicate the head
            {
                is.seekg(pos);
                heads[i]->load_weights(is);
            }
        }
        else
        {
            for (auto& head : heads) head->load_weights(is);
        }
        linear->load_weights(is);
    }
};

template <typename T = FloatT, typename Act1 = Relu<T>, typename Act2 = IActivation<T>>
struct FeedForward : Node<T>
{
    using LinearIn = Linear<T, Act1>;
    using LinearOut = Linear<T, Act2>;
    using LinIni = typename LinearIn::LinearInput;
    using LinOuti = typename LinearOut::LinearInput;

    std::unique_ptr<LinearIn> l_in;
    std::unique_ptr<Dropout<T>> dropout;
    std::unique_ptr<LinearOut> l_out;

    FeedForward(LinIni l1i, LinOuti l2i, FloatT dropout_ratio, const std::string& name = "MLP")
        : Node<T>(l1i.prev->shape.set(WIDTH_IDX, l2i.out_size), {l1i.prev}, name, 1)
    {
        if (l2i.prev != nullptr)
        {
            throw_rte_with_backtrace(
                "MLP: Linear2 should not have a previous node (it's assigned to as yet "
                "non-existent Linear1)");
        }
        l_in = std::make_unique<LinearIn>(l1i);
        dropout = std::make_unique<Dropout<T>>(dropout_ratio, l_in.get(), name + "_Dropout");
        l2i.prev = dropout.get();
        l_out = std::make_unique<LinearOut>(l2i);

        this->prev_nodes = {l_out.get()};
    }

    FeedForward(uint32 l1_size, uint32 l2_size, const NodePtr<T> prev, FloatT dropout_ratio,
                const std::string& name = "MLP")
        : FeedForward({l1_size, prev, false, name + "_Lin1"},
                      {l2_size, nullptr, true, name + "_Lin2"}, dropout_ratio, name)
    {
    }

    void forward() override { this->copy(l_out->begin()); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                  gradientIn->shape);
        l_out->backward(gradientIn);
    }

    virtual std::string dot_repr() override
    {
        uint32 learnable_count = l_in->param_count() + l_out->param_count();
        std::stringstream ss;
        ss << " [label=\"" << this->name
           << "\", shape=doubleoctagon, style=filled, fillcolor=\"#46bfe8\"]\n"
           << "subgraph cluster_" << this->id << "   {\nlabel = \"" << this->name << "\n["
           << learnable_count << "]\"\n"
           << '\t' << l_in->id << '\n'
           << '\t' << dropout->id << '\n'
           << '\t' << l_out->id << '\n'
           << '\t' << this->id << "\n}\n";
        return ss.str();
    }

    void debug_print()
    {
        LOG("Debug print for ", this->name);
        l_in->debug_print();
        dropout->debug_print();
        l_out->debug_print();
    }

    void save_weights(std::ostream& os) const override
    {
        l_in->save_weights(os);
        l_out->save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        l_in->load_weights(is);
        l_out->load_weights(is);
    }
};

#endif  // NODES_PARAMETERIZED_HPP
