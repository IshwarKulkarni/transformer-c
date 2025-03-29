#ifndef NODES_PARAMETERIZED_HPP
#define NODES_PARAMETERIZED_HPP

/*
Various nodes that have learnable parameters, currently they are either Linear<> or use Linear<>
*/

#include <cmath>
#include <cstdlib>
#include <memory>
#include "matrix_ops.hpp"
#include "node.hpp"
#include "nodes/unparameterized.hpp"

template <typename T = FloatT>
struct LinearInput  // Consolidated input arguments for Linear.
{
    uint32 out_size;
    NodePtr<T> prev;
    bool useBias;
    std::string act_name;
    std::string name;
};

/*
Implements torch.Linear with Bias and activation, y = Act(X @ W^T + b)
X: stack of row vectors, W: weight matrix, b: bias vector, Act: activation function
*/
template <typename T = FloatT>
struct Linear : Node<T>
{
    Node<T>* input_node;
    Parameter<T, T> W;
    Parameter<T, T> b;
    Matrix<T> gradientOut;
    Matrix<T> WGradUpdate;
    Matrix<T> bGradUpdate;
    Matrix<T> tempT, temp;
    const bool useBias;
    const ActivationEnum act;

    Linear(uint32 out_width, NodePtr<T> prev, bool useBias, std::string act_name,
           const std::string& name)
        : Node<T>(prev->shape.set(WIDTH_IDX, out_width), {prev}, name, 1),
          input_node(prev),
          W({out_width, prev->width()}, name + "_W"),
          b({1, out_width}, name + "_b"),
          gradientOut(prev->shape, name + "_gradientOut"),
          WGradUpdate(W.shape.set(BATCH_IDX, prev->batch()), name + "_WGradUpdate"),
          bGradUpdate(b.shape.set(BATCH_IDX, prev->batch()), name + "_bGradUpdate"),
          tempT(this->shape.t(), name + "_tempT2"),
          temp(this->shape, name + "_temp"),
          useBias(useBias),
          act(get_activation_enum(act_name))
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
            "| Activation: ", get_act_name(act));

        if (act == ActivationEnum::Relu) kaiming_init(W);
    }

    explicit Linear(const LinearInput<T>& inp)
        : Linear(inp.out_size, inp.prev, inp.useBias, inp.act_name, inp.name)
    {
    }

    __attribute__((always_inline)) inline void forward() override
    {
        auto bias = useBias ? Optional<Matrix<T>>(b) : Optional<Matrix<T>>();
        switch (act)
        {
            case ActivationEnum::Relu:
                mmTadd<T, typename Relu<T>::ReluF>(*this, *input_node, W, bias);
                break;
            case ActivationEnum::LeakyRelu:
                mmTadd<T, typename LeakyRelu<T>::LeakyReluF>(*this, *input_node, W, bias);
                break;
            case ActivationEnum::TanH:
                mmTadd<T, typename TanH<T>::TanhF>(*this, *input_node, W, bias);
                break;
            case ActivationEnum::Sigmoid:
                mmTadd<T, typename Sigmoid<T>::SigmoidF>(*this, *input_node, W, bias);
                break;
            case ActivationEnum::IActivation:
                mmTadd<T, Identity<T>>(*this, *input_node, W, bias);
                break;
            default:
                throw_rte_with_backtrace("Unknown activation function: ", get_act_name(act));
        }
    }

    __attribute__((always_inline)) void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE(GRAY, "Backward", RESET, " for ", this->name,
                       " with gradientIn: ", gradientIn->name, gradientIn->shape);
        auto const* gradIn = &temp;
        switch (act)
        {
            case ActivationEnum::IActivation:
                gradIn = gradientIn;
                break;
            case ActivationEnum::Relu:
                binary_apply(temp, *this, *gradientIn, ActBackwardMul<T, Relu<T>>());
                break;
            case ActivationEnum::LeakyRelu:
                binary_apply(temp, *this, *gradientIn, ActBackwardMul<T, LeakyRelu<T>>());
                break;
            case ActivationEnum::TanH:
                binary_apply(temp, *this, *gradientIn, ActBackwardMul<T, TanH<T>>());
                break;
            case ActivationEnum::Sigmoid:
                binary_apply(temp, *this, *gradientIn, ActBackwardMul<T, Sigmoid<T>>());
                break;
            default:
                throw_rte_with_backtrace("Unknown activation function: ", get_act_name(act));
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
        snprintf(activation, sizeof(activation), "%s", get_act_name(act));
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
        if (strcmp(activation, get_act_name(act)) != 0)
            LOG(RED, "Activation mismatch for ", this->name, " expected ", get_act_name(act),
                " but got ", activation);

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
template <typename T = FloatT>
struct Attention : Node<T>
{
    std::unique_ptr<Linear<T>> Ql;
    std::unique_ptr<Linear<T>> Kl;
    std::unique_ptr<Linear<T>> Vl;                   // The projection nodes.
    DividedBy<T> denom;                              // The denominator for scaling, sqrt(emb_size)
    std::unique_ptr<ProductT<T, DividedBy<T>>> qkT;  // The product of Q and K^T
    std::unique_ptr<SoftmaxDim0<T>> attention_weights;   // The softmax of qkT (along the dim=-1)
    std::unique_ptr<Product<T, Identity<T>>> attention;  // Product of Attention Weights and V

    std::unique_ptr<InputProxy<T>>
        KV_proxy;  // when K and V have same input, like in CrossAttention

    Attention(LinearInput<T> Qinp, LinearInput<T> Kinp, LinearInput<T> Vinp,
              std::string name = "Attention")
        : Node<T>(Qinp.prev->shape.set(WIDTH_IDX, Vinp.out_size), {}, name, 0),
          denom(sqrt(Qinp.out_size))
    {
        if (Qinp.out_size != Kinp.out_size)
            throw_rte_with_backtrace("Q and V output sizes do not match for Attention ",
                                     Qinp.out_size, " != ", Kinp.out_size);

        if (Vinp.prev->id == Kinp.prev->id and Kinp.prev->id != Qinp.prev->id)
        {
            LOG(YELLOW, "Attention: ", name, "'s K and V have same input, using ProxyNode");
            KV_proxy = std::make_unique<InputProxy<T>>(Kinp.prev, name + "_KV_proxy");
            Kinp.prev = KV_proxy.get();
            Vinp.prev = KV_proxy.get();
        }
        Ql = std::make_unique<Linear<T>>(Qinp);
        Kl = std::make_unique<Linear<T>>(Kinp);
        Vl = std::make_unique<Linear<T>>(Vinp);

        qkT = std::make_unique<ProductT<T, DividedBy<T>>>(NodePtrVec<T>{Ql.get(), Kl.get()}, denom,
                                                          name + "_Q*K^T");
        attention_weights = std::make_unique<SoftmaxDim0<T>>(qkT.get(), name + "_Softmax");
        attention = std::make_unique<Product<T, Identity<T>>>(
            NodePtrVec<T>{attention_weights.get(), Vl.get()}, Identity<T>(), name + "_Softmax*V");

        this->set_data(attention->get_data());
    }

    void forward() override { attention->compute(); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        // attention.backward() takes care of Q prev backward; and K, V prev's backward
        // if they are distinct
        attention->backward(gradientIn);
        if (KV_proxy)  // if K and V have same input, use proxy node to back-propagate
            KV_proxy->proxy_backward();
    }

    void print_desc()
    {
        LOG(BLUE, "Attention output size: ", this->shape, " for Q, K: ", Ql->shape,
            " V: ", Vl->shape, " Q.W and K.W: ", Ql->W.shape, " V.W.shape: ", Vl->W.shape);
    }

    virtual std::string dot_repr() override
    {
        uint32 learnable_count = Ql->param_count() + Kl->param_count() + Vl->param_count();
        std::stringstream ss;
        ss << " [label=\"" << this->name << "\", shape=box3d style=filled fillcolor=\"#4eb0f1\"]\n";

        std::set<uint32> input_ids{Ql->prev(0).id, Vl->prev(0).id, Kl->prev(0).id};
        NodePtrList<T> nodes = {
            Ql.get(), Kl.get(), Vl.get(), qkT.get(), attention_weights.get(), attention.get()};

        ss << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "["
           << num_to_si(learnable_count, true) << "]\"\n";
        for (auto& n : nodes) ss << '\t' << n->id << ' ';

        if (input_ids.size() == 1) ss << Ql->prev(0).id << "\n\t";
        ss << "\n\t{rank=same; " << Ql->id << ' ' << Kl->id << ' ' << Vl->id << " }"
           << "\n\t{rank=same; " << attention_weights->id << ' ' << attention->id << " }"
           << "\n\t" << this->id << "}\n";  // is attention

        return ss.str();
    }

    NodePtr<T> get_terminal_node() override { return attention.get(); }

    void save_weights(std::ostream& os) const override
    {
        Ql->save_weights(os);
        Kl->save_weights(os);
        Vl->save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        Ql->load_weights(is);
        Kl->load_weights(is);
        Vl->load_weights(is);
    }

    Linear<T>& Q() { return *Ql; }
    Linear<T>& K() { return *Kl; }
    Linear<T>& V() { return *Vl; }
};

template <typename T = FloatT>
struct SelfAttention : Node<T>
{
    NodePtr<T> prev;
    InputProxy<T> x = InputProxy<T>(prev, "SA-Input"); // trick untesteed for non-identity act or bias
    Attention<T> attn;

    SelfAttention(const LinearInput<T>& inp)
        : Node<T>(inp.prev->shape.set(WIDTH_IDX, inp.out_size), {inp.prev}, inp.name, 1),
          prev(inp.prev),
          attn({inp.out_size, &x, inp.useBias, inp.act_name, inp.name + "_Q"},
               {inp.out_size, &x, inp.useBias, inp.act_name, inp.name + "_K"},
               {inp.out_size, &x, inp.useBias, inp.act_name, inp.name + "_V"}, inp.name)
    {
        LOG(BLUE, "SAttn: ", this->name, this->shape, " for input: ", prev->name, prev->shape,
            " attention node: ", attn.name, attn.shape);
    }

    void forward() override
    {
        attn.compute();
        this->copy(attn.get_data().get());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        attn.backward(gradientIn);
        x.proxy_backward();
    }

    NodePtr<T> get_terminal_node() override { return &attn; }

    virtual std::string dot_repr() override { return attn.dot_repr(); }

    void save_weights(std::ostream& os) const override { attn.save_weights(os); }

    void load_weights(std::istream& is) override { attn.load_weights(is); }

    Linear<T>& Q() { return *attn.Ql; }
    Linear<T>& K() { return *attn.Kl; }
    Linear<T>& V() { return *attn.Vl; }
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
    using Att = Attention<T>;
    std::vector<std::unique_ptr<Att>> heads;
    std::unique_ptr<Concat0<T>> concat;
    std::unique_ptr<Linear<T>> linear;

    std::unique_ptr<InputProxy<T>> Q_proxy;
    std::unique_ptr<InputProxy<T>> K_proxy;  // could be null if KV_proxy or QKV_proxy is used
    std::unique_ptr<InputProxy<T>> V_proxy;  // could be null if KV_proxy or QKV_proxy is used

    std::unique_ptr<InputProxy<T>>
        KV_proxy;  // could be null if K_proxy and V_proxy is used or QKV_proxy is used
    std::unique_ptr<InputProxy<T>>
        QKV_proxy;  // could be null if Q_proxy, K_proxy and V_proxy is used

    MultiHeadAttention(uint32 num_heads, LinearInput<T> Qinp, LinearInput<T> Kinp,
                       LinearInput<T> Vinp, LinearInput<T> Oinp, std::string name = "MHA")
        : Node<T>({Qinp.prev->batch(), Qinp.prev->height(), Oinp.out_size},
                  {Qinp.prev, Kinp.prev, Vinp.prev}, name, 3)
    {
        if (num_heads == 1)
        {
            throw_rte_with_backtrace("num_heads should be greater than 1 for MultiHeadAttention ",
                                     num_heads, " != ", 1);
        }

        std::set<uint32> input_ids{Qinp.prev->id, Kinp.prev->id, Vinp.prev->id};

        if (input_ids.size() == 1)
        {
            LOG(YELLOW, "All inputs to ", name, " are same, using commong proxy node ");
            QKV_proxy = std::make_unique<InputProxy<T>>(Qinp.prev, name + "_QKV_proxy");
        }
        else if (input_ids.size() == 2)
        {
            if (Kinp.prev->id == Qinp.prev->id or Vinp.prev->id == Qinp.prev->id)
                throw_rte_with_backtrace("Q input cannot share input with either K or V inputs in ",
                                         name);

            Q_proxy = std::make_unique<InputProxy<T>>(Qinp.prev, name + "_Q_proxy");
            KV_proxy = std::make_unique<InputProxy<T>>(Kinp.prev, name + "_KV_proxy");
            Kinp.prev = KV_proxy.get();
            Vinp.prev = KV_proxy.get();
        }
        else if (input_ids.size() == 3)
        {
            Q_proxy = std::make_unique<InputProxy<T>>(Qinp.prev, name + "_Q_proxy");
            K_proxy = std::make_unique<InputProxy<T>>(Kinp.prev, name + "_K_proxy");
            V_proxy = std::make_unique<InputProxy<T>>(Vinp.prev, name + "_V_proxy");
        }

        NodePtrVec<T> head_ptrs;
        for (uint32 i = 0; i < num_heads; ++i)
        {
            auto att = new Att(Qinp, Kinp, Vinp, name + "_Head_" + std::to_string(i));
            heads.emplace_back(att);
            head_ptrs.push_back(att);
        }
        concat = std::make_unique<Concat0<T>>(head_ptrs, name + "_Concat");
        Oinp.prev = concat.get();
        Oinp.name = name + "_Linear";
        linear = std::make_unique<Linear<T>>(Oinp);
        this->set_data(linear->get_data());
        this->prev_nodes = linear->prev_nodes;
    }

    void forward() override {}

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        linear->backward(gradientIn);
        if (Q_proxy) Q_proxy->proxy_backward();
        if (K_proxy) K_proxy->proxy_backward();
        if (V_proxy) V_proxy->proxy_backward();
        if (KV_proxy) KV_proxy->proxy_backward();
        if (QKV_proxy) QKV_proxy->proxy_backward();
    }

    void print_desc()
    {
        LOG(BLUE, "MultiHeadAttention with ", heads.size(),
            " heads; Linear projection matrix shape: ", linear->W().shape,
            " to output: ", this->shape, " each attention looks like: ");
        heads[0]->print_desc();
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;

        ss << " [label=\"" << this->name << '\n'
           << linear->W.shape << ':' << linear->W.numels()
           << " \", shape=box3d,  style=filled, fillcolor=azure ]\n"
           << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "\"\n"
           << '\t' << concat->id << '\n'
           << '\t' << this->id << "\n}\n";
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

template <typename T = FloatT>
struct FeedForward : Node<T>
{
    std::unique_ptr<Linear<T>> l_in;
    std::unique_ptr<Dropout<T>> dropout;
    std::unique_ptr<Linear<T>> l_out;

    FeedForward(LinearInput<T> l1i, LinearInput<T> l2i, FloatT dropout_ratio,
                const std::string& name = "MLP")
        : Node<T>(l1i.prev->shape.set(WIDTH_IDX, l2i.out_size), {l1i.prev}, name, 1)
    {
        if (l2i.prev != nullptr)
        {
            throw_rte_with_backtrace(
                "MLP: Linear2 should not have a previous node (it's assigned to as yet "
                "non-existent Linear1)");
        }
        l_in = std::make_unique<Linear<T>>(l1i);
        dropout = std::make_unique<Dropout<T>>(dropout_ratio, l_in.get(), name + "_Dropout");
        l2i.prev = dropout.get();
        l_out = std::make_unique<Linear<T>>(l2i);

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
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
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
