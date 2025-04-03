#ifndef NODES_PARAMETERIZED_HPP
#define NODES_PARAMETERIZED_HPP

/*
Various nodes that have learnable parameters, currently they are either Linear<> or use Linear<>
*/

#include <cmath>
#include <cstdlib>
#include <memory>
#include "functors.cuh"
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
        auto* input_node = this->prev_nodes[0];
        LOG_NODE_TRACE(CYAN, "Linear::forward for ", RESET, this->name,
                       " with input: ", input_node->name, input_node->shape);
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
        LOG_NODE_TRACE(CYAN, "Linear-", get_act_name(act), "::backward for ", this->name,
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

        auto* input_node = this->prev_nodes[0];
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

// A proxy for an Linear node, used to pass input data to nodes. This does not mark input
// node as "prev", so that when backward is called, it does not backpropagate through to input node.
// This is helpfull in mitigating a compounding number of back-prop path. E.g. if 2 SelfAttention
// are connected sequentially, viz. x->SA1->SA2, x is input to all the Linear nodes inside SA1,
// When back-prop'ing 9 gradients that come to x will be:
//  sa2q->sa1q->x, sa2q->sa1k->x, sa2v->sa1v->x
//  sa2k->sa1q->x, sa2k->sa1k->x, sa2k->sa1v->x
//  sa2v->sa1q->x, sa2v->sa1k->x, sa2v->sa1v->x
//  Instead if we use LinearProxy and make the graph Proxy(x)->SA1->Proxy(SA1)->SA2, then only 3
//  gradients will be back-propagated to xp:
// 3 from SA2(q,k,v)->SA1(q,k,v)->Proxy(x). Now there will be 6 paths of length 2, instead of 9 of
// length 3 This effect becomes even more pronounced in MultiHeadAttention, where the number of
// paths get multiplied by the number of heads. this ::backward() only accumulates the gradient, so
// the owner of this node should call the ::proxy_backward() so that back-prop'ing happens to the
// actual input node
// This of course works only for Linear nodes, because gradient of sum is sum of gradients if
// transform is Linear;
template <typename T>
struct LinearProxy : Node<T>
{
    Linear<T>* in;
    Matrix<T> gradientOut = Matrix<T>(this->shape, this->name + "_gradientOut");
    LinearProxy(Linear<T>* prev, const std::string& name)
        : Node<T>(prev->shape, {}, name + "_proxy", 0), in(prev)
    {
        gradientOut.set_val(T(0));
        this->set_data(in->get_data());
    }
    void forward() override {}
    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("LinearProxy::backward for ", this->name,
                       " with gradientIn: ", gradientIn->name, gradientIn->shape,
                       "accumulating grads");
        binary_apply(gradientOut, *gradientIn, Plus<T>());
    }
    void proxy_backward()
    {
        LOG_NODE_TRACE("LinearProxy::proxy_backward for ", this->name,
                       " with gradientOut: ", gradientOut.name, gradientOut.shape);
        in->backward(&gradientOut);
        gradientOut.set_val(0.f);
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ret;
        ret << " [label=\"" << this->name
            << "\", shape=rect, style=filled, fillcolor=\"#b9cbd2\"]\n";
        ret << this->id << " -> " << in->id << " [style=dotted arrowhead=none]";
        return ret.str();
    }

    static LinearProxy<T>* get_proxy(const LinearInput<T>& inp)
    {
        if (auto linear = dynamic_cast<Linear<T>*>(inp.prev))
        {
            return new LinearProxy<T>(linear, inp.name);
        }
        return nullptr;
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
    Linear<T> Q;
    Linear<T> K;
    Linear<T> V;                        // The projection nodes.
    DividedBy<T> denom;                 // The denominator for scaling, sqrt(emb_size)
    ProductT<T, DividedBy<T>> qkT;      // The product of Q and K^T
    SoftmaxDim0<T> attention_weights;   // The softmax of qkT (along the dim=-1)
    Product<T, Identity<T>> attention;  // Product of Attention Weights and V

    Attention(const LinearInput<T>& Qinp, const LinearInput<T>& Kinp, const LinearInput<T>& Vinp,
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
        if (Qinp.out_size != Kinp.out_size)
            throw_rte_with_backtrace("Q and V output sizes do not match for Attention ",
                                     Qinp.out_size, " != ", Kinp.out_size);
        this->set_data(attention.get_data());
    }

    void forward() override
    {
        LOG_NODE_TRACE("Attention::forward for ", this->name, " with input: ", Q.prev(0).name,
                       Q.prev(0).shape);
        attention.compute();
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Attention::backward for ", this->name,
                       " with gradientIn: ", gradientIn->name);
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
        for (auto& n : nodes) ss << '\t' << n->id << ' ';

        if (input_ids.size() == 1) ss << Q.prev(0).id << "\n\t";
        ss << "\n\t{rank=same; " << Q.id << ' ' << K.id << ' ' << V.id << " }"
           << "\n\t{rank=same; " << attention_weights.id << ' ' << attention.id << " }"
           << "\n\t" << this->id << "}\n";  // is attention

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

    virtual std::vector<NodePtr<T>> get_dependencies() const override
    {
        return {Q.prev_nodes[0], K.prev_nodes[0], V.prev_nodes[0]};
    }
};

template <typename T = FloatT>
struct SelfAttention : Attention<T>  // Optimizes number of gradient paths when Q, K, V inputs are
                                     // same, using LinearProxy
{
    std::unique_ptr<LinearProxy<T>> x;
    SelfAttention(const LinearInput<T>& inp)
        : Attention<T>({inp.out_size, inp.prev, inp.useBias, inp.act_name, inp.name + "_Q"},
                       {inp.out_size, inp.prev, inp.useBias, inp.act_name, inp.name + "_K"},
                       {inp.out_size, inp.prev, inp.useBias, inp.act_name, inp.name + "_V"},
                       inp.name)
    {
        LOG(BLUE, "SAttn: ", this->name, this->shape, " for input: ", inp.prev->name,
            inp.prev->shape);
        if (auto proxy = LinearProxy<T>::get_proxy(inp))
        {
            x = std::unique_ptr<LinearProxy<T>>(proxy);
            // this->prev_nodes = {proxy->in};
            this->Q.prev_nodes = {x.get()};
            this->K.prev_nodes = {x.get()};
            this->V.prev_nodes = {x.get()};
        }
        else
            LOG(YELLOW, "SelfAttention: ", this->name, " with input: ", inp.prev->name,
                inp.prev->shape, " does not use LinearProxy");
    }

    void forward() override
    {
        LOG_NODE_TRACE("SelfAttention::forward for ", this->name);
        if (x) x->in->compute();
        Attention<T>::forward();
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("SelfAttention::backward for ", this->name,
                       " with gradientIn: ", gradientIn->name);
        Attention<T>::backward(gradientIn);
        if (x) x->proxy_backward();
    }

    virtual std::vector<NodePtr<T>> get_dependencies() const override { return {x->in}; }
};

template <typename T = FloatT>
struct CrossAttention : Attention<T>  // Optimizes number of gradient paths using LinearProxy, when
                                      // K & V inputs are same, and act is IActivation
{
    std::unique_ptr<LinearProxy<T>> KV_proxy;
    CrossAttention(const LinearInput<T>& Qinp, const LinearInput<T>& KVinp,
                   std::string name = "CrossAttention")
        : Attention<T>(
              {Qinp.out_size, Qinp.prev, Qinp.useBias, Qinp.act_name, Qinp.name + "_Q"},
              {KVinp.out_size, KVinp.prev, KVinp.useBias, KVinp.act_name, KVinp.name + "_K"},
              {KVinp.out_size, KVinp.prev, KVinp.useBias, KVinp.act_name, KVinp.name + "_V"}, name)
    {
        LOG(BLUE, "CrossAttn: ", this->name, this->shape, " for input Q: ", Qinp.prev->name,
            Qinp.prev->shape, " and KV: ", KVinp.prev->name, KVinp.prev->shape);
        if (auto proxy = LinearProxy<T>::get_proxy(KVinp))
        {
            KV_proxy = std::unique_ptr<LinearProxy<T>>(proxy);
            //            this->prev_nodes = {proxy->in};
            this->K.prev_nodes = {KV_proxy.get()};
            this->V.prev_nodes = {KV_proxy.get()};
        }
        else
            LOG(YELLOW, "CrossAttention: ", this->name, " with input: ", KVinp.prev->name,
                KVinp.prev->shape, " does not use LinearProxy");
    }

    void forward() override
    {
        LOG_NODE_TRACE("CrossAttention::forward for ", this->name);
        if (KV_proxy) KV_proxy->in->compute();
        Attention<T>::forward();
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name);
        Attention<T>::backward(gradientIn);
        if (KV_proxy) KV_proxy->proxy_backward();
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;
        ss << Attention<T>::dot_repr() << "subgraph cluster_" << Attention<T>::id << "{"
           << this->KV_proxy->id << "}";
        return ss.str();
    }

    virtual std::vector<NodePtr<T>> get_dependencies() const
    {
        std::vector<NodePtr<T>> deps = {this->Q.prev_nodes[0]};
        if (KV_proxy) deps.push_back(KV_proxy->in);
        return deps;
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
    using Att = Attention<T>;
    std::vector<std::unique_ptr<Att>> heads;
    std::unique_ptr<Concat0<T>> concat;
    std::unique_ptr<Linear<T>> linear;

    MultiHeadAttention(uint32 num_heads, LinearInput<T> Qinp, LinearInput<T> Kinp,
                       LinearInput<T> Vinp, LinearInput<T> Oinp, std::string name = "MHA")
        : Node<T>({Qinp.prev->batch(), Qinp.prev->height(), Oinp.out_size},
                  {Qinp.prev, Kinp.prev, Vinp.prev}, name, 3)
    {
        LOG(BLUE, "MultiHeadAttention: ", this->name, " with input Qinp: ", Qinp.prev->name,
            Qinp.prev->shape, " Kinp: ", Kinp.prev->name, Kinp.prev->shape,
            " Vinp: ", Vinp.prev->name, Vinp.prev->shape, " Oinp: ", Oinp.prev->name,
            Oinp.prev->shape);
        if (num_heads == 1)
        {
            throw_rte_with_backtrace("num_heads 1 , use Attention instead");
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
    }

    void forward() override { linear->compute(); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        linear->backward(gradientIn);
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

    virtual std::vector<NodePtr<T>> get_dependencies() const override
    {
        return heads[0]->get_dependencies();
    }
};

template <typename T = FloatT>
struct MultiHeadSelfAttention : MultiHeadAttention<T>
{
    std::unique_ptr<LinearProxy<T>> x;
    MultiHeadSelfAttention(uint32 num_heads, LinearInput<T> Linp, LinearInput<T> Oinp,
                           std::string name = "MHSA")
        : MultiHeadAttention<T>(num_heads, Linp, Linp, Linp, Oinp, name),
          x(Linp.prev, name + "_LinearProxy")
    {
        LOG(BLUE, "MultiHeadSelfAttention: ", this->name, " with input Linp: ", Linp.prev->name,
            Linp.prev->shape, " Oinp: ", Oinp.prev->name, Oinp.prev->shape);

        if (auto proxy = LinearProxy<T>::get_proxy(Linp))
        {
            x = std::unique_ptr<LinearProxy<T>>(proxy);
            for (uint32 i = 0; i < num_heads; ++i)
            {
                this->heads[i]->Q.prev_nodes = {x.get()};
                this->heads[i]->K.prev_nodes = {x.get()};
                this->heads[i]->V.prev_nodes = {x.get()};
            }
        }
    }
    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        MultiHeadAttention<T>::backward(gradientIn);
        if (x) x->proxy_backward();
    }
};

template <typename T = FloatT>
struct MultiHeadCrossAttention : MultiHeadAttention<T>
{
    std::unique_ptr<LinearProxy<T>> x;
    MultiHeadCrossAttention(uint32 num_heads, LinearInput<T> Qinp, LinearInput<T> KVinp,
                            LinearInput<T> Oinp, std::string name = "MHXA")
        : MultiHeadAttention<T>(num_heads, Qinp, KVinp, KVinp, Oinp, name),
          x(Qinp.prev, name + "_LinearProxy")
    {
        if (auto proxy = LinearProxy<T>::get_proxy(KVinp))
        {
            x = std::unique_ptr<LinearProxy<T>>(proxy);
            for (uint32 i = 0; i < num_heads; ++i)
            {
                this->heads[i]->K.prev_nodes = {x.get()};
                this->heads[i]->V.prev_nodes = {x.get()};
            }
        }
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
