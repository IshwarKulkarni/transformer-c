#ifndef PARAMETERIZED_COMPOSITE_HPP
#define PARAMETERIZED_COMPOSITE_HPP

#include <memory>
#include <string>
#include "node.hpp"
#include "nodes/parameterized.hpp"
/*
File contains parameterized composite nodes, which are nodes that contain other attention and linear
nodes.
*/

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
    std::unique_ptr<Linear<T>> linear;
    std::unique_ptr<Concat0<T>> concat;
    std::vector<std::unique_ptr<Att>> heads;

    // Oinp.prev and Oinp.name are ignored
    MultiHeadAttention(uint32 num_heads, LinearInput<T> Qinp, LinearInput<T> Kinp,
                       LinearInput<T> Vinp, LinearInput<T> Oinp, std::string name = "MHA")
        : Node<T>({Qinp.prev->batch(), Qinp.prev->height(), Oinp.out_size}, {}, name, 0)
    {
        LOG(BLUE, "MultiHeadAttention: ", this->name, " with input Qinp: ", Qinp.prev->name);
        LOG(BLUE, Qinp.prev->shape, " Kinp: ", Kinp.prev->name, Kinp.prev->shape,
            " Vinp: ", Vinp.prev->name, Vinp.prev->shape, " Out Shape: ", this->shape);
        if (num_heads == 1)
        {
            throw_rte_with_backtrace("num_heads 1 , use Attention instead");
        }

        NodePtrVec<T> head_ptrs;
        for (uint32 i = 0; i < num_heads; ++i)
        {
            heads.emplace_back(
                std::make_unique<Att>(Qinp, Kinp, Vinp, name + "_Head_" + std::to_string(i)));
            head_ptrs.push_back(heads.back().get());
        }
        concat = std::make_unique<Concat0<T>>(head_ptrs, name + "_Concat");
        Oinp.prev = concat.get();
        Oinp.name = name + "_Linear";
        linear = std::make_unique<Linear<T>>(Oinp);
        this->set_data(linear->get_data());
    }

    void forward(Context* ctx) override { linear->compute(ctx); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        linear->backward(gradientIn, ctx);
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;

        ss << " [label=\"" << this->name << '\n'
           << linear->W.shape << ':' << linear->W.numels()
           << " \", shape=box3d,  style=filled, fillcolor=azure ]\n"
           << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "\"\n"
           << '\t' << concat->id << '\n'
           << '\t' << linear->id << '\n'
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

    Att* get_head(uint32 i) { return heads[i].get(); }

    virtual uint32 param_count() override
    {
        return std::accumulate(
                   heads.begin(), heads.end(), 0,
                   [](uint32 sum, const auto& head) { return sum + head->param_count(); }) +
               linear->param_count();
    }

    NodePtr<T> get_terminal_node() override { return linear.get(); }
};

template <typename T = FloatT>
struct MultiHeadSelfAttention : MultiHeadAttention<T>
{
    std::unique_ptr<LinearProxy<T>> x;
    MultiHeadSelfAttention(uint32 num_heads, LinearInput<T> Linp, LinearInput<T> Oinp,
                           std::string name = "MHSA")
        : MultiHeadAttention<T>(num_heads, Linp, Linp, Linp, Oinp, name)
    {
        LOG(BLUE, "MultiHeadAttention: ", this->name, " with input Linp: ", Linp.prev->name);

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
        else
            LOG(YELLOW, "High branching in gradient paths starting from ", this->name);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        MultiHeadAttention<T>::backward(gradientIn, ctx);
        if (x) x->proxy_backward(ctx);
    }
};

template <typename T = FloatT>
struct MultiHeadCrossAttention : MultiHeadAttention<T>
{
    std::unique_ptr<LinearProxy<T>> x;
    MultiHeadCrossAttention(uint32 num_heads, LinearInput<T> Qinp, LinearInput<T> KVinp,
                            LinearInput<T> Oinp, std::string name = "MHXA")
        : MultiHeadAttention<T>(num_heads, Qinp, KVinp, KVinp, Oinp, name)
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
        else
            LOG(YELLOW, "High branching in gradient paths starting from ", this->name);
    }

    virtual void forward(Context* ctx) override
    {
        if (x) x->in->compute(ctx);
        MultiHeadAttention<T>::forward(ctx);
    }

    virtual void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        MultiHeadAttention<T>::backward(gradientIn, ctx);
        if (x) x->proxy_backward(ctx);
    }

    virtual std::vector<NodePtr<T>> get_dependencies() const override
    {
        auto out = MultiHeadAttention<T>::get_dependencies();
        out.push_back(x.get());
        return out;
    }
};

// FeedForward block: Linear -> Dropout -> Linear -> Dropout -> LayerNorm
template <typename T = FloatT>
struct FeedForward : Node<T>
{
    Linear<T> linear1;
    Dropout<T> dropout1;
    std::unique_ptr<Add<T>> residual1;
    Linear<T> linear2;
    Dropout<T> dropout2;
    std::unique_ptr<Add<T>> residual2;
    Normalize<T, WIDTH_IDX> layer_norm;

    // inp.out_size is the output size of this block, intermediate_dim is the hidden size of the
    // linear layer Does not use residual connections
    FeedForward(const LinearInput<T>& inp, uint32 intermediate_dim, float32 dropout_rate1,
                float32 dropout_rate2, std::string name)
        : Node<T>(inp.prev->shape.set(WIDTH_IDX, inp.out_size), {}, name, 0),
          linear1(intermediate_dim, inp.prev, inp.useBias, inp.act_name, name + "_L1"),
          dropout1(dropout_rate1, &linear1, name + "_D1"),
          linear2(inp.out_size, &dropout1, inp.useBias, "identity", name + "_L2"),
          dropout2(dropout_rate2, &linear2, name + "_D2"),
          layer_norm(&dropout2, name + "_norm")
    {
        LOG(BLUE, "FeedForward: ", this->name, " with input: ", inp.prev->name, inp.prev->shape,
            " no residual connections");
        this->set_data(linear2.get_data());
    }

    // residual version, intermediate_dim is same as inp.out_size
    FeedForward(const LinearInput<T>& inp, float32 dropout_rate1, float32 dropout_rate2,
                std::string name)
        : Node<T>(inp.prev->shape.set(WIDTH_IDX, inp.out_size), {}, name, 0),
          linear1(inp.out_size, inp.prev, inp.useBias, inp.act_name, name + "_L1"),
          dropout1(dropout_rate1, &linear1, name + "_D1"),
          residual1(new Add<T>({inp.prev, &dropout1}, name + "_residual1")),
          linear2(inp.out_size, residual1.get(), inp.useBias, "identity", name + "_L2"),
          dropout2(dropout_rate2, &linear2, name + "_D2"),
          residual2(new Add<T>(NodePtrVec<T>{residual1.get(), &dropout2}, name + "_residual2")),
          layer_norm(residual2.get(), name + "_norm")
    {
        LOG(BLUE, "FeedForward residual: ", this->name, " with input: ", inp.prev->name,
            inp.prev->shape);
        this->set_data(linear2.get_data());
    }

    void forward(Context* ctx) override
    {
        LOG_NODE_TRACE("FeedForward::forward for ", this->name);
        layer_norm.compute(ctx);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("FeedForward::backward for ", this->name);
        layer_norm.backward(gradientIn, ctx);
    }

    virtual std::vector<NodePtr<T>> get_dependencies() const override { return linear1.prev_nodes; }

    virtual NodePtr<T> get_terminal_node() override { return &layer_norm; }

    virtual uint32 param_count() override { return linear1.param_count() + linear2.param_count(); }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;
        uint32 learnable_count = linear1.param_count() + linear2.param_count();
        ss << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "["
           << num_to_si(learnable_count, true) << "]\"\n";
        // add internal nodes
        NodePtrVec<T> nodes = {&linear1, &dropout1, &linear2, &dropout2, &layer_norm};
        if (residual1) nodes.push_back(residual1.get());
        if (residual2) nodes.push_back(residual2.get());
        for (auto& n : nodes) ss << '\t' << n->id << ' ';
        ss << "{rank = max; " << this->id << ";}\n";
        ss << "\n\t" << this->id << "}\n";
        if (residual1) ss << residual1->dot_repr() << '\n';
        if (residual2) ss << residual2->dot_repr() << '\n';
        ss << this->id << " [label = \"" << this->name << "[" << num_to_si(learnable_count, true)
           << "]\", shape=box3d,  style=filled, fillcolor=azure ]\n";
        return ss.str();
    }
};

template <typename T = FloatT>
struct SAEncoder : Node<T>
{
    MultiHeadSelfAttention<T> mhsa;  // Outlinear is same output dim as input
    FeedForward<T> ff_in;            // same input dim as MHA output
    Dropout<T> dropout1;
    FeedForward<T> ff_out;
    Dropout<T> dropout2;
    Normalize<T, WIDTH_IDX> layer_norm;

    SAEncoder(uint32 num_heads, LinearInput<T> in_linearinput, LinearInput<T> mha_outlinear_input,
              LinearInput<T> ff_in_inp, LinearInput<T> ff_out_inp, FloatT ff_in_dropout = 0.15f,
              FloatT ff_out_dropout = 0.15f, FloatT dropout1_rate = 0.15f,
              FloatT dropout2_rate = 0.15f, std::string name = "SAEncoder")
        : Node<T>(ff_out_inp.out_size, {in_linearinput.prev}, name, 1),
          mhsa(num_heads, in_linearinput, mha_outlinear_input, name + "_MHA"),
          ff_in(ff_in_inp, ff_in_dropout, name + "_FF_In"),
          dropout1(dropout1_rate, name + "_Dropout1"),
          ff_out(ff_out_inp, ff_out_dropout, name + "_FF_Out"),
          dropout2(dropout2_rate, name + "_Dropout2"),
          layer_norm(name + "_LayerNorm")
    {
        LOG(BLUE, "SAEncoder: ", this->name, " with input shape: ", in_linearinput.prev->shape);
        this->set_data(layer_norm.get_data());
    }

    void forward(Context* ctx) override
    {
        LOG_NODE_TRACE("SAEncoder::forward for ", this->name);
        layer_norm.compute(ctx);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("SAEncoder::backward for ", this->name);
        layer_norm.backward(gradientIn, ctx);
    }

    virtual std::string dot_repr() override
    {
        uint32 learnable_count = mhsa.param_count() + ff_in.param_count() + ff_out.param_count() +
                                 layer_norm.param_count();

        std::stringstream ss;
        ss << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "["
           << num_to_si(learnable_count, true) << "]\"\n";
        return ss.str();
    }
};

#endif  // PARAMETERIZED_COMPOSITE_HPP
