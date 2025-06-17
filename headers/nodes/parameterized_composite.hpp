/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef PARAMETERIZED_COMPOSITE_HPP
#define PARAMETERIZED_COMPOSITE_HPP

#include <memory>
#include <string>
#include "matrix.cuh"
#include "node.hpp"
#include "nodes/parameterized.hpp"
/*
File contains parameterized composite nodes, which are nodes that contain other attention &&linear
nodes.
*/

/*
MultiHeadAttention:
Input is a std::vector of 3 matrices, each of size `S x Ei`, where S is the sequence length.
With `n_heads`, each head projects querys &&keys to `S x q_size`
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

    // Oinp.prev &&Oinp.name are ignored
    MultiHeadAttention(uint32 num_heads, LinearInput<T> Qinp, LinearInput<T> Kinp,
                       LinearInput<T> Vinp,
                       LinearInput<T> Oinp,  // Oinp.prev &&Oinp.name are ignored
                       std::string name = "MHA")
        : Node<T>({Qinp.prev->batch(), Qinp.prev->height(), Oinp.out_size}, {}, name, 0)
    {
        if (num_heads == 1)
        {
            throw_rte_with_backtrace("num_heads 1 , use Attention instead");
        }

        NodePtrVec<T> head_ptrs;
        for (uint32 i = 0; i < num_heads; ++i)
        {
            std::string head_str = "_h" + std::to_string(i);
            heads.emplace_back(std::make_unique<Att>(
                Qinp.set_name(Qinp.name + head_str), Kinp.set_name(Kinp.name + head_str),
                Vinp.set_name(Vinp.name + head_str), name + head_str));
            head_ptrs.push_back(heads.back().get());
        }
        concat = std::make_unique<Concat0<T>>(head_ptrs, name + "_Concat");
        Oinp.prev = concat.get();
        Oinp.name = name + "_Out";
        linear = std::make_unique<Linear<T>>(Oinp);

        LOG_NODE_NAME(Qinp.prev->shape, R_JUST("->", 4), linear->shape);
        this->set_data(linear->get_data());
    }

    void forward(Context* ctx) override
    {
        linear->compute(ctx);
        this->copy_extents(*linear);
    }

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

    virtual NodePtrVec<T> get_dependencies() const override { return heads[0]->get_dependencies(); }

    Att* get_head(uint32 i) { return heads[i].get(); }

    virtual uint32 param_count() override
    {
        auto head_count =
            std::accumulate(heads.begin(), heads.end(), 0,
                            [](uint32 sum, const auto& head) { return sum + head->param_count(); });
        auto linear_count = linear->param_count();
        return head_count + linear_count;
    }

    NodePtr<T> get_terminal_node() override { return linear.get(); }

    virtual std::string type() const override { return "MultiHeadAttn"; }
};

template <typename T = FloatT>
struct MultiHeadSelfAttention : MultiHeadAttention<T>
{
    std::unique_ptr<LinearProxy<T>> x;
    MultiHeadSelfAttention(uint32 num_heads, LinearInput<T> Linp,
                           LinearInput<T> Oinp,  // Oinp.prev &&Oinp.name are ignored
                           std::string name = "MHSA")
        : MultiHeadAttention<T>(num_heads, Linp.set_name(Linp.name + "_Q"),
                                Linp.set_name(Linp.name + "_K"), Linp.set_name(Linp.name + "_V"),
                                Oinp.set_name(name + "_O"), name)
    {
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

    void forward(Context* ctx) override
    {
        if (x) x->in->compute(ctx);
        MultiHeadAttention<T>::forward(ctx);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        MultiHeadAttention<T>::backward(gradientIn, ctx);
        if (x) x->proxy_backward(ctx);
    }

    virtual NodePtrVec<T> get_dependencies() const override
    {
        if (x) return {x->in};
        return MultiHeadAttention<T>::get_dependencies();
    }

    virtual std::string type() const override { return "MultiHeadSelfAttention"; }
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

    virtual std::string type() const override { return "MultiHeadCrossAttention"; }
};

/* FeedForward block:
    Four versions:
       V0: Prev-> Linear1 -> Dropout1 -> Linear2 -> Dropout2 -> LayerNorm
       V1: Prev-> Linear1 -> Dropout1 -> Residual -> Linear2 -> Dropout2 -> LayerNorm
              \___________________________/
       V2: Prev-> Linear1 -> Dropout1 -> Residual1 -> Linear2 -> Dropout2 -> Residual2 -> LayerNorm
                                                 \___________________________/
       V3: Prev-> Linear1 -> Dropout1 -> Residual1 -> Linear2 -> Dropout2 -> Residual2 -> LayerNorm
              \___________________________/     \___________________________/

    V0: No residual connections, V1: Residual across Linear1, V2: Residual Linear2, V3: Residual
   across both

    if PWidth == IWidth &&IWidth == OWidth, then V3
    else if PWidth != IWidth &&IWidth == OWidth, then V2
    else if PWidth == IWidth &&IWidth != OWidth, then V1
    else V0
*/
template <typename T = FloatT>
struct FeedForward : Node<T>
{
    FeedForward(LinearInput<T> l_inp1,  // ::out_size is ignored, intermediate_dim is out_size of
                                        // Linear1, name is ignoredst
                FloatT dropout1_rate, uint32 intermediate_dim,
                LinearInput<T> l_inp2,  // ::prev is ignored, prev is either residual1 || dropout1,
                                        // name is ignored
                FloatT dropout2_rate, std::string name = "FeedForward")
        : Node<T>(l_inp1.prev->shape.set(WIDTH_IDX, l_inp2.out_size), {}, name, 0)
    {
        uint32 prev_width = l_inp1.prev->width();
        uint32 out_width = l_inp2.out_size;
        l_inp1.out_size = intermediate_dim;

        linear1 = std::make_unique<Linear<T>>(l_inp1);
        dropout1 = std::make_unique<Dropout<T>>(dropout1_rate, linear1.get(), name + "_Dropout1");

        l_inp2.prev = dropout1.get();
        if (prev_width == intermediate_dim)
        {
            residual1 = std::make_unique<Add<T>>(NodePtrVec<T>{l_inp1.prev, dropout1.get()},
                                                 name + "_Residual1");
            l_inp2.prev = residual1.get();
        }

        linear2 = std::make_unique<Linear<T>>(l_inp2);
        dropout2 = std::make_unique<Dropout<T>>(dropout2_rate, linear2.get(), name + "_Dropout2");
        NodePtr<T> prev = dropout2.get();
        if (intermediate_dim == out_width)
        {
            residual2 = std::make_unique<Add<T>>(NodePtrVec<T>{l_inp2.prev, dropout2.get()},
                                                 name + "_Residual2");
            prev = residual2.get();
        }
        layer_norm = std::make_unique<Normalize<T, WIDTH_IDX>>(prev, name + "_LayerNorm");
        this->set_data(layer_norm->get_data());
        LOG_NODE_NAME(l_inp1.prev->shape, R_JUST("->", 4), linear2->shape);
    }

    void forward(Context* ctx) override
    {
        LOG_NODE_TRACE("FeedForward::forward for ", this->name);
        layer_norm->compute(ctx);
        this->copy_extents(*layer_norm);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("FeedForward::backward for ", this->name);
        layer_norm->backward(gradientIn, ctx);
    }

    virtual std::vector<NodePtr<T>> get_dependencies() const override
    {
        return linear1->prev_nodes;
    }

    virtual NodePtr<T> get_terminal_node() override { return layer_norm.get(); }

    virtual uint32 param_count() override
    {
        return linear1->param_count() + linear2->param_count();
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ss;
        uint32 learnable_count = linear1->param_count() + linear2->param_count();
        ss << "subgraph cluster_" << this->id << "{\n    label = \"" << this->name << "["
           << num_to_si(learnable_count, true) << "]\"\n";
        // add internal nodes
        NodePtrVec<T> nodes = {linear1.get(), dropout1.get(), linear2.get(), dropout2.get(),
                               layer_norm.get()};
        if (residual1) nodes.push_back(residual1.get());
        if (residual2) nodes.push_back(residual2.get());
        for (auto& n : nodes)
        {
            if (n) ss << '\t' << n->id << ' ';
        }
        ss << "{rank = max; " << this->id << ";}\n";
        ss << "\n\t" << this->id << "}\n";
        if (residual1) ss << residual1->dot_repr() << '\n';
        if (residual2) ss << residual2->dot_repr() << '\n';
        ss << this->id << " [label = \"" << this->name << "[" << num_to_si(learnable_count, true)
           << "]\", shape=box3d,  style=filled, fillcolor=azure ]\n";
        return ss.str();
    }

    virtual std::string type() const override { return "FeedForward"; }

    void save_weights(std::ostream& os) const override
    {
        linear1->save_weights(os);
        linear2->save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        linear1->load_weights(is);
        linear2->load_weights(is);
    }

 private:
    std::unique_ptr<Linear<T>> linear1;
    std::unique_ptr<Dropout<T>> dropout1;
    std::unique_ptr<Add<T>> residual1;
    std::unique_ptr<Linear<T>> linear2;
    std::unique_ptr<Dropout<T>> dropout2;
    std::unique_ptr<Add<T>> residual2;
    std::unique_ptr<Normalize<T, WIDTH_IDX>> layer_norm;
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
        this->copy_extents(layer_norm);
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

    virtual std::string type() const override { return "SAEncoder"; }

    void save_weights(std::ostream& os) const override
    {
        mhsa.save_weights(os);
        ff_in.save_weights(os);
        ff_out.save_weights(os);
        layer_norm.save_weights(os);
    }

    void load_weights(std::istream& is) override
    {
        mhsa.load_weights(is);
        ff_in.load_weights(is);
        ff_out.load_weights(is);
        layer_norm.load_weights(is);
    }
};

#endif  // PARAMETERIZED_COMPOSITE_HPP
