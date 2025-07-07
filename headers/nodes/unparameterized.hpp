/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef NODES_UNPARAMETERIZED_HPP
#define NODES_UNPARAMETERIZED_HPP

/*
Various Nodes with no learnable parameters, these include softmaxes, matrix products, norms and
means.
*/

#include <cmath>
#include <sstream>
#include "logger.hpp"
#include "matrix.cuh"
#include "node.hpp"
#include "nodes/elemwise.hpp"
#include "types"

/*
Implements softmax along width of x => each output column sum to 1 <br>
```
inp = this_prev_0_
assert inp.dim == 2
s = inp.exp() / inp.exp().sum(dim=0, keepdim=True)
```
*/
template <typename T = FloatT>
struct SoftmaxDim1 : Node<T>
{
    Matrix<T> exp = Matrix<T>(this->shape.t());
    Matrix<T> sumExps = Matrix<T>(exp.shape.set(0, 1));
    Matrix<T> softmax = Matrix<T>(exp.shape);
    Matrix<T> gradientOut = Matrix<T>(this->shape, this->name + "_gradientOut");
    Matrix<T> gradientInT = Matrix<T>(this->shape.t(), this->name + "_gradientOutT");
    Exp<T> ExpOp = Exp<T>(std::sqrt(this->height()));

    SoftmaxDim1(NodePtr<T> prev, const std::string& name = "SftMxDim1")
        : Node<T>(prev->shape, {prev}, name, 1)
    {
        if (prev->height() == 1)
        {
            throw_rte_with_backtrace("Dim[1] of previous node, ", prev->name, prev->shape,
                                     " is 1 softmaxDim1 is invalid");
        }
        LOG(BLUE, this->name, "\t", prev->shape, " reduced along HEIGHT [", prev->height(), "]");
    }

    // computes softmax along height of x => each output column sums to 1
    void forward(Context* ctx) override
    {
        (void)ctx;
        transpose<FloatT, Exp<T>>(exp, this->prev(0), ExpOp);
        reduce(sumExps, exp);
        binary_apply(softmax, exp, sumExps, Div<T>());
        transpose(*this, softmax);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        transpose(gradientInT, *gradientIn);
        softmax_gradient(gradientOut, softmax, gradientInT);
        this->prev_nodes[0]->backward(&gradientOut, ctx);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=LightSkyBlue, shape=rect] ";
    }

    virtual std::string type() const override { return "SoftmaxDim1"; }
};

// computes softmax along Width of x => each output row sums to 1, equivalent to
// torch.Softmax(dim=-1)
template <typename T = FloatT>
struct SoftmaxDim0 : Node<T>
{
    Matrix<T> exp = Matrix<T>(this->shape);
    Matrix<T> sumExps = Matrix<T>(exp.shape.set(0, 1));
    Matrix<T> gradientOut = Matrix<T>(this->shape.t(), this->name + "_gradientOut");
    Matrix<T> gradientOutT = Matrix<T>(this->shape, this->name + "_gradientOutT");
    Matrix<T> outT = Matrix<T>(this->shape.t(), "outT");
    Exp<T> expOp = Exp<T>(std::sqrt(this->width()));

    SoftmaxDim0(NodePtr<T> prev, const std::string& name = "SftMxDim0")
        : Node<T>(prev->shape, {prev}, name, 1)
    {
        if (prev->width() == 1)
        {
            throw_rte_with_backtrace("Dim[0] of previous node, ", prev->name, prev->shape,
                                     " is 1 softmaxDim0 cannot be used");
        }
    }

    void forward(Context* ctx) override
    {
        (void)ctx;
        unary_apply(exp, this->prev(0), expOp);
        reduce(sumExps, exp);
        binary_apply(*this, exp, sumExps, Div<T>());
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        softmax_gradient(gradientOut, *this, *gradientIn);
        transpose(gradientOutT, gradientOut);
        this->prev_nodes[0]->backward(&gradientOutT, ctx);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=LightSkyBlue, shape=rect] ";
    }

    virtual std::string type() const override { return "SoftmaxDim0"; }
};

template <typename T, typename PostProcess>
struct Product : Node<T>
{
    Matrix<T> aT, a_grad_in, b_grad_in;
    PostProcess pProcess;
    Composition<T, Neg<T>, PostProcess> pProcessN = {Neg<T>(), pProcess};

    Product(NodePtrVec<T> prevs, PostProcess pProcess, const std::string& name)
        : Node<T>({prevs[0]->batch(), prevs[0]->height(), prevs[1]->width()}, prevs, name, 2),
          aT(this->prev(0).shape.t()),
          a_grad_in(this->prev(0).shape),
          b_grad_in(this->prev(1).shape),
          pProcess(pProcess)
    {
        if (this->prev(0).width() != this->prev(1).height())
            throw_rte_with_backtrace("Matrix dimensions do not match for product between ",
                                     this->prev(0).shape, " and ", this->prev(1).shape);
    }

    void forward(Context*) override { mmadd(*this, this->prev(0), this->prev(1), {}, pProcess); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        transpose(aT, this->prev(0), Neg<T>());
        mmTadd(a_grad_in, *gradientIn, this->prev(1), {}, pProcess);
        mmadd(b_grad_in, aT, *gradientIn, {}, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_in, ctx);
        this->prev_nodes[1]->backward(&b_grad_in, ctx);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=azure, shape=rect] ";
    }

    virtual std::string type() const override { return "Product"; }
};

/* Implements a multiplication between
    matrix and transpose of another: output = A * B^T

 Here's an equivalent python code:
 def Product(a, b):
    assert(a.shape[1] == b.shape[0])
    return torch.mm(a, b.t())
*/
template <typename T, typename PostProcess = Identity<T>>
struct ProductT : Node<T>
{
    Matrix<T> a_grad_inN, b_grad_in;
    Matrix<T> gradInT;
    PostProcess pProcess;
    Composition<T, Neg<T>, PostProcess> pProcessN = {Neg<T>(), pProcess};

    ProductT(NodePtrVec<T> prevs, PostProcess pProcess, const std::string& name)
        : Node<T>({prevs[0]->batch(), prevs[0]->height(), prevs[1]->height()}, prevs, name, 2),
          a_grad_inN(this->prev(0).shape),
          b_grad_in(this->prev(1).shape),
          gradInT(this->shape.t()),
          pProcess(pProcess)
    {
        if (this->prev(0).width() != this->prev(1).width())
            throw_rte_with_backtrace("Matrix dimensions do not match for ProductT between ",
                                     this->prev(0).name, this->prev(0).shape, " and ",
                                     this->prev(1).name, this->prev(1).shape);
    }

    void forward(Context*) override { mmTadd(*this, this->prev(0), this->prev(1), {}, pProcess); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape,
                       " and prev1: ", this->prev(1).name, this->prev(1).shape);
        mmadd(a_grad_inN, *gradientIn, this->prev(1), {}, pProcess);
        transpose(gradInT, *gradientIn, Neg<T>());
        mmadd(b_grad_in, gradInT, this->prev(0), {}, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_inN, ctx);
        this->prev_nodes[1]->backward(&b_grad_in, ctx);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=azure, shape=rect] ";
    }

    virtual std::string type() const override { return "ProductT"; }
};

template <typename T = FloatT>
struct Add : Node<T>
{
    Add(NodePtrVec<T> prevs, const std::string& name) : Node<T>(prevs[0]->shape, prevs, name, 2)
    {
        if (prevs[0]->shape != prevs[1]->shape)
            throw_rte_with_backtrace("Matrix dimensions do not match for Plus between ",
                                     prevs[0]->name, prevs[0]->shape, " and ", prevs[1]->name,
                                     prevs[1]->shape);
        LOG_NODE_NAME(this->shape);
    }

    void forward(Context*) override
    {
        binary_apply(*this, this->prev(0), this->prev(1), Plus<T>());
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape,
                       " and prev1: ", this->prev(1).name, this->prev(1).shape);
        this->prev_nodes[0]->backward(gradientIn, ctx);
        this->prev_nodes[1]->backward(gradientIn, ctx);
    }

    virtual std::string type() const override { return "Add"; }

    virtual std::string dot_repr() override
    {
        std::string xlabel = " xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\"> Add </font>>";
        return " [label=\"" + this->name + "\", shape=rect, style=filled, fillcolor=lightblue" +
               xlabel + "]";
    }
};

template <typename T = FloatT>
struct Transpose : Node<T>
{
    Matrix<T> gradientOut = Matrix<T>(this->shape.t(), this->name + "_gradientOut");

    Transpose(NodePtr<T> prev, const std::string& name) : Node<T>(prev->shape.t(), {prev}, name, 1)
    {
        if (this->prev(0).shape.t() != this->shape)
            throw_rte_with_backtrace("Matrix dimensions do not match for Transpose between ",
                                     prev->name, " and ", this->name);

        LOG(BLUE, this->name, "\t", prev->shape, " -> ", this->shape);
    }

    void forward(Context* ctx) override { transpose(*this, this->prev(0)); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with  gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        transpose(gradientOut, *gradientIn);
        this->prev_nodes[0]->backward(&gradientOut, ctx);
    }

    virtual std::string type() const override { return "Transpose"; }
};

template <typename T = FloatT, uint32 Dim = 0>
struct MeanUnext : Node<T>
{
    MeanUnext(NodePtr<T> prev, const std::string& name = "Average")
        : Node<T>(prev->shape.set(Dim, 1), {prev}, name, 1),
          divOp(DividedBy<T>(prev->shape[Dim])),
          gradientOut(prev->shape, name + "_gradientOut")
    {
        if (prev->shape[Dim] == 1)
            throw_rte_with_backtrace("Cannot reduce along dimension ", Dim, " for ", prev->name,
                                     prev->shape, " already 1");
        LOG_NODE_NAME(prev->shape, R_JUST("->", 4), this->shape);
    }

    void forward(Context*) override
    {
        reduce<T, Dim>(*this, this->prev(0), Plus<T>(), T(0), divOp);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        unary_apply(this->gradientOut, *gradientIn, divOp);
        if (this->name == "mean{250}") LOG_SYNC(this->name, " Gradient out: ", this->gradientOut);
        this->prev_nodes[0]->backward(&this->gradientOut, ctx);
    }
    DividedBy<T> divOp;
    Matrix<T> gradientOut;

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Mean" +
               "</font>>]\n";
    }

    virtual std::string type() const override { return "Mean-" + std::to_string(Dim); }
};

template <typename T = FloatT, uint32 Dim = 0>
struct MeanExt : Node<T>
{
    MeanExt(NodePtr<T> prev, const std::string& name = "Average")
        : Node<T>(prev->shape.set(Dim, 1), {prev}, name, 1),
          gradientOut(prev->shape, name + "_gradientOut")
    {
        if (prev->shape[Dim] == 1)
            throw_rte_with_backtrace("Cannot reduce along dimension ", Dim, " for ", prev->name,
                                     prev->shape, " already 1");
    }

    void forward(Context*) override { reduce_mean_ext<T, Dim>(*this, this->prev(0)); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        unary_apply(this->gradientOut, *gradientIn, divOp);
        this->prev_nodes[0]->backward(&this->gradientOut, ctx);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Mean" +
               "</font>>]\n";
    }

    virtual std::string type() const override { return "Mean-" + std::to_string(Dim); }

    Matrix<T> gradientOut;
    DivByExtent<T, Dim> divOp;
};

template <typename T = FloatT, uint32 Dim = 0>
using Mean = MeanExt<T, Dim>;

// A proxy for an input node, used to pass input data to a node. But does not mark input
// node as "prev", so that when backward is called, it does not backpropagate through to input node.
// This is helpfull in mitigating a compounding number of back-prop path. E.g. if 2 SelfAttention
// are connected sequentially, viz. x->SA1->SA2, x is input to all the Linear nodes inside SA1,
// When back-prop'ing 9 gradients that come to x will be:
//  sa2q->sa1q->x, sa2q->sa1k->x, sa2v->sa1v->x
//  sa2k->sa1q->x, sa2k->sa1k->x, sa2k->sa1v->x
//  sa2v->sa1q->x, sa2v->sa1k->x, sa2v->sa1v->x
//  Instead if we use InputProxy and make the graph Proxy(x)->SA1->Proxy(SA1)->SA2, then only 3
//  gradients will be back-propagated to xp:
// 3 from SA2(q,k,v)->SA1(q,k,v)->Proxy(x). Now there will be 6 paths of length 2, instead of 9 of
// length 2 This effect becomes even more pronounced in MultiHeadAttention, where the number of
// paths get multiplied by the number of heads. this ::backward() only accumulates the gradient, so
// the owner of this node should call the ::proxy_backward() so that back-prop'ing happens to the
// actual input node
template <typename T>
struct InputProxy : Node<T>
{
    NodePtr<T> in;
    Matrix<T> gradientOut = Matrix<T>(this->shape, this->name + "_gradientOut");
    InputProxy(NodePtr<T> prev, const std::string& name)
        : Node<T>(prev->shape, {}, name + "_proxy", 0), in(prev)
    {
        gradientOut.set_val(T(0));
        this->set_data(in->get_data());
    }
    void forward(Context*) override { this->copy_extents(*in); }
    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        binary_apply(gradientOut, *gradientIn, Plus<T>());
    }
    void proxy_backward(Context* ctx)
    {
        LOG_NODE_TRACE("Proxy backward for ", this->name, " with gradientOut: ", gradientOut.name,
                       gradientOut.shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        in->backward(&gradientOut, ctx);
    }

    virtual std::string dot_repr() override
    {
        std::stringstream ret;
        ret << " [label=\"" << this->name
            << "\", shape=rect, style=filled, fillcolor=\"#b9cbd2\"]\n";
        ret << this->id << " -> " << in->id << " [style=dotted arrowhead=none]";
        return ret.str();
    }

    virtual std::string type() const override { return "InputProxy"; }
};

// Normalization, Dim=WIDTH_IDX woult be similar to layer norm,
// Dim=BATCH_IDX would be similar to  Batchorm with no momentum and no affine transform.
template <typename T = FloatT, uint32 Dim = WIDTH_IDX>
struct Normalize : public Node<T>
{
    NodePtr<T> in;
    InputProxy<T> x = InputProxy<T>(in, "nrm");
    Mean<T> mu = Mean<T>(&x, "nrm-mu");
    Power<T> mu_sq = Power<T>(&mu, 2, "nrm-mu^2");

    Power<T> sq = Power<T>(&x, 2, "nrm-x^2");
    Mean<T> sq_mu = Mean<T>(&sq, "nrm-x^2_mu");

    Subtract<T> var = Subtract<T>({&sq_mu, &mu_sq}, "nrm-var");
    Power<T> std = Power<T>(&var, 0.5, "nrm-std");
    Subtract<T> norm_num_sub = Subtract<T>(NodePtrVec<T>{&x, &mu}, "nrm-x-mu");
    Division<T> norm = Division<T>({&norm_num_sub, &std}, "nrm-Div");

    Normalize(NodePtr<T> prev, const std::string& name = "Normalize")
        : Node<T>(prev->shape, {prev}, name, 1), in(prev)
    {
        LOG_NODE_NAME(prev->shape, R_JUST("->", 4), this->shape);
        this->set_data(norm.get_data());
    }

    void forward(Context* ctx) override
    {
        x.in->compute(ctx);
        x.copy_extents(*x.in);
        norm.compute(ctx);
        this->copy_extents(norm);
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("", this->name, "gradientIn: ", gradientIn->name, gradientIn->shape);
        norm.backward(gradientIn, ctx);
        this->prev_nodes[0]->backward(&x.gradientOut, ctx);
    }

    virtual NodePtr<T> get_terminal_node() override { return &norm; }
    virtual std::string dot_repr() override
    {
        const char* dims[3] = {"Layer", "Seq", "Batch"};
        NodePtrVec<T> nodes = {&x, &mu, &mu_sq, &sq, &sq_mu, &var, &std, &norm_num_sub, &norm};
        std::stringstream ss;
        ss << " subgraph cluster_" << this->id << " {\n\tlabel = \"\n" << this->name << "\"\n\t";

        for (auto& n : nodes) ss << n->id << ' ';
        ss << "\n\t{rank=same;" << var.id << ' ' << std.id << ' ' << norm.id
           << "}\n }\n";  // end of cluster

        ss << this->id << " [label=\"" << dims[Dim] << "\n"
           << this->name << "\" style=filled fillcolor=gray shape=rect]\n";
        return ss.str();
    }

    virtual std::string type() const override { return "Normalize"; }
};

typedef Normalize<FloatT, WIDTH_IDX> LayerNorm;
typedef Normalize<FloatT, BATCH_IDX> BatchNorm;

template <typename T = FloatT>
struct Concat0 : Node<T>  // Concatenates many matrices along width, to produce a wider matrix
{
    std::vector<Matrix<T>> grads;
    std::vector<Matrix<T>*> prevs_as_mats;
    std::vector<Matrix<T>*> grad_ptrs;
    Concat0(NodePtrVec<T> prevs, const std::string& name)
        : Node<T>({prevs[0]->shape.set(WIDTH_IDX, prevs[0]->shape[WIDTH_IDX] * prevs.size())},
                  prevs, name, prevs.size())
    {
        grads.reserve(prevs.size());
        for (auto p : prevs)
        {
            if (p->height() != this->height())
                throw_rte_with_backtrace("Matrix dimensions do not match for Concat0 between ",
                                         p->name, p->shape, " and ", this->name, p->shape);
            grads.push_back(shaped_like(*p));
            prevs_as_mats.push_back((Matrix<T>*)p);
        }

        grad_ptrs.resize(prevs.size());

        for (uint32 i = 0; i < grads.size(); ++i) grad_ptrs[i] = &grads[i];
        LOG_NODE_NAME(prevs[0]->shape, " x ", this->prev_nodes.size(), R_JUST("->", 4),
                      this->shape);
    }

    void forward(Context*) override { concat(*this, prevs_as_mats); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        split(grad_ptrs, *gradientIn);
        for (uint32 i = 0; i < this->prev_nodes.size(); ++i)
            this->prev_nodes[i]->backward(&grads[i], ctx);
    }

    void print_desc()
    {
        LOG(BLUE, "Concatenating ", this->prev_nodes.size(), " inputs in ", this->name,
            " each of shape ", this->prev_nodes[0]->shape_str);
    }

    virtual std::string type() const override { return "Concat0"; }
};

template <typename T = FloatT>
struct Input : Node<T>
{
    Input(uint32 b, uint32_t num_samples, uint32_t row_vec_size, const std::string& name)
        : Node<T>({b, num_samples, row_vec_size}, {}, name, 0)
    {
    }
    Input(Shape shape, const std::string& name) : Node<T>(shape, {}, name, 0) {}
    void forward(Context*) override {}

    void backward(const Matrix<T>*, Context*) override
    {
        LOG_NODE_TRACE("Backward for ", this->name);
    }

    // TODO this should move to NodeBase
    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", shape=cylinder]";
    }

    virtual std::string type() const override { return "Input"; }
};

template <typename T>
struct Dropout : Node<T>
{
    Matrix<float32> mask;
    Matrix<T> gradientOut;
    const FloatT drop_probability;
    NodePtr<T> prev;
    Dropout(float32 p, NodePtr<T> prev, const std::string& name = "Dropout")
        : Node<T>(prev->shape, {prev}, name, 1),
          mask(prev->shape),
          gradientOut(this->shape),
          drop_probability(p),
          prev(prev)
    {
        if (p < 0 || p >= 1)
            throw_rte_with_backtrace("Dropout probability should be in the range [0, 1): ", p);
        LOG_NODE_NAME(this->shape, " Prob: ", p);
    }

    void forward(Context*) override
    {
        LOG_NODE_TRACE("Forward for ", this->name, " with probability: ", drop_probability);
        if (drop_probability > 0 && this->is_training)
            dropout(*this, *prev, mask, drop_probability);
        else
            this->copy(prev->begin());
    }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", prev->name, prev->shape);
        if (drop_probability > 0 && this->is_training)
        {
            dropout(gradientOut, *gradientIn, mask, -1);
            gradientOut.copy_extents(*gradientIn);
            prev->backward(&gradientOut, ctx);
        }
        else
        {
            prev->backward(gradientIn, ctx);
        }
    }

    std::string dot_repr() override
    {
        char buff[100];
        snprintf(buff, 100, " [label=\"%s\n%.2f\", style=filled, fillcolor=lightgray ]\n",
                 this->name.c_str(), drop_probability);
        return std::string(buff);
    }

    void debug_print()
    {
        LOG(BLUE, "Dropout with probability: ", drop_probability, " for ", this->name);
        if (drop_probability > 0) LOG(" mask: ", mask);
    }

    virtual std::string type() const override { return "Dropout"; }
};

template <typename T>
struct SinePositionalEmbedding : Node<T>
{
    Matrix<T> pos_emb;
    const float64 base = 1000;
    SinePositionalEmbedding(NodePtr<T> prev, const std::string& name = "SinePositionalEmbedding")
        : Node<T>(prev->shape, {prev}, name, 1), pos_emb(prev->shape.set(BATCH_IDX, 1))
    {
        LOG_NODE_NAME(this->shape);
        for (uint32 y = 0; y < this->height(); ++y)
        {
            for (uint32 x = 0; x < this->width(); ++x)
            {
                if (x % 2 == 0)
                {
                    FloatT div = std::pow(base, x / this->width());
                    pos_emb(y, x) = std::sin(y / div);
                }
                else
                {
                    FloatT div = std::pow(base, (x - 1) / this->width());
                    pos_emb(y, x) = std::cos(y / div);
                }
            }
        }
    }
    void forward(Context*) override { binary_apply(*this, pos_emb, this->prev(0), Plus<T>()); }

    void backward(const Matrix<T>* gradientIn, Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        this->prev_nodes[0]->backward(gradientIn, ctx);
    }

    virtual std::string type() const override { return "SinePosEmbn"; }
};

#endif  // NODES_UNPARAMETERIZED_HPP
