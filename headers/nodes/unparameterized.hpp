#ifndef NODES_UNPARAMETERIZED_HPP
#define NODES_UNPARAMETERIZED_HPP

/*
Various Nodes with no learnable parameters, these include softmaxes, matrix products, norms and
means.
*/

#include <cmath>
#include <sstream>
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
    void forward() override
    {
        transpose<FloatT, Exp<T>>(exp, this->prev(0), ExpOp);
        reduce(sumExps, exp);
        binary_apply(softmax, exp, sumExps, Div<T>());
        transpose(*this, softmax);
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        transpose(gradientInT, *gradientIn);
        softmax_gradient(gradientOut, softmax, gradientInT);
        this->prev_nodes[0]->backward(&gradientOut);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=LightSkyBlue, shape=rect] ";
    }
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
                                     " is 1 softmaxDim0 is invalid");
        }
        LOG(BLUE, R_JUST(this->name, 18), prev->shape, " reduced on WIDTH [", prev->width(), "]");
    }

    void forward() override
    {
        unary_apply(exp, this->prev(0), expOp);
        reduce(sumExps, exp);
        binary_apply(*this, exp, sumExps, Div<T>());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        softmax_gradient(gradientOut, *this, *gradientIn);
        transpose(gradientOutT, gradientOut);
        this->prev_nodes[0]->backward(&gradientOutT);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=LightSkyBlue, shape=rect] ";
    }
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

    void forward() override { mmadd(*this, this->prev(0), this->prev(1), {}, pProcess); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape);
        transpose(aT, this->prev(0), Neg<T>());
        mmTadd(a_grad_in, *gradientIn, this->prev(1), {}, pProcess);
        mmadd(b_grad_in, aT, *gradientIn, {}, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_in);
        this->prev_nodes[1]->backward(&b_grad_in);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=azure, shape=rect] ";
    }
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

    void forward() override { mmTadd(*this, this->prev(0), this->prev(1), {}, pProcess); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape, " and prev0: ", this->prev(0).name, this->prev(0).shape,
                       " and prev1: ", this->prev(1).name, this->prev(1).shape);
        mmadd(a_grad_inN, *gradientIn, this->prev(1), {}, pProcess);
        transpose(gradInT, *gradientIn, Neg<T>());
        mmadd(b_grad_in, gradInT, this->prev(0), {}, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_inN);
        this->prev_nodes[1]->backward(&b_grad_in);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", style=filled, fillcolor=azure, shape=rect] ";
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

    void forward() override { transpose(*this, this->prev(0)); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with  gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        transpose(gradientOut, *gradientIn);
        this->prev_nodes[0]->backward(&gradientOut);
    }
};

template <typename T = FloatT, uint32 Dim = 0>
struct Mean : Node<T>
{
    Mean(NodePtr<T> prev, const std::string& name = "Average")
        : Node<T>(prev->shape.set(Dim, 1), {prev}, name, 1), divOp(DividedBy<T>(prev->shape[Dim]))
    {
        if (prev->shape[Dim] == 1)
            throw_rte_with_backtrace("Cannot reduce along dimension ", Dim, " for ", prev->name,
                                     prev->shape, " already 1");
        LOG(BLUE, "Mean ", this->name, prev->shape, " -> ", this->shape);
    }

    void forward() override { reduce<T, Dim>(*this, this->prev(0), Plus<T>(), T(0), divOp); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        unary_apply(this->gradientOut, *gradientIn, divOp);
        this->prev_nodes[0]->backward(&this->gradientOut);
    }
    Matrix<T> gradientOut = Matrix<T>(this->shape, this->name + "_gradientOut");
    DividedBy<T> divOp;

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Mean" +
               "</font>>]\n";
    }
};

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
    void forward() override {}
    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        binary_apply(gradientOut, *gradientIn, Plus<T>());
    }
    void proxy_backward()
    {
        LOG_NODE_TRACE("Proxy backward for ", this->name, " with gradientOut: ", gradientOut.name,
                       gradientOut.shape);
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
};

// Normalization, Dim=WIDTH_IDX woult be similar to layer norm,
// Dim=BATCH_IDX would be similar to  Batchorm with no momentum and no affine transform.
template <typename T = FloatT, uint32 Dim = WIDTH_IDX>
struct Normalize : public Node<T>
{
    NodePtr<T> in;
    InputProxy<T> x = InputProxy<T>(in, "NormInput");
    Mean<T> mu = Mean<T>(&x, "mu");
    Power<T> mu_sq = Power<T>(&mu, 2, "mu^2");

    Power<T> sq = Power<T>(&x, 2, "x^2");
    Mean<T> sq_mu = Mean<T>(&sq, "x^2_mu");

    Subtract<T> var = Subtract<T>({&sq_mu, &mu_sq}, "var");
    Power<T> std = Power<T>(&var, 0.5, "std");
    Subtract<T> norm_num_sub = Subtract<T>(NodePtrVec<T>{&x, &mu}, "x-mu");
    Division<T> norm = Division<T>({&norm_num_sub, &std}, "Div");

    Normalize(NodePtr<T> prev, const std::string& name = "LayerNormSplit")
        : Node<T>(prev->shape, {prev}, name, 1), in(prev)
    {
        LOG(BLUE, this->name, "\t", prev->shape, " -> ", this->shape);
        this->set_data(norm.get_data());
    }

    void forward() override { norm.compute(); }

    void backward(const Matrix<T>* gradientIn) override
    {
        norm.backward(gradientIn);
        this->prev_nodes[0]->backward(&x.gradientOut);
        x.gradientOut.set_val(T(0));
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
};

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
    }

    void forward() override { concat(*this, prevs_as_mats); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        split(grad_ptrs, *gradientIn);
        for (uint32 i = 0; i < this->prev_nodes.size(); ++i)
            this->prev_nodes[i]->backward(&grads[i]);
    }

    void print_desc()
    {
        LOG(BLUE, "Concatenating ", this->prev_nodes.size(), " inputs in ", this->name,
            " each of shape ", this->prev_nodes[0]->shape_str);
    }
};

template <typename T = FloatT>
struct Input : Node<T>
{
    Input(uint32 b, uint32_t num_samples, uint32_t row_vec_size, const std::string& name)
        : Node<T>({b, num_samples, row_vec_size}, {}, name, 0)
    {
    }
    Input(Shape shape, const std::string& name) : Node<T>(shape, {}, name, 0) {}
    void forward() override {}

    void backward(const Matrix<T>*) override { LOG_NODE_TRACE("Backward for ", this->name); }

    // TODO this should move to NodeBase
    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", shape=cylinder]";
    }
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
        if (p < 0 or p >= 1)
            throw_rte_with_backtrace("Dropout probability should be in the range [0, 1): ", p);
    }

    void forward() override
    {
        if (drop_probability > 0 and this->is_training)
            dropout(*this, *prev, mask, drop_probability);
        else
            this->copy(prev->begin());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn: ", gradientIn->name,
                       gradientIn->shape);
        if (drop_probability > 0 and this->is_training)
        {
            dropout(gradientOut, *gradientIn, mask, -1);
            prev->backward(&gradientOut);
        }
        else
        {
            prev->backward(gradientIn);
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
};

#endif  // NODES_UNPARAMETERIZED_HPP
