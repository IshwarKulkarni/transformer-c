#ifndef NODES_HPP
#define NODES_HPP

#include <cmath>
#include "matrix.cuh"
#include "node_parameter.hpp"
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

    SoftmaxDim1(NodePtr<T> prev, const std::string& name = "SoftmaxDim1")
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
        transpose(gradientInT, *gradientIn);
        softmax_gradient(gradientOut, softmax, gradientInT);
        this->prev_nodes[0]->backward(&gradientOut);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\", style=filled, fillcolor=LightSkyBlue, shape=octagon] ";
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

    SoftmaxDim0(NodePtr<T> prev, const std::string& name = "SoftmaxDim0")
        : Node<T>(prev->shape, {prev}, name, 1)
    {
        if (prev->width() == 1)
        {
            throw_rte_with_backtrace("Dim[0] of previous node, ", prev->name, prev->shape,
                                     " is 1 softmaxDim0 is invalid");
        }
        LOG(BLUE, this->name, "\t", prev->shape, " reduced along WIDTH [", prev->width(), "]");
    }

    void forward() override
    {
        unary_apply(exp, this->prev(0), expOp);
        reduce(sumExps, exp);
        binary_apply(*this, exp, sumExps, Div<T>());
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        softmax_gradient(gradientOut, *this, *gradientIn);
        transpose(gradientOutT, gradientOut);
        this->prev_nodes[0]->backward(&gradientOutT);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\", style=filled, fillcolor=LightSkyBlue, shape=octagon] ";
    }
};

template <typename T, typename PostProcess>
struct Product : Node<T>
{
    Matrix<T> aT, a_grad_in, b_grad_in;
    PostProcess pProcess;
    Composition<T, Neg<T>, PostProcess> pProcessN = {Neg<T>(), pProcess};

    Product(NodePtrs<T> prevs, PostProcess pProcess, const std::string& name)
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
        transpose(aT, this->prev(0), Neg<T>());  // disable this in inference only mode.
        mmTadd(a_grad_in, *gradientIn, this->prev(1), {}, pProcess);
        mmadd(b_grad_in, aT, *gradientIn, {}, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_in);
        this->prev_nodes[1]->backward(&b_grad_in);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\", style=filled, fillcolor=azure2, shape=parallelogram] ";
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

    ProductT(NodePtrs<T> prevs, PostProcess pProcess, const std::string& name)
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
        mmadd(a_grad_inN, *gradientIn, this->prev(1), {}, pProcess);
        transpose(gradInT, *gradientIn, Neg<T>());
        mmadd(b_grad_in, gradInT, this->prev(0), {}, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_inN);
        this->prev_nodes[1]->backward(&b_grad_in);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\", style=filled, fillcolor=azure, shape=parallelogram] ";
    }
};

template <typename T = FloatT>
struct Transpose : Node<T>
{
    Matrix<T> gradientOut = Matrix<T>(this->shape.t(), this->name + "_gradientOut");

    Transpose(NodePtr<T> prev, const std::string& name) : Node<T>(prev->shape.t(), {prev}, name, 1)
    {
        if (!(prev->height == this->width && prev->width == this->height))
            throw_rte_with_backtrace("Matrix dimensions do not match for Transpose between ",
                                     prev->name, " and ", this->name);
    }

    void forward() override { transpose(*this, this->prev(0)); }

    void backward(const Matrix<T>* gradientIn) override
    {
        transpose(gradientOut, *gradientIn);
        this->prev_nodes[0]->backward(&gradientOut);
    }
};

template <typename T = FloatT>
struct Concat0 : Node<T>  // Concatenates many matrices along width, to produce a wider matrix
{
    std::vector<Matrix<T>> grads;
    std::vector<Matrix<T>*> prevs_as_mats;
    std::vector<Matrix<T>*> grad_ptrs;
    Concat0(NodePtrs<T> prevs, const std::string& name)
        : Node<T>({prevs[0]->batch(), prevs[0]->height, prevs[0]->width * prevs.size()}, prevs,
                  name, prevs.size())
    {
        grads.reserve(prevs.size());
        for (auto p : prevs)
        {
            if (p->height != this->height)
                throw_rte_with_backtrace("Matrix dimensions do not match for Concat0 between ",
                                         p->name, p->shape_str, " and ", this->name, p->shape_str);
            grads.push_back(shaped_like(*p));
            prevs_as_mats.push_back((Matrix<T>*)p);
        }
        grad_ptrs.resize(prevs.size());
        for (uint32 i = 0; i < grads.size(); ++i) grad_ptrs[i] = &grads[i];
    }

    void forward() override { concat(*this, prevs_as_mats); }

    void backward(const Matrix<T>* gradientIn) override
    {
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

    void backward(const Matrix<T>*) override {}

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
    Dropout(float32 p, NodePtr<T> prev, const std::string& name = "Dropout")
        : Node<T>(prev->shape, {prev}, name, 1),
          mask(prev->shape),
          gradientOut(this->shape),
          drop_probability(p)
    {
        if (p < 0 or p >= 1)
            throw_rte_with_backtrace("Dropout probability should be in the range [0, 1): ", p);
    }

    void forward() override
    {
        if (drop_probability > 0 and this->is_training)
            dropout(*this, *this->prev_nodes[0], mask, drop_probability);
        else
            fill(*this, *this->prev_nodes[0]);
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        if (drop_probability > 0 and this->is_training)
        {
            dropout(gradientOut, *gradientIn, mask, -1);
            this->prev_nodes[0]->backward(&gradientOut);
        }
        else
        {
            this->prev_nodes[0]->backward(gradientIn);
        }
    }

    std::string dot_repr() override
    {
        char buff[100];
        snprintf(buff, 100, " [label=\"%s\n%.2f\", style=filled, fillcolor=azure ]\n",
                 this->name.c_str(), drop_probability);
        return std::string(buff);
    }

    void debug_print()
    {
        LOG(BLUE, "Dropout with probability: ", drop_probability, " for ", this->name);
        if (drop_probability > 0) LOG(" mask: ", mask);
    }
};

#endif  // NODES_HPP