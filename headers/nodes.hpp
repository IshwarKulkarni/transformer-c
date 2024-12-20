#ifndef NODES_HPP
#define NODES_HPP

#include "matrix.cuh"
#include "node_parameter.hpp"
#include "types"

/*
Implements softmax along width of x => output column sum to 1 <br>
```
inp = this_prev_0_
assert inp.dim == 2
s = inp.exp() / inp.exp().sum(dim=0, keepdim=True)
```
*/
template <typename T = FloatT>
struct SoftmaxDim0 : Node<T>
{
    Matrix<T> exp;
    Matrix<T> sumExps;
    Matrix<T> softmax;
    Matrix<T> gradientOut;
    Matrix<T> gradientInT;
    Exp<T> ExpOp;

    SoftmaxDim0(NodePtr<T> prev, const std::string& name)
        : Node<T>(prev->shape(), {prev}, name, 1),
          exp(prev->t_shape()),
          sumExps(exp.height, 1),
          softmax(prev->t_shape()),
          gradientOut(prev->shape(), name + "_gradientOut"),
          gradientInT(prev->t_shape(), name + "_gradientOutT"),
          ExpOp(sqrt(this->height))
    {
    }

    // computes softmax along height of x => each output column sums to 1
    void forward() override
    {
        transpose<FloatT, Exp<T>>(exp, this->prev(0), ExpOp);
        reduce_sum(sumExps, exp);
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

// computes softmax along Width of x => each output row sums to 1
template <typename T = FloatT>
struct SoftmaxDim1 : Node<T>
{
    Matrix<T> exp;
    Matrix<T> sumExps;
    Matrix<T> gradientOut;
    Matrix<T> gradientOutT;
    Exp<T> expOp;

    SoftmaxDim1(NodePtr<T> prev, const std::string& name = "Softmax")
        : Node<T>(prev->shape(), {prev}, name, 1),
          exp(prev->shape()),
          sumExps(exp.height, 1),
          gradientOut(prev->t_shape(), name + "_gradientOut"),
          gradientOutT(prev->shape()),
          expOp()
    {
    }

    void forward() override
    {
        unary_apply(exp, this->prev(0), expOp);
        reduce_sum(sumExps, exp);
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
        : Node<T>(prevs[0]->height, prevs[1]->width, prevs, name, 2),
          aT(this->prev(0).t_shape()),
          a_grad_in(this->prev(0).shape()),
          b_grad_in(this->prev(1).shape()),
          pProcess(pProcess)
    {
        if (this->prev(0).width != this->prev(1).height)
            throw_rte_with_backtrace("Matrix dimensions do not match for product between ",
                                     this->prev(0).shape_str, " and ", this->prev(1).shape_str);
    }

    void forward() override
    {
        mmadd(*this, this->prev(0), this->prev(1), (Matrix<T>*)nullptr, pProcess);
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        transpose(aT, this->prev(0), Neg<T>());  // disable this in inference only mode.
        mmTadd(a_grad_in, *gradientIn, this->prev(1), (Matrix<T>*)nullptr, pProcess);
        mmadd(b_grad_in, aT, *gradientIn, (Matrix<T>*)nullptr, pProcessN);
        this->prev_nodes[0]->backward(&a_grad_in);
        this->prev_nodes[1]->backward(&b_grad_in);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\", style=filled, fillcolor=azure2, shape=parallelogram] ";
    }
};

/* Implements a matrix and transpose of another: output = A * B^T
 * Here's an equivalent python code:
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
        : Node<T>(prevs[0]->height, prevs[1]->height, prevs, name, 2),
          a_grad_inN(this->prev(0).shape()),
          b_grad_in(this->prev(1).shape()),
          gradInT(this->t_shape()),
          pProcess(pProcess)
    {
        if (this->prev(0).width != this->prev(1).width)
            throw_rte_with_backtrace("Matrix dimensions do not match for ProductT between ",
                                     this->prev(0).name, " and ", this->prev(1).name);
    }

    void forward() override
    {
        mmTadd(*this, this->prev(0), this->prev(1), (Matrix<T>*)nullptr, pProcess);
    }

    void backward(const Matrix<T>* gradientIn) override
    {
        mmadd(a_grad_inN, *gradientIn, this->prev(1), (Matrix<T>*)nullptr, pProcess);
        transpose(gradInT, *gradientIn, Neg<T>());
        mmadd(b_grad_in, gradInT, this->prev(0), (Matrix<T>*)nullptr, pProcessN);
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
    Matrix<T> gradientOut;

    Transpose(NodePtr<T> prev, const std::string& name)
        : Node<T>(prev->t_shape(), {prev}, name, 1),
          gradientOut(this->t_shape(), name + "_gradientOut")
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
struct Concat : Node<T>  // Concatenates many matrices along width, to produce a wider matrix
{
    std::vector<Matrix<T>> grads;
    std::vector<Matrix<T>*> prevs_as_mats;
    std::vector<Matrix<T>*> grad_ptrs;
    Concat(NodePtrs<T> prevs, const std::string& name)
        : Node<T>(prevs[0]->height, prevs[0]->width * prevs.size(), prevs, name, prevs.size())
    {
        grads.reserve(prevs.size());
        for (auto p : prevs)
        {
            if (p->height != this->height)
                throw_rte_with_backtrace("Matrix dimensions do not match for Concat between ",
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
    Input(uint32_t num_samples, uint32_t row_vec_size, const std::string& name)
        : Node<T>(num_samples, row_vec_size, {}, name, 0)
    {
    }
    Input(std::pair<uint32_t, uint32_t> shape, const std::string& name)
        : Node<T>(shape, {}, name, 0)
    {
    }
    void forward() override {}

    void backward(const Matrix<T>*) override {}

    void eye()
    {
        if (this->height != this->width)
            throw_rte_with_backtrace("Input matrix should be square for eye operation");
        fillCPU(*this, 0);
        for (uint32 i = 0; i < this->height; i++) (*this)(i, i) = 1;
    }

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
        : Node<T>(prev->shape(), {prev}, name, 1),
          mask(prev->shape()),
          gradientOut(this->shape()),
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