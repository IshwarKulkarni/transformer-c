#ifndef NODES_ELEM_HPP
#define NODES_ELEM_HPP

/*
Nodes that perform per element operations like arithmetic operations
*/

#include "matrix.cuh"
#include "node.hpp"
#include "types"

// Element wise sum
template <typename T = FloatT>
struct Sum : Node<T>
{
    Sum(NodePtrVec<T> prevs, const std::string& name = "Sum")
        : Node<T>(prevs[0]->shape, prevs, name, 2)
    {
        if (prevs[0]->shape != prevs[1]->shape)
            throw_rte_with_backtrace("Matrix dimensions do not match for sum between ",
                                     prevs[0]->name, prevs[0]->shape, " and ", prevs[1]->name,
                                     prevs[1]->shape);
        LOG(BLUE, this->name, "\t", prevs[0]->shape, " -> ", this->shape);
    }

    void forward() override { binary_apply(*this, this->prev(0), this->prev(1), Plus<T>()); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn shape: ", gradientIn->shape);
        this->prev_nodes[0]->backward(gradientIn);
        this->prev_nodes[1]->backward(gradientIn);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Sum" +
               "</font>>]\n";
    }
};

// Element wise subtraction, prevs[0](=A) - prevs[1](=B), output shape is same as A, B is
// broadcasted to A
template <typename T = FloatT>
struct Subtract : Node<T>
{
    NodePtr<T> A, B;
    Matrix<T> gradientB = Matrix<T>(B->shape, this->name + "_gradientOutB");
    Subtract(NodePtrVec<T> prevs, const std::string& name = "Subtract")
        : Node<T>(prevs[0]->shape, prevs, name, 2), A(prevs[0]), B(prevs[1])

    {
        if (!broadcastable<0>(B->shape, A->shape) or !broadcastable<1>(B->shape, A->shape) or
            !broadcastable<2>(B->shape, A->shape))
            throw_rte_with_backtrace("Matrix dimensions do not match for Subtract between ",
                                     A->name, A->shape, " and ", B->name, B->shape, " for ",
                                     this->name);
        uint32 mimatches = 0;
        for (uint32 i = 0; i < 3; i++) mimatches += (A->shape[i] != B->shape[i]);
        if (mimatches >= 2)
            throw_rte_with_backtrace(B->name, B->shape, " broadcasts to ", A->name, A->shape,
                                     "in more than 1 dimension in ", this->name);
    }

    void forward() override { binary_apply(*this, this->prev(0), this->prev(1), Sub<T>()); }

    void backward(const Matrix<T>* gradIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn shape: ", gradIn->shape);
        A->backward(gradIn);

        // We could have broadcasted B to A, so we need to sum the gradient in broadcasted
        // dimensions, and negate the result
        if (gradientB.shape[0] != gradIn->shape[0])
            reduce<T, 0>(gradientB, *gradIn, Plus<T>(), T(0), Neg<T>());
        else if (gradientB.shape[1] != gradIn->shape[1])
            reduce<T, 1>(gradientB, *gradIn, Plus<T>(), T(0), Neg<T>());
        else if (gradientB.shape[2] != gradIn->shape[2])
            reduce<T, 2>(gradientB, *gradIn, Plus<T>(), T(0), Neg<T>());
        else
            unary_apply(gradientB, *gradIn, Neg<T>());  // no broadcast
        B->backward(&gradientB);
    }
    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Subtract" +
               "</font>>]\n";
    }
};

// Element wise division, prevs[0] / prevs[1], output shape is same as prevs[0], prevs[1] is
// broadcasted to prevs[0]
template <typename T>
struct Division : Node<T>
{
    NodePtr<T> num, denom;
    Matrix<T> gradientOut = Matrix<T>(this->shape, this->name + "_gradientOut");  //
    Matrix<T> gradientOutDenom = Matrix<T>(denom->shape, this->name + "_gradientOutDenom");

    Division(NodePtrVec<T> prevs, const std::string& name = "Division")
        : Node<T>(prevs[0]->shape, prevs, name, 2), num(prevs[0]), denom(prevs[1])
    {
        if (!broadcastable<0>(denom->shape, num->shape) or
            !broadcastable<1>(denom->shape, num->shape) or
            !broadcastable<2>(denom->shape, num->shape))
            throw_rte_with_backtrace("In Division node ", this->name, ", ", denom->name,
                                     denom->shape, " cannot be broadcasted to ", num->name,
                                     num->shape);

        uint32 mimatches = 0;
        for (uint32 i = 0; i < 3; i++) mimatches += (num->shape[i] != denom->shape[i]);
        if (mimatches >= 2)
            throw_rte_with_backtrace(denom->name, denom->shape, " broadcasts to ", denom->name,
                                     denom->shape, "in more than 1 dimension in ", this->name);
    }

    void forward() override { binary_apply(*this, *num, *denom, Div<T>()); }

    void backward(const Matrix<T>* gradIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn shape: ", gradIn->shape);
        binary_apply(gradientOut, *gradIn, this->prev(1), Div<T>());
        num->backward(&gradientOut);

        // reusing gradientOut
        ternary_apply(gradientOut, *gradIn, this->prev(0), this->prev(1), DivDiff<T>());

        if (gradientOutDenom.shape[0] != gradIn->shape[0])
            reduce(gradientOutDenom, gradientOut);
        else if (gradientOutDenom.shape[1] != gradIn->shape[1])
            reduce(gradientOutDenom, gradientOut);
        else if (gradientOutDenom.shape[2] != gradIn->shape[2])
            reduce(gradientOutDenom, gradientOut);

        denom->backward(&gradientOutDenom);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Division" +
               "</font>>]\n";
    }
};

template <typename T = FloatT>
struct Power : Node<T>
{
    Matrix<T> gradientOut = Matrix<T>(this->shape, this->name + "_gradientOut");
    const FloatT power;
    Power(NodePtr<T> prev, FloatT power, const std::string& name = "Power")
        : Node<T>(prev->shape, {prev}, name, 1), power(power)
    {
    }

    void forward() override { unary_apply(*this, this->prev(0), Pow<T>(power)); }

    void backward(const Matrix<T>* gradientIn) override
    {
        LOG_NODE_TRACE("Backward for ", this->name, " with gradientIn shape: ", gradientIn->shape);
        binary_apply(gradientOut, this->prev(0), *gradientIn, PowDiff<T>(power));
        this->prev_nodes[0]->backward(&gradientOut);
    }

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name +
               "\" shape=rect  xlabel=<<font color=\"green\" POINT-SIZE=\"10.0\">" + "Power" +
               "</font>>]\n";
    }
};
#endif  // NODES_HPP
