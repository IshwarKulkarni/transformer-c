#ifndef ERROR_NODES_HPP
#define ERROR_NODES_HPP

#include "node_parameter.hpp"

///////////////////////////// Error Nodes ////////////////////////////////

template <typename T = FloatT>
struct Loss2Node : Node<T>  // 2 input loss node
{
    Loss2Node(const NodePtrs<T>& prevs, const std::string& name = "Loss")
        : Node<T>(1, 1, prevs, name, 2)
    {
        if (this->prev(0).height != this->prev(1).height ||
            this->prev(0).width != this->prev(1).width)
            throw_rte_with_backtrace(
                "LossNode inputs must have the same shape input 1: " + this->prev(0).shape_str +
                " and input 2: " + this->prev(1).shape_str);
    }

    virtual void backward(const Matrix<T>* null) override
    {
        if (null)
            throw_rte_with_backtrace(
                "LossNode backward should not be called with a null argument or call the backward "
                "with no arguments");
        this->backward();
    }

    virtual void backward() = 0;

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", shape=diamond]";
    }
};
/*
L2 loss computes (Y - Yt)^2 , first input is value, second is target
*/
template <typename T = FloatT>
struct L2Loss : Loss2Node<T>
{
    Matrix<T> diff;         // storage for (Y - Y)
    Matrix<T> nDiff;        // storage for (Y - Y)^2
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L2Loss(const NodePtrs<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(inputs[0]->shape()),
          nDiff(inputs[0]->shape()),
          temp1d(inputs[0]->height, 1),
          gradientOut(inputs[0]->shape()),
          times2ByNumels(FloatT(2.) / (diff.numels()))
    {
        if (inputs.size() != 2) throw_rte_with_backtrace("L2Loss requires 2 inputs");
    }

    void forward() override
    {
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        if (nDiff.width > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(*this, temp1d);
        }
        else
            reduce_mean(*this, nDiff);
    }

    void backward() override
    {
        // gradient is 2 * (y - target) / numels
        unary_apply(gradientOut, diff, times2ByNumels);
        for (auto& p : this->prev_nodes) p->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params() override
    {
        return diff.numels() + nDiff.numels() + temp1d.numels() + gradientOut.numels();
    }
};

template <typename T = FloatT>
struct L1Loss : Node<T>  // L1 loss computes (Y^ - Y)^2 , first input is target, second is Y
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^N
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> timesNByNumels;

    L1Loss(NodePtrs<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(this->shape()),
          nDiff(this->shape()),
          temp1d(this->height, 1),
          gradientOut(this->shape()),
          timesNByNumels(FloatT(1.) / (diff.numels()))
    {
    }

    void forward() override
    {
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Abs<T>());
        if (nDiff.width > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(*this, temp1d);
        }
        else
            reduce_mean(*this, nDiff);
    }

    void backward() override
    {
        unary_apply(gradientOut, diff, Sign<T>{FloatT(1) / diff.numels()});
        for (auto& p : this->prev_nodes) p->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params() override
    {
        return diff.numels() + nDiff.numels() + temp1d.numels() + gradientOut.numels();
    }
};

template <typename T = FloatT>
struct CrossEntropyLoss : Loss2Node<T>  // y is target, second is y
{
    Matrix<T> tOverY;
    Matrix<T> gradientOut;
    Matrix<T> ce;  // per element cross entropy
    Matrix<T> temp1d;

    CrossEntropyLoss(NodePtrs<T> prevs, const std::string& name)
        : Loss2Node<T>(prevs, name),
          tOverY(prevs[0]->shape()),
          gradientOut(prevs[0]->shape()),
          ce(prevs[0]->shape()),
          temp1d(prevs[0]->height, 1)
    {
    }

    void forward() override
    {
        binary_apply(ce, this->prev(0), this->prev(1), CrossEntropy<T>());
        if (ce.width > 1)
        {
            reduce_sum(temp1d, ce);
            reduce_sum(*this, temp1d);
        }
        else
            reduce_sum(*this, ce);
    }

    void backward() override
    {
        binary_apply(gradientOut, this->prev(0), this->prev(1), NegDiv<T>());
        for (auto& p : this->prev_nodes) p->backward(&gradientOut);
    }

    virtual uint32 n_untrainable_params() override
    {
        return tOverY.numels() + gradientOut.numels() + ce.numels() + temp1d.numels();
    }
};

#endif  // ERROR_NODES_HPP
