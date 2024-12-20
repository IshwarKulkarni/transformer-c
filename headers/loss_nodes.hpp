#ifndef ERROR_NODES_HPP
#define ERROR_NODES_HPP

#include "node_parameter.hpp"

///////////////////////////// Error Nodes ////////////////////////////////

template <typename T = FloatT>
struct Loss2Node : Node<T>  // 2 input loss node
{
    Loss2Node(const NodePtrs<T>& prevs, const std::string& name = "Loss")
        : Node<T>(1, 1, prevs, name, 2),
          target(dynamic_cast<Input<FloatT>*>(prevs[1])),
          predictions(prevs[0])
    {
        if (this->prev(0).height != this->prev(1).height ||
            this->prev(0).width != this->prev(1).width)
            throw_rte_with_backtrace(
                "LossNode inputs must have the same shape input 1: " + this->prev(0).shape_str +
                " and input 2: " + this->prev(1).shape_str);

        if (target == nullptr)
        {
            throw_rte_with_backtrace("Loss2Node: second argument should be target");
        }
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

    FloatT value() const { return (*this)(0, 0); }
    const Input<FloatT>* target;
    const Node<T>* predictions;
};
/*
L2 loss computes (Y - Yt)^2 , first input is value, second is target
*/
template <typename T = FloatT>
struct L2Loss : Loss2Node<T>
{
    Matrix<T> diff;         // storage for (Y - Yt)
    Matrix<T> nDiff;        // storage for (Y - Yt)^2
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L2Loss(const NodePtrs<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(inputs[0]->shape()),
          nDiff(inputs[0]->shape()),
          temp1d(inputs[0]->height, 1),
          gradientOut(inputs[0]->shape(), name + "_gradientOut"),
          times2ByNumels(FloatT(2.) / (diff.numels()))
    {
    }

    void forward() override
    {
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        if (nDiff.width > 1 && temp1d.numels() > 1)
        {
            reduce_mean(temp1d, nDiff);
            reduce_mean(*this, temp1d);
        }
        else
            reduce_mean(*this, nDiff);
    }

    void backward() override
    {
        unary_apply(gradientOut, diff, times2ByNumels);
        this->prev_nodes[0]->backward(&gradientOut);
    }
};

template <typename T = FloatT>
struct L1Loss : Loss2Node<T>  // L1 loss computes (Y^ - Y)^2 , first input is target, second is Y
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^N
    Matrix<T> temp1d;       // storage for output of reduction to 1d
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> timesNByNumels;

    L1Loss(const NodePtrs<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(this->shape()),
          nDiff(this->shape()),
          temp1d(this->height, 1),
          gradientOut(this->shape(), name + "_gradientOut"),
          timesNByNumels(FloatT(1.) / (diff.numels()))
    {
    }

    void forward() override
    {
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Abs<T>());
        if (nDiff.width > 1 && temp1d.numels() > 1)
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
};

template <typename T = FloatT>
struct NLLLoss : Loss2Node<T>  // first input is Y, second is target
{
    Matrix<T> tOverY;
    Matrix<T> gradientOut;
    Matrix<T> nll;  // per element -t*log(y)
    Matrix<T> temp1d;

    NLLLoss(NodePtrs<T> prevs, const std::string& name = "NLLLoss")
        : Loss2Node<T>(prevs, name),
          tOverY(prevs[0]->shape()),
          gradientOut(prevs[0]->shape(), name + "_gradientOut"),
          nll(prevs[0]->shape()),
          temp1d(prevs[0]->height, 1)
    {
        if (dynamic_cast<SoftmaxDim1<T>*>(prevs[0]) == nullptr)
        {
            throw_rte_with_backtrace("NLLLoss: first argument should be softmax");
        }
    }

    void forward() override
    {
        binary_apply(nll, this->prev(1), this->prev(0), NegLogLossFwd<T>());
        if (nll.width > 1 && temp1d.numels() > 1)  // we need the reduction.
        {
            reduce_sum(temp1d, nll);
            reduce_sum(*this, temp1d);
        }
        else
            reduce_sum(*this, nll);
    }

    void backward() override
    {
        binary_apply(gradientOut, this->prev(1), this->prev(0), NegLogLossBckwd<T>());
        for (auto& p : this->prev_nodes) p->backward(&gradientOut);
    }
};

// Apply log-softmax to incoming row-vectors and then apply cross entropy loss against target
// This is equivalent to torch.nn.CrossEntropy , but takes any probability distribution target
// (doesn't check that target rows are normal, doesn't take class indices as inputs.)
template <typename T = FloatT>
struct LogSoftmaxCELoss : Loss2Node<T>
{
    const std::pair<uint32, uint32> prevSize;
    Matrix<T> exps;             // e^xi
    Matrix<T> logSumExps;       // log(Sum(e^xj))
    Matrix<T> negLogSoftmax;    // [log(Sum(e^xj)) - xi]    (-ve log-softmax)
    Matrix<T> tgtNegLogSmProd;  //  [t * (xi - log(Sum(e^xj)))]   (multiply by t instead of -t,
                                //  (because -ve value above))
    Matrix<T> tgtLogSmProdSum;  // sum ( -t * (xi - log(Sum(e^xj))) ) in 1D
    Matrix<T> gradientOut;
    Matrix<T> softmax;

    LogSoftmaxCELoss(NodePtrs<T> prevs, const std::string& name = "CELoss")
        : Loss2Node<T>(prevs, name),
          prevSize(prevs[0]->shape()),
          exps(prevSize, name + "_exps"),
          logSumExps(prevSize.first, 1, name + "_logSumExps"),
          negLogSoftmax(prevSize, name + "_negLogSoftmax"),
          tgtNegLogSmProd(prevSize, name + "_tgtNegLogSmProd"),
          tgtLogSmProdSum(prevSize.first, 1, name + "_tgtLogSmProd1d"),
          gradientOut(prevSize, name + "_gradOut"),
          softmax(prevSize, name + "_softmax")
    {
    }

    void forward() override
    {
        // exps = e^xi
        unary_apply(exps, *this->predictions, Exp<T>());
        // logSumExps = log(Sum(e^xj))
        reduce(logSumExps, exps, Plus<T>(), T(0), Loge<T>());
        // neglogSoftmax = [log(Sum(e^xj)) - xi]
        binary_apply(negLogSoftmax, logSumExps, *this->predictions, Sub<T>());
        // tgtLogSmProd =  [t * (xi - log(Sum(e^xj)))]
        binary_apply(tgtNegLogSmProd, *this->target, negLogSoftmax, Mul<T>());
        if (tgtLogSmProdSum.numels() > 1)
        {
            // tgtLogSmProd1d = sum ( -t * (xi - log(Sum(e^xj))) ), computed for each instance
            reduce_sum(tgtLogSmProdSum, tgtNegLogSmProd);
            // loss = mean ( -t * (xi - log(Sum(e^xj))) )
            reduce_mean(*this, tgtLogSmProdSum);
        }
        else
            reduce_sum(*this, tgtNegLogSmProd);
        // TODO(remove) : for debug:
        unary_apply(softmax, negLogSoftmax, NLSToSoftmax<T>());
    }

    void backward() override
    {
        binary_apply(gradientOut, *this->target, negLogSoftmax, LSMCEBkwd<T>());
        this->prev_nodes[0]->backward(&gradientOut);
    }

    void debug_print()
    {
        LOG("\nDescription of ", this->name, '\n', exps, '\n', logSumExps, '\n', negLogSoftmax,
            '\n', tgtNegLogSmProd, '\n', tgtLogSmProdSum, '\n', gradientOut);
    }
};

#endif  // ERROR_NODES_HPP
