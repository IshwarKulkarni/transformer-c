#ifndef ERROR_NODES_HPP
#define ERROR_NODES_HPP

#include "matrix.cuh"
#include "node_parameter.hpp"
#include "nodes.hpp"

///////////////////////////// Error Nodes ////////////////////////////////

template <typename T = FloatT>
struct Loss2Node : Node<T>  // 2 input loss node
{
    Loss2Node(const NodePtrs<T>& prevs, const std::string& name = "Loss")
        : Node<T>({1, 1, 1}, prevs, name, 2),
          predictions(prevs[0]),
          target(dynamic_cast<Input<FloatT>*>(prevs[1]))
    {
        if (this->prev(0).shape != this->prev(1).shape)
            throw_rte_with_backtrace(
                "LossNode inputs must have the same shape input 1: " + this->prev(0).shape.str() +
                " and input 2: " + this->prev(1).shape.str());

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
    }

    virtual void backward() = 0;

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", shape=diamond]";
    }

    FloatT value() const { return (*this)(0, 0); }
    Node<T>* predictions;
    Input<FloatT>* target;
};
/*
L2 loss computes (Y - Yt)^2 , first input is value, second is target
*/
template <typename T = FloatT>
struct L2Loss : Loss2Node<T>
{
    Matrix<T> diff;         // storage for (Y - Yt)
    Matrix<T> nDiff;        // storage for (Y - Yt)^2
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> times2ByNumels;

    L2Loss(const NodePtrs<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(inputs[0]->shape),
          nDiff(inputs[0]->shape),
          gradientOut(inputs[0]->shape, name + "_gradientOut"),
          times2ByNumels(FloatT(2.) / (diff.numels()))
    {
    }

    void forward() override
    {
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Square<T>());
        reduce_to_scalar<T>(nDiff, Plus<T>(), T(0), DividedBy<T>(nDiff.numels()));
        this->copy(nDiff.begin());
    }

    void backward() override
    {
        unary_apply(gradientOut, diff, times2ByNumels);
        this->predictions->backward(&gradientOut);
    }
};

template <typename T = FloatT>
struct L1Loss : Loss2Node<T>  // L1 loss computes (Y^ - Y)^2 , first input is target, second is Y
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^N
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> timesNByNumels;

    L1Loss(const NodePtrs<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(this->shape),
          nDiff(this->shape),
          gradientOut(this->shape, name + "_gradientOut"),
          timesNByNumels(FloatT(1.) / (diff.numels()))
    {
    }

    void forward() override
    {
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Abs<T>());
        reduce_to_scalar(nDiff, Plus<T>(), T(0), DividedBy<T>(nDiff.numels()));
        this->copy(nDiff.begin());
    }

    void backward() override
    {
        unary_apply(gradientOut, diff, Sign<T>{FloatT(1) / diff.numels()});
        this->predictions->backward(&gradientOut);
    }
};

template <typename T = FloatT>
struct NLLLoss : Loss2Node<T>  // first input is Y, second is target
{
    Matrix<T> tOverY = Matrix<T>(this->prev(0).shape);
    Matrix<T> gradientOut = Matrix<T>(this->prev(0).shape, this->name + "_gradientOut");

    Matrix<T> nll = Matrix<T>(this->prev(0).shape);

    NLLLoss(NodePtrs<T> prevs, const std::string& name = "NLLLoss") : Loss2Node<T>(prevs, name)
    {
        if (dynamic_cast<SoftmaxDim0<T>*>(prevs[0]) == nullptr and
            dynamic_cast<SoftmaxDim1<T>*>(prevs[0]) == nullptr)
        {
            throw_rte_with_backtrace("NLLLoss: first argument should be a Softmax Node");
        }
    }

    void forward() override
    {
        binary_apply(nll, this->prev(1), this->prev(0), NegLogLossFwd<T>());
        reduce_to_scalar(nll, Plus<T>(), T(0), DividedBy<T>(nll.numels()));
        this->copy(nll.begin());
    }

    void backward() override
    {
        NegLogLossBckwd<T> functor;
        functor.normalizing_factor = nll.numels();
        binary_apply(gradientOut, this->prev(1), this->prev(0), functor);
        this->predictions->backward(&gradientOut);
    }
};

// Apply log-softmax to incoming row-vectors and then apply cross entropy loss against target
// This is equivalent to torch.nn.CrossEntropy , but takes any probability distribution target
// (doesn't check that target rows are normal, doesn't take class indices as inputs.)
template <typename T = FloatT>
struct LogSoftmaxCELoss : Loss2Node<T>
{
    const Shape prevSize;
    Matrix<T> exps;             // e^xi
    Matrix<T> logSumExps;       // log(Sum(e^xj))
    Matrix<T> negLogSoftmax;    // [log(Sum(e^xj)) - xi]    (-ve log-softmax)
    Matrix<T> tgtNegLogSmProd;  //  [t * (xi - log(Sum(e^xj)))]   (multiply by t instead of -t,
                                //  (because -ve value above))
    Matrix<T> tgtLogSmProdSum;  // sum ( -t * (xi - log(Sum(e^xj))) ) . summed along width
    Matrix<T>
        tgtLogSmProdSum1D;  // sum(sum ( -t * (xi - log(Sum(e^xj))) ) ), now summer along height
    Matrix<T> gradient;
    Matrix<T> gradientOut;
    Matrix<T> softmax;

    LogSoftmaxCELoss(NodePtrs<T> prevs, const std::string& name = "CELoss")
        : Loss2Node<T>(prevs, name),
          prevSize(prevs[0]->shape),
          exps(prevSize, name + "_exps"),
          logSumExps(prevSize.set(WIDTH_IDX, 1), name + "_logSumExps"),
          negLogSoftmax(prevSize, name + "_negLogSoftmax"),
          tgtNegLogSmProd(prevSize, name + "_tgtNegLogSmProd"),
          tgtLogSmProdSum(prevSize.set(WIDTH_IDX, 1), name + "_tgtLogSmProd1d"),
          tgtLogSmProdSum1D(tgtLogSmProdSum.shape.set(HEIGHT_IDX, 1), name + "_tgtLogSmProd1d"),
          gradientOut(prevSize, name + "_gradientOut"),
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
        // tgtLogSmProd =  -[t * (log(Sum(e^xj)) - xi)]
        binary_apply(tgtNegLogSmProd, *this->target, negLogSoftmax, Mul<T>());
        // tgtLogSmProd1d = sum ( -t * (xi - log(Sum(e^xj))) ), computed for each instance
        reduce(tgtLogSmProdSum, tgtNegLogSmProd, Plus<T>(), T(0),
               DividedBy<T>(tgtNegLogSmProd.height()));

        // loss = mean ( -t * (xi - log(Sum(e^xj))) )
        reduce<T, HEIGHT_IDX>(tgtLogSmProdSum1D, tgtLogSmProdSum);
        if (tgtLogSmProdSum1D.batch() > 1)
            reduce<T, BATCH_IDX>(tgtLogSmProdSum1D, tgtLogSmProdSum1D, Plus<T>(), T(0),
                                 DividedBy<T>(tgtLogSmProdSum1D.batch()));
        this->copy(tgtLogSmProdSum1D.begin());
    }

    void backward() override
    {
        LSMCEBkwd<T> func;
        func.factor = gradientOut.height() * gradientOut.batch();
        binary_apply(gradientOut, *this->target, negLogSoftmax, func);
        this->predictions->backward(&gradientOut);
    }

    void debug_print()
    {
        LOG("\nDescription of ", this->name, '\n', exps, '\n', logSumExps, '\n', negLogSoftmax,
            '\n', tgtNegLogSmProd, '\n', tgtLogSmProdSum, '\n', gradient);
    }
};

#endif  // ERROR_NODES_HPP
