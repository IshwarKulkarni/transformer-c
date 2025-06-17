/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef NODES_LOSS_HPP
#define NODES_LOSS_HPP

/*
Nodes that implement various loss functions.
*/

#include "node.hpp"
#include "unparameterized.hpp"

template <typename T = FloatT>
struct Loss2Node : Node<T>  // 2 input loss node
{
    Loss2Node(const NodePtrVec<T>& prevs, const std::string& name = "Loss")
        : Node<T>({1, 1, 1}, prevs, name, 2),
          predictions(prevs[0]),
          target(dynamic_cast<Input<FloatT>*>(prevs[1]))
    {
        if (this->prev(0).shape != this->prev(1).shape)
            throw_rte_with_backtrace(
                "LossNode inputs must have the same shape input 0 (predictions): ",
                predictions->name, predictions->shape.str(), " &&input 1 (target): ", target->name,
                target->shape.str());

        if (target == nullptr)
        {
            throw_rte_with_backtrace("Loss2Node: second argument should be target");
        }
    }

    virtual void backward(const Matrix<T>* null, Context* ctx) override
    {
        (void)ctx;
        if (null)
            throw_rte_with_backtrace(
                "LossNode backward should not be called with a null argument || call the backward "
                "with no arguments");
    }

    virtual void backward(Context* ctx) = 0;

    virtual std::string dot_repr() override
    {
        return " [label=\"" + this->name + "\", shape=diamond]";
    }

    virtual std::string type() const override { return "Loss2Node"; }

    void reduce_and_copy(Matrix<T>& in)
    {
        if (in.batch() > 1) reduce<T, BATCH_IDX>(in, Plus<T>(), T(0), DividedBy<T>(in.batch()));
        if (in.height() > 1) reduce<T, HEIGHT_IDX>(in, Plus<T>(), T(0), DividedBy<T>(in.height()));
        if (in.width() > 1) reduce<T, WIDTH_IDX>(in, Plus<T>(), T(0), DividedBy<T>(in.width()));
        this->copy(in.begin());
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

    L2Loss(const NodePtrVec<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(inputs[0]->shape),
          nDiff(inputs[0]->shape),
          gradientOut(inputs[0]->shape, name + "_gradientOut"),
          times2ByNumels(FloatT(2.) / (diff.numels()))
    {
        LOG_NODE_NAME(this->prev(0).shape, R_JUST("->", 4), this->shape);
    }

    void forward(Context* ctx) override
    {
        (void)ctx;
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Pow<T>(2));
        this->reduce_and_copy(nDiff);
    }

    void backward(Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name);
        unary_apply(gradientOut, diff, times2ByNumels);
        this->predictions->backward(&gradientOut, ctx);
    }

    virtual std::string type() const override { return "L2Loss"; }
};

template <typename T = FloatT>
struct L1Loss : Loss2Node<T>  // L1 loss computes (Y^ - Y)^2 , first input is target, second is Y
{
    Matrix<T> diff;         // storage for (y - y_tartget)
    Matrix<T> nDiff;        // storage for (y - y_tartget)^N
    Matrix<T> gradientOut;  // storage for output  of reduction for backward
    MultiplyBy<T> timesNByNumels;

    L1Loss(const NodePtrVec<T>& inputs, const std::string& name = "L2Loss")
        : Loss2Node<T>(inputs, name),
          diff(inputs[0]->shape),
          nDiff(inputs[0]->shape),
          gradientOut(inputs[0]->shape, name + "_gradientOut"),
          timesNByNumels(FloatT(1.) / (diff.numels()))
    {
        LOG_NODE_NAME(this->prev(0).shape, R_JUST("->", 4), this->shape);
    }

    void forward(Context* ctx) override
    {
        (void)ctx;
        binary_apply(diff, this->prev(0), this->prev(1), Sub<T>());
        unary_apply(nDiff, diff, Abs<T>());
        this->reduce_and_copy(nDiff);
    }

    void backward(Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name);
        unary_apply(gradientOut, diff, Sign<T>{FloatT(1) / diff.numels()});
        this->predictions->backward(&gradientOut, ctx);
    }

    virtual std::string type() const override { return "L1Loss"; }
};

template <typename T = FloatT>
struct NLLLoss : Loss2Node<T>  // first input is Y, second is target
{
    Matrix<T> tOverY = Matrix<T>(this->prev(0).shape);
    Matrix<T> gradientOut = Matrix<T>(this->prev(0).shape, this->name + "_gradientOut");

    Matrix<T> nll = Matrix<T>(this->prev(0).shape);

    NLLLoss(NodePtrVec<T> prevs, const std::string& name = "NLLLoss") : Loss2Node<T>(prevs, name)
    {
        if (dynamic_cast<SoftmaxDim0<T>*>(prevs[0]) == nullptr and
            dynamic_cast<SoftmaxDim1<T>*>(prevs[0]) == nullptr)
        {
            throw_rte_with_backtrace("NLLLoss: first argument should be a Softmax Node");
        }
        LOG_NODE_NAME(this->prev(0).shape, R_JUST("->", 4), this->shape);
    }

    void forward(Context* ctx) override
    {
        (void)ctx;
        binary_apply(nll, this->prev(1), this->prev(0), NegLogLossFwd<T>());
        this->reduce_and_copy(nll);
    }

    void backward(Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name);
        NegLogLossBckwd<T> functor;
        functor.normalizing_factor = nll.numels();
        binary_apply(gradientOut, this->prev(1), this->prev(0), functor);
        this->predictions->backward(&gradientOut, ctx);
    }

    virtual std::string type() const override { return "NLLLoss"; }
};

// Apply log-softmax to incoming row-vectors &&then apply cross entropy loss against target
// This is equivalent to torch.nn.CrossEntropy (except this always applies the softmax in dim=-1,
// ie. WIDTH_IDX) but takes any probability distribution target &&doesn't check that target rows
// are normal, doesn't take class indices as inputs either.
template <typename T = FloatT>
struct LogSoftmaxCELoss : Loss2Node<T>
{
    const Shape prevSize;
    Matrix<T> exps;               // e^xi
    Matrix<T> logSumExps;         // log(Sum(e^xj))
    Matrix<T> negLogSoftmax;      // [log(Sum(e^xj)) - xi]    (-ve log-softmax)
    Matrix<T> tgtNegLogSmProd;    //  [t * (xi - log(Sum(e^xj)))]  (multiply by t instead of -t,
                                  //  (because -ve value above))
    Matrix<T> tgtLogSmProdSum;    // sum (-t * (xi - log(Sum(e^xj)))) . summed along width
    Matrix<T> tgtLogSmProdSum1D;  // sum(sum(-t * (xi - log(Sum(e^xj))))), now summer along height
    Matrix<T> gradient;
    Matrix<T> gradientOut;
    Matrix<T> softmax;

    LogSoftmaxCELoss(NodePtrVec<T> prevs, const std::string& name = "CELoss")
        : Loss2Node<T>(prevs, name),
          prevSize(prevs[0]->shape),
          exps(prevSize, name + "_exps"),
          logSumExps(prevSize.set(WIDTH_IDX, 1), name + "_logSumExps"),
          negLogSoftmax(prevSize, name + "_negLogSoftmax"),
          tgtNegLogSmProd(prevSize, name + "_tgtNegLogSmProd"),
          tgtLogSmProdSum(prevSize.set(WIDTH_IDX, 1), name + "_tgtLogSmProd"),
          tgtLogSmProdSum1D(tgtLogSmProdSum.shape.set(HEIGHT_IDX, 1), name + "_tgtLogSmProd1d"),
          gradientOut(prevSize, name + "_gradientOut"),
          softmax(prevSize, name + "_softmax")
    {
        LOG_NODE_NAME(this->prev(0).shape, R_JUST("->", 4), this->shape);
    }

    void forward(Context* ctx) override
    {
        (void)ctx;
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
               DividedBy<T>(tgtNegLogSmProd.width()));

        // loss = mean ( -t * (xi - log(Sum(e^xj))) )

        auto* temp = &tgtLogSmProdSum;

        if (tgtLogSmProdSum.height() > 1)
        {
            reduce<T, HEIGHT_IDX>(tgtLogSmProdSum1D, tgtLogSmProdSum);
            temp = &tgtLogSmProdSum1D;
        }

        if (temp->batch() > 1)
        {
            reduce<T, BATCH_IDX>(*this, *temp, Plus<T>(), T(0), DividedBy<T>(temp->batch()));
        }
        // this->copy(temp->begin());
    }

    void backward(Context* ctx) override
    {
        (void)ctx;
        LOG_NODE_TRACE("Backward for ", this->name);
        LSMCEBkwd<T> func;
        func.factor = gradientOut.height() * gradientOut.batch();
        binary_apply(gradientOut, *this->target, negLogSoftmax, func);
        this->predictions->backward(&gradientOut, ctx);
    }

    void debug_print()
    {
        LOG("\nDescription of ", this->name, "\ninput:\n", *this->predictions, '\n', exps, '\n',
            logSumExps, '\n', negLogSoftmax, '\n', tgtNegLogSmProd, '\n', tgtLogSmProdSum, '\n',
            tgtLogSmProdSum1D, '\n', gradientOut);
    }

    void print(std::ostream& out)
    {
        auto get_max_offset = [](const Matrix<FloatT>& m, uint32 b, uint32 h) {
            // get the offset of the max value of w elements at offset (b, h, 0)
            FloatT* ptr = m.get_data().get() + m.shape.offset(b, h, 0);
            FloatT max_val = ptr[0];
            uint32 max_offset = 0;
            for (uint32 w = 1; w < m.width(); w++)
            {
                if (ptr[w] > max_val)
                {
                    max_val = ptr[w];
                    max_offset = w;
                }
            }
            return max_offset;
        };

        auto in_flag = out.flags();
        out.setf(std::ios::fixed, std::ios::floatfield);
        out.precision(6);
        const Matrix<FloatT>& pred = *this->predictions;
        const Matrix<FloatT>& tgts = *this->target;
        const Matrix<FloatT>& loss = tgtNegLogSmProd;

        out << "Predictions:\t\t\tTarget:\t\t\tLoss:\n";
        uint32 num_matches = 0;
        for (uint32 b = 0; b < pred.batch(); b++)
        {
            for (uint32 h = 0; h < pred.height(); h++)
            {
                bool match = (get_max_offset(pred, b, h) == get_max_offset(tgts, b, h));
                num_matches += match;
                out << "(" << b << "," << h << "):\t";
                for (uint32 w = 0; w < pred.width(); w++) out << pred(b, h, w) << " ";
                out << "|\t";
                for (uint32 w = 0; w < tgts.width(); w++) out << tgts(b, h, w) << " ";
                out << "|\t";
                for (uint32 w = 0; w < loss.width(); w++) out << loss(b, h, w) << " ";
                out << (match ? "  ✓" : "  ✗") << "\n";
            }

            out << (pred.height() == 1 ? "" : "\n");
        }
        out << "Number of matches: " << num_matches << "\n";
        out.flags(in_flag);
    }

    virtual std::string type() const override { return "LogSoftmaxCELoss"; }
};

#endif  // NODES_LOSS_HPP
