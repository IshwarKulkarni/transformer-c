/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef NETWORK_TRAINER_HPP
#define NETWORK_TRAINER_HPP

#include "dataset.hpp"
#include "lr_sched.hpp"
#include "network_graph.hpp"
#include "nodes/loss.hpp"
#include "string_utils.hpp"

// Count misses for classification models
uint32 count_misses(const Matrix<FloatT>* softmax, const Matrix<FloatT>* target);

class NetworkTrainer
{
 public:
    NetworkTrainer(NetworkGraph& graph, Dataset& train_dataset, Dataset& val_dataset,
                   const VarArgs& args)
        : m_graph(graph),
          m_train_dataset(train_dataset),
          m_val_dataset(val_dataset),
          m_lr_scheduler(create_lr_scheduler(args, m_train_dataset.batches())),
          m_loss_node(dynamic_cast<Loss2Node<FloatT>*>(m_graph.get_root_node()))
    {
        m_run_dir = args.get("run_dir", std::string("runs"));
        LOG(MAGENTA, "Run directory: ", m_run_dir);
        if (m_loss_node == nullptr)
        {
            throw_rte_with_backtrace("No loss (of type Loss2Node<FloatT>) found in the network");
        }
        LOG("Number of matrices allocated: ", MatrixInitUitls::peek_id(),
            ", bytes allocated: ", convertMemorySting(MatrixInitUitls::get_alloced_bytes()));
    }

    void train(const VarArgs& args);

    void validate(bool is_one_hot);

    const NetworkGraph& graph() const { return m_graph; }

    float32 validation_loss() const { return m_validation_loss; }

    void set_max_batches(uint32_t max_batches) { m_max_batches = max_batches; }

    float32 update(Context* ctx, float32 l1_lambda, float32 l2_lambda, float32 Wfactor);

    void end_training();

    float32 get_train_acc(uint32_t num_batches);  // returns % of correct predictions

 private:
    uint32_t m_max_batches = 1000;
    NetworkGraph& m_graph;
    Dataset& m_train_dataset;
    Dataset& m_val_dataset;
    std::unique_ptr<LRScheduler> m_lr_scheduler;

    Loss2Node<FloatT>* m_loss_node;

    // Validation metrics
    float32 m_validation_loss = std::numeric_limits<float32>::quiet_NaN();
    float32 m_validation_acc =
        std::numeric_limits<float32>::quiet_NaN();  // percetage validation accuracy
    float32 m_validation_acc_prev = std::numeric_limits<float32>::quiet_NaN();

    float32 m_current_lr = 0.0f;
    std::string m_run_dir;  // directory of latest run
};

#endif
