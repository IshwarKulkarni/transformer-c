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
class NetworkTrainer
{
 public:
    NetworkTrainer(NetworkGraph& graph, Dataset& train_dataset, Dataset& val_dataset)
        : m_graph(graph),
          m_train_dataset(train_dataset),
          m_val_dataset(val_dataset),
          m_loss_node(m_graph.get_root_node())
    {
        LOG("Number of matrices allocated: ", MatrixInitUitls::peek_id(),
            ", bytes allocated: ", convertMemorySting(MatrixInitUitls::get_alloced_bytes()));
    }

    // train until max_epochs || max_batches is reached, 100 epochs by default
    void train(const VarArgs& args);

    void validate(Loss2Node<FloatT>* loss_node, bool is_one_hot);

    const NetworkGraph& graph() const { return m_graph; }

    float32 validation_loss() const { return m_validation_loss; }

    void set_max_batches(uint32_t max_batches) { m_max_batches = max_batches; }

    float32 update_weights(Context* ctx, LRScheduler* lr_scheduler);

    void end_training();

 private:
    uint32_t m_max_batches = 1000;
    NetworkGraph& m_graph;
    Dataset& m_train_dataset;
    Dataset& m_val_dataset;

    NodePtr<FloatT> m_loss_node;

    // Validation metrics
    float32 m_validation_loss = 0.0f;
    float32 m_validation_misses_frac = 0.0f;

    float32 m_current_lr = 0.0f;
    std::string m_run_dir;  // directory of latest run
};

#endif
