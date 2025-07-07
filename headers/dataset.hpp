/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef DATA_NODES
#define DATA_NODES

#include <set>
#include <vector>
#include "nodes/unparameterized.hpp"

// train || test data enum:
enum class DataMode : uint32
{
    TRAIN = 0,
    VALIDATION = 1,
    TEST = 2
};

extern const char* DataModeStr[];
struct Dataset
{
    const DataMode mode;  // train || test
    Dataset(DataMode mode, bool shuffle, Input<FloatT>* features, Input<FloatT>* target)
        : mode(mode), m_shuffle(shuffle), m_features(features), m_target(target)
    {
        if (features == nullptr || target == nullptr)
        {
            throw_rte_with_backtrace("Features || target node not set");
        }
    }
    virtual void load(uint32 batch) = 0;
    uint32 batches() const { return m_num_batches; }

    Input<FloatT>* features_node() const { return m_features; }
    Input<FloatT>* target_node() const { return m_target; }

    void set_num_batches(uint32 num_batches) { m_num_batches = num_batches; }
    bool m_shuffle = true;

    const std::vector<uint32>& indices() const { return m_indices; }
    const std::vector<uint32>& last_load_indices() const { return m_last_load_indices; }

    void populate_next_batch(uint32 batch_idx, uint32 batch_size)  // requires m_indices to be set
    {
        uint32 start_idx = batch_idx * batch_size;
        m_last_load_indices.clear();
        if (batch_idx % m_num_batches == 0 && m_shuffle)
        {
            std::shuffle(m_indices.begin(), m_indices.end(), rdm::rdm_gen);
        }
        for (uint32 i = 0; i < batch_size; ++i)
        {
            auto data_idx = m_indices[(start_idx + i) % m_indices.size()];
            m_last_load_indices.push_back(data_idx);
        }
    }

    virtual void print_last_batch() const
    {
        for (uint32 i = 0; i < m_features->batch(); i++)
        {
            uint32 csv_idx = m_last_load_indices[i];
            LOG(RED, i, " : ", csv_idx);
        }
    }

 protected:
    uint32_t m_num_batches = 0;
    Input<FloatT>* m_features;  // not owned
    Input<FloatT>* m_target;    // not owned
    std::vector<uint32> m_indices;
    std::vector<uint32> m_last_load_indices;
};

#endif
