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

static const char* DataModeStr[] = {"Train", "Validation", "Test"};

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

    void set_num_batches(uint32 num_batches)
    {
        LOG("Num batches for ", DataModeStr[uint32(mode)], " set to ", GREEN, num_batches);
        m_num_batches = num_batches;
    }
    bool m_shuffle = true;

 protected:
    uint32_t m_num_batches = 0;
    Input<FloatT>* m_features;  // not owned
    Input<FloatT>* m_target;    // not owned
};

#endif
