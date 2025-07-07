/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef LR_SCHED_HPP
#define LR_SCHED_HPP

#include <algorithm>
#include "context.hpp"
#include "string_utils.hpp"
#include "types"

class LRScheduler
{
 public:
    LRScheduler(float64 init_lr) : m_init_lr(init_lr), m_current_lr(init_lr) {}

    virtual float32 update_lr(Context* ctx)
    {
        (void)ctx;
        return m_current_lr;
    }

    virtual float32 get_lr() const { return m_init_lr; }

    virtual ~LRScheduler() = default;

 protected:
    const float64 m_init_lr;
    float32 m_current_lr;
};

class StepLRScheduler : public LRScheduler
{
 public:
    StepLRScheduler(float64 init_lr, float64 min_lr, uint32 step_size, float64 factor)
        : LRScheduler(init_lr), m_step_size(step_size), m_factor(factor), m_min_lr(min_lr)
    {
    }

    virtual float32 update_lr(Context* ctx) override
    {
        uint32 step = ctx->get_weight_update_count();
        if (step % m_step_size == 0 && step > 0)
        {
            auto lr = m_current_lr * m_factor;
            m_current_lr = std::max<float64>(lr, m_min_lr);
            m_last_update_step = step;
        }
        return m_current_lr;
    }

 private:
    const uint32 m_step_size;
    const float64 m_factor;

    const float64 m_min_lr;
    uint32 m_last_update_step = 0;
};

inline LRScheduler* create_lr_scheduler(const VarArgs& args, uint32 max_num_batches)
{
    (void)max_num_batches;
    auto lr_sched = args.get("lr_sched", std::string(""));

    if (lr_sched.empty() || lr_sched == "default") return new LRScheduler(args.get("lr", 0.01));

    if (lr_sched == "step")
        return new StepLRScheduler(args.get("lr", 0.001), args.get("min_lr", 1e-5),
                                   args.get("step_size", 1000), args.get("step_lr_factor", 0.95));
    throw_rte_with_backtrace("Unknown lr scheduler: ", lr_sched);
}
#endif  // LR_SCHED_HPP
