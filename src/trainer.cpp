/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include "trainer.hpp"
#include <filesystem>
#include <fstream>
#include "nodes/loss.hpp"
#include "nodes/parameterized.hpp"
#include "utils.hpp"

// Count misses for classification models
uint32_t count_misses(const Matrix<FloatT>* softmax, const Matrix<FloatT>* target)
{
    uint32_t misses = 0;
    for (uint32_t b = 0; b < target->batch(); b++)
    {
        for (uint32_t y = 0; y < target->height(); y++)
        {
            uint32_t row_offset = softmax->shape.offset(b, y, 0);
            const auto* sm = softmax->begin() + row_offset;
            const auto* tg = target->begin() + row_offset;

            auto max_idx = std::max_element(sm, sm + softmax->width()) - sm;
            auto target_idx = std::max_element(tg, tg + target->width()) - tg;
            if (max_idx != target_idx) misses++;
        }
    }
    return misses;
}

void NetworkTrainer::train(const VarArgs& args)
{
    auto log_file = args.get<std::string>("log_file");
    if (log_file)
    {
        Log::Logger::get().tee(log_file.get());
        LOG(GREEN, "Logging to ", log_file.get());
    }

    auto loss_node = dynamic_cast<Loss2Node<FloatT>*>(m_loss_node);
    if (loss_node == nullptr)
    {
        throw_rte_with_backtrace(m_loss_node->name, " is not a loss node");
    }

    Context& ctx = Context::get();
    m_run_dir = args.get("save_dir", std::string("runs"));

    std::string dir = m_run_dir + "/" + ctx.session_id;
    std::filesystem::create_directories(dir);
    LOG(GREEN, "Run directory: ", dir);

    // create a soft link to the run directory
    std::string latest_dir = m_run_dir + "/latest";
    if (std::filesystem::exists(latest_dir)) std::filesystem::remove(latest_dir);

    std::filesystem::create_symlink(ctx.session_id, latest_dir);

    std::ofstream train_csv(dir + "/train_losses.csv");
    train_csv << "batch,loss,misses,lr\n";

    std::ofstream valdn_csv(dir + "/valdn_losses.csv");
    valdn_csv << "batch,loss,misses,lr\n";

    std::ofstream grad_file(dir + "/grads.txt");

    uint32_t train_batches = m_train_dataset.batches();
    if (train_batches == 0) throw_rte_with_backtrace("No training batches found");

    uint32 max_epochs = args.get("max_epochs", 100u);
    uint32 max_batches = args.get("max_batches", UINT32_MAX);

    uint32 max_batches_actual = std::min(train_batches * max_epochs, max_batches);

    LOG(GREEN, "Training for ", max_batches_actual, " batches");

    this->set_max_batches(max_batches_actual);

    Timer timer("Training");

    uint32_t batch = 0;
    uint32_t epoch = 0;

    uint32_t log_interval_batch = args.get("log_interval_batch", 50u);
    uint32_t file_log_interval_batch = args.get("file_log_interval_batch", 20u);
    uint32_t save_interval_epochs = args.get("save_interval_epochs", 10u);
    uint32_t save_interval_batches = save_interval_epochs * train_batches;

    auto lr_scheduler = create_lr_scheduler(args);
    auto is_one_hot = args.get("one_hot_classes", 0) > 0;

    while (batch < max_batches_actual && epoch < max_epochs)
    {
        epoch = batch / train_batches;

        m_train_dataset.load(batch);

        ctx.forward_pass();
        loss_node->compute(&ctx);

        ctx.backward_pass();
        loss_node->backward(&ctx);

        batch++;

        auto train_loss = loss_node->value();

        auto lr = this->update_weights(&ctx, lr_scheduler);

        if (std::isnan(train_loss) || std::isinf(train_loss))
        {
            m_graph.print_node_values();
            throw_rte_with_backtrace("Training loss is nan at batch ", batch);
        }

        if (batch % file_log_interval_batch == 0)
            train_csv << batch << "," << train_loss << ",0," << lr << std::endl;

        if (batch % log_interval_batch == 0)
        {
            auto time_taken = timer.check();
            LOG(time_taken, ", Epoch: ", epoch, " Batch: ", batch, " - Train Loss: ", train_loss,
                " LR: ", lr);
        }

        if (batch % train_batches == 0)
        {
            this->validate(loss_node, is_one_hot);
            valdn_csv << batch << "," << m_validation_loss << "," << m_validation_misses_frac << ","
                      << lr << std::endl;

            std::string misses =
                is_one_hot ? "\tMisses: " + std::to_string(m_validation_misses_frac) : "";

            LOG("Epoch: ", epoch, " Batch: ", batch, " - Validation Loss: ", YELLOW,
                m_validation_loss, misses);
        }

        if (batch % save_interval_batches == 0)
        {
            LOG(MAGENTA, "Saving model at epoch ", epoch);
            m_graph.save_network(dir + "/model_epoch_" + std::to_string(epoch) + ".ngw");
        }

        if (false && batch % (m_train_dataset.batches() * 10) == 0 ||
            batch % (m_train_dataset.batches() * 10) == 1)  // print before and after every 10 epoch
        {
            LOG(MAGENTA, "Logging gradients at batch ", batch, " epoch ", epoch);
            grad_file << "Batch: " << batch << " Epoch: " << epoch << "\n";
            char line[256];  // right justifying the name
            snprintf(line, 256, "%-20s\t%10s\t%10s", "Parameter", "Magnitude", "Gradient");
            grad_file << line << "\n";
            for (auto param : Parameter<FloatT>::get_all_params())
            {
                auto p = dynamic_cast<Parameter<FloatT>*>(param);

                snprintf(line, 256, "%-20s\t%10.6f\t%10.6f", p->name.c_str(), p->param_magnitude2(),
                         p->grad_magnitude2());
                grad_file << line << "\n";
            }
            grad_file << std::endl;
        }
        if (log_file)
        {
            Log::Logger::get().flush();
        }
    }
    grad_file.close();
    end_training();
}

float32 NetworkTrainer::update_weights(Context* ctx, LRScheduler* lr_scheduler)
{
    auto lr = lr_scheduler->update_lr(ctx->weight_update());

    for (auto param : Parameter<FloatT>::get_all_params())
    {
        auto p = dynamic_cast<Parameter<FloatT>*>(param);
        p->update(lr, ctx);
    }
    m_current_lr = lr;
    return lr;
}

void NetworkTrainer::validate(Loss2Node<FloatT>* loss_node, bool is_one_hot)
{
    const auto* preds = loss_node->predictions;
    const auto* tgts = loss_node->target;

    loss_node->set_is_training(false);

    float32 val_loss = 0;
    uint32_t misses = 0;
    Context& ctx = Context::get();

    uint32_t val_batches = m_val_dataset.batches();
    uint32 total_samples = 0;

    for (uint32_t j = 0; j < val_batches; j++)
    {
        m_val_dataset.load(j);
        ctx.forward_pass();

        loss_node->compute(&ctx);
        val_loss += loss_node->value();
        if (is_one_hot) misses += count_misses(preds, tgts);
        total_samples += tgts->batch() * tgts->height();
    }
    val_loss /= val_batches;
    if (is_one_hot)
    {
        m_validation_misses_frac = misses / (float32)(total_samples);
    }

    loss_node->set_is_training(true);
    m_validation_loss = val_loss;
}

void NetworkTrainer::end_training()
{
    LOG(RED, "Ending training. Running validation and quitting.");
    this->validate(dynamic_cast<Loss2Node<FloatT>*>(m_loss_node), false);
    if (auto xel = dynamic_cast<LogSoftmaxCELoss<FloatT>*>(m_loss_node))
    {
        xel->print(std::cout);
    }

    auto& ctx = Context::get();
    std::ofstream grad_file(m_run_dir + "/" + ctx.session_id + "/grads_final.txt");
    grad_file << "Parameter,Magnitude,Gradient\n";

    for (auto param : Parameter<FloatT>::get_all_params())
    {
        auto p = dynamic_cast<Parameter<FloatT>*>(param);
        grad_file << p->name << "," << p->param_magnitude2() << "," << p->grad_magnitude2()
                  << "\nGrads:\n"
                  << p->prev_grads() << "\nValues:\n"
                  << *p << "\n";
    }
    grad_file.close();
    std::string model_name = m_run_dir + "/" + ctx.session_id + "/model_epoch_final.ngw";
    std::string misses =
        m_validation_misses_frac > 0 ? "\tMisses: " + std::to_string(m_validation_misses_frac) : "";
    LOG(CYAN, "Final loss: ", m_validation_loss, misses, " Saving model to ", model_name);
    m_graph.save_network(model_name);
    LOG(CYAN, "Done!");
}
