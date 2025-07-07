/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include "trainer.hpp"
#include <filesystem>
#include <fstream>
#include "lr_sched.hpp"
#include "matrix.cuh"
#include "nodes/loss.hpp"
#include "string_utils.hpp"
#include "text_classification_dataset.hpp"
#include "utils.hpp"

void copy_params(std::map<Parameter<FloatT>*, Matrix<FloatT>*>& copy_map)
{
    for (auto p : get_params_of_type<FloatT, FloatT>())
    {
        if (copy_map.find(p) == copy_map.end())
            throw_rte_with_backtrace("Parameter ", p->name, " not found in copy_map");
        auto copy = copy_map[p];
        cudaMemcpy(copy->get().get(), p->begin(), p->shape.numels * sizeof(FloatT),
                   cudaMemcpyDefault);
    }
}

inline void handle_nan_loss(float32 loss, uint32 batch, const NetworkGraph& graph,
                            const Dataset* train_dataset)
{
    // if (!std::isnan(loss) && !std::isinf(loss)) return;
    if (batch <= 2) return;
    LOG(RED, "Training loss is nan at batch ", batch);
    auto indices = train_dataset->last_load_indices();
    for (uint32 i = 0; i < indices.size(); i++)
    {
        uint32 idx = indices[i];
        LOG(i, ": ", idx);
    }
    graph.print_node_values();
    for (auto p : get_params_of_type<FloatT, FloatT>())
    {
        // LOG(p->name, ":\n", p->grads());
    }
    train_dataset->print_last_batch();
    throw_rte_with_backtrace("Training loss is nan at batch ", batch);
}

void NetworkTrainer::train(const VarArgs& args)
{
    auto log_file = args.get<std::string>("log_file");
    if (log_file)
    {
        Log::Logger::get().tee(log_file.get());
        LOG(GREEN, "Logging to ", log_file.get());
    }

    FILE* metrics_csv = fopen((m_run_dir + "/training_metrics.csv").c_str(), "w");
    fprintf(metrics_csv, "batches,epoch,train_loss,train_acc,valdn_loss,valdn_acc,lr\n");

    std::ofstream grad_file;

    uint32_t train_batches = m_train_dataset.batches();
    if (train_batches == 0) throw_rte_with_backtrace("No training batches found");

    uint32 max_epochs = args.get("max_epochs", 100u);
    uint32 max_batches = args.get("max_batches", UINT32_MAX);

    uint32 max_batches_actual = std::min(train_batches * max_epochs, max_batches);
    if (!max_batches_actual) throw_rte_with_backtrace("Training for 0 batches?");

    LOG(GREEN, "Training for ", commas_int(max_batches_actual), " batches");

    this->set_max_batches(max_batches_actual);

    Timer timer("Training");

    uint32_t batch = 0;
    uint32_t epoch = 0;

    uint32_t log_interval_batch = args.get("log_interval_batch", UINT32_MAX);
    uint32_t file_log_interval_batch = args.get("file_log_interval_batch", UINT32_MAX);
    uint32_t save_interval_batches = args.get("save_interval_epochs", UINT32_MAX) * train_batches;
    uint32_t log_grads_interval_batches = args.get("log_grads_interval_batches", UINT32_MAX);

    float32 l1_lambda = args.get("l1_lambda", 0.0f);
    float32 l2_lambda = args.get("l2_lambda", 0.0f);
    float32 Wfactor = args.get("adamW_factor", 0.f);

    auto is_one_hot = args.get("one_hot_classes", 0) > 0;
    if (log_grads_interval_batches != UINT32_MAX) grad_file.open(m_run_dir + "/grads.txt");

    Context& ctx = Context::get();
    float32 train_loss_sum = 0;
    float32 train_loss_epoch_sum = 0;
    progress_bar(batch, max_batches_actual);
    bool call_end_training = false;

    while (batch < max_batches_actual && epoch < max_epochs)
    {
        epoch = batch / train_batches;

        progress_bar(batch % train_batches, train_batches);

        m_train_dataset.load(batch);

        ctx.forward_pass();
        m_loss_node->compute(&ctx);

        ctx.backward_pass();
        m_loss_node->backward(&ctx);

        auto train_loss = m_loss_node->value();
        train_loss_sum += train_loss;
        train_loss_epoch_sum += train_loss;

        batch++;

        handle_nan_loss(train_loss, batch, m_graph, &m_train_dataset);

        this->update(&ctx, l1_lambda, l2_lambda, Wfactor);

        if (batch % file_log_interval_batch == 0)
        {
            float32 train_loss_avg = train_loss_sum / file_log_interval_batch;
            fprintf(metrics_csv, "%d,%d,%f,nan,nan,nan,%f\n", batch, epoch, train_loss_avg,
                    m_current_lr);
            fflush(metrics_csv);
            train_loss_sum = 0;
        }

        if (batch % log_interval_batch == 0)
        {
            auto time_taken = timer.check();
            LOG(time_taken, ", Epoch: ", epoch, " Batch: ", batch, " - Train Loss: ", train_loss,
                " LR: ", m_current_lr);
        }

        if (batch % train_batches == 0)
        {
            this->validate(is_one_hot);
            float32 mean_train_loss = train_loss_epoch_sum / train_batches;
            train_loss_epoch_sum = 0;

            std::string acc_str;
            float32 train_acc = std::numeric_limits<float32>::quiet_NaN();
            if (is_one_hot)
            {
                train_acc = get_train_acc(m_val_dataset.batches());
                auto val_acc_str = std::to_string(m_validation_acc);
                acc_str += "\tAccuracy T/V: " GREEN + std::to_string(train_acc) + YELLOW + "\t" +
                           val_acc_str;

                if (m_validation_acc_prev - m_validation_acc > 25)
                {
                    acc_str += "\t" RED + std::to_string(m_validation_acc_prev) + " -> " +
                               val_acc_str + RESET;
                }
            }

            fprintf(metrics_csv, "%d,%d,%f,%f,%f,%f,%f\n", batch, epoch, mean_train_loss, train_acc,
                    m_validation_loss, m_validation_acc, m_current_lr);
            fflush(metrics_csv);

            LOG(R_JUST(epoch, 3), " | ", R_JUST(batch, 8), "\tLosses T/V: ", mean_train_loss, " | ",
                m_validation_loss, acc_str);
        }

        if (batch % save_interval_batches == 0)
        {
            std::string model_name =
                m_run_dir + "/model_epoch_" + std::to_string(epoch + 1) + ".ngw";
            LOG(MAGENTA, "Saving model at epoch ", epoch, " to ", model_name);
            m_graph.save_network(model_name);
        }

        if (batch % log_grads_interval_batches == 0)
        {
            LOG(MAGENTA, "Logging gradients at batch ", batch, " epoch ", epoch);
            grad_file << "Batch: " << batch << " Epoch: " << epoch << "\n";
            char line[256];  // right justifying the name
            snprintf(line, 256, "%-20s\t%10s\t%10s", "Parameter", "Magnitude", "Gradient");
            grad_file << line << "\n";
            for (auto p : get_params_of_type<FloatT, FloatT>())
            {
                snprintf(line, 256, "%-20s\t%10.6f\t%10.6f", p->name.c_str(), p->param_magnitude2(),
                         p->grad_magnitude2());
                grad_file << line << "\n";
            }
            grad_file << std::endl;
        }
    }
    grad_file.close();
    if (call_end_training) end_training();
}

uint32_t count_misses(const Matrix<FloatT>* softmax, const Matrix<FloatT>* target)
{
    cudaErrCheck(cudaDeviceSynchronize());
    uint32_t misses = 0;
    for (uint32_t b = 0; b < target->batch(); b++)
    {
        for (uint32_t y = 0; y < target->height(); y++)
        {
            uint64_t row_offset = softmax->shape.offset(b, y, 0);
            const auto* sm = softmax->begin() + row_offset;
            const auto* tg = target->begin() + row_offset;

            auto max_idx = std::max_element(sm, sm + softmax->width()) - sm;
            auto target_idx = std::max_element(tg, tg + target->width()) - tg;
            if (max_idx != target_idx) misses++;
        }
    }
    return misses;
}

float32 NetworkTrainer::get_train_acc(uint32_t num_batches)
{
    bool original_training_state = m_loss_node->is_training;
    m_graph.set_all_is_training(false);

    const auto* preds = m_loss_node->predictions;
    const auto* tgts = m_loss_node->target;
    Context& ctx = Context::get();
    float32 misses = 0;
    for (uint32_t i = 0; i < num_batches; i++)
    {
        m_train_dataset.load(i);
        ctx.forward_pass();
        m_loss_node->compute(&ctx);
        misses += count_misses(preds, tgts);
    }
    float32 total_samples = tgts->batch() * tgts->height() * num_batches;

    m_graph.set_all_is_training(original_training_state);
    return 100 * (1 - misses / total_samples);
}

float32 NetworkTrainer::update(Context* ctx, float32 l1_lambda, float32 l2_lambda, float32 Wfactor)
{
    ctx->weight_update();
    for (auto p : get_params_of_type<FloatT, FloatT>())
        p->update(m_current_lr, ctx, l1_lambda, l2_lambda, Wfactor);

    ctx->lr_update();
    m_current_lr = m_lr_scheduler->update_lr(ctx);
    return m_current_lr;
}

void NetworkTrainer::validate(bool is_one_hot)
{
    const auto* preds = m_loss_node->predictions;
    const auto* tgts = m_loss_node->target;

    m_graph.set_all_is_training(false);
    Context& ctx = Context::get();

    float32 val_batches = m_val_dataset.batches();
    float32 val_loss = 0;
    float32 misses = 0;
    float32 total_samples = 0;

    for (uint32_t j = 0; j < val_batches; j++)
    {
        m_val_dataset.load(j);
        ctx.forward_pass();
        m_loss_node->compute(&ctx);
        val_loss += m_loss_node->value();
        if (is_one_hot) misses += count_misses(preds, tgts);
        total_samples += tgts->batch() * tgts->height();
    }
    val_loss /= val_batches;
    if (is_one_hot)
    {
        m_validation_acc_prev = m_validation_acc;
        m_validation_acc = 100 * (1 - misses / total_samples);
    }

    m_graph.set_all_is_training(true);
    m_validation_loss = val_loss;
}

void NetworkTrainer::end_training()
{
    LOG(MAGENTA, "Ending training. Running validation and Saving model.");
    this->validate(false);

    auto& ctx = Context::get();
    std::ofstream grad_file(m_run_dir + "/grads_final.txt");
    grad_file << "Parameter,Magnitude,Gradient\n";

    for (auto p : get_params_of_type<FloatT, FloatT>())
    {
        grad_file << p->name << "," << p->param_magnitude2() << "," << p->grad_magnitude2()
                  << "\nGrads:\n"
                  << p->prev_grads() << "\nValues:\n"
                  << *(Matrix<FloatT>*)p << "\n";
    }

    for (auto p : get_params_of_type<FloatT, FloatT>())
    {
        print_param_histogram(grad_file, (Matrix<FloatT>*)p, 31);
    }

    if (auto lsmce = dynamic_cast<LogSoftmaxCELoss<FloatT>*>(m_loss_node))
    {
        lsmce->print(grad_file);
        lsmce->print(std::cout);
    }

    grad_file.close();
    std::string model_name = m_run_dir + "/" + ctx.session_id + "/model_epoch_final.ngw";
    std::string acc_str = m_validation_acc > 0 ? "\tAcc: " + std::to_string(m_validation_acc) : "";
    LOG(CYAN, "Final loss: ", m_validation_loss, acc_str, " Saving model to ", model_name);
    m_graph.save_network(model_name);
    LOG(CYAN, "Done!");
}
