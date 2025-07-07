/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <filesystem>
#include <initializer_list>
#include "csv_dataset.hpp"
#include "logger.hpp"
#include "network_graph.hpp"
#include "signal.h"
#include "string_utils.hpp"
#include "text_classification_dataset.hpp"
#include "trainer.hpp"

void test_empty_kernel();

// Forward declarations for new help functions
void print_creator_help(const std::string& creator_name, std::ostream& os = std::cout);
void print_all_creator_help(std::ostream& os = std::cout);

NetworkTrainer* trainer_ptr = nullptr;

inline void sigaction_handler(int32 signum)
{
    LOG(RED, "Interrupt received: ", signum);
    trainer_ptr->end_training();
    exit(0);
}

void train_network(VarArgs& args)
{
    // Get info to create the graph
    uint32 batch = args.get<uint32>("batch", 32);
    NetworkGraph graph;
    graph.define_variable("$batch", batch);

    std::string graph_file = args.get("graph_file", "");
    std::ifstream net_desc(graph_file);

    std::string dot_file = args.get("dot_file", "");
    dot_file = args.get("run_dir", std::string(".")) + "/" + dot_file;
    LOG("Graph File: ", GREEN, graph_file, RESET, " with Batch Size: ", GREEN, batch,
        " writing DOT: ", GREEN, dot_file, RESET);

    // actually read the graph from the file, and display it
    graph.load_from_desc_stream(net_desc);
    graph.print_nodes();
    graph.write_dotviz(dot_file);

    // Get info to create the datasets
    auto data = graph.get_typed_node<Input<FloatT>>("input");
    auto target = graph.get_typed_node<Input<FloatT>>("target");

    std::string train_csv = args.get("train_dataset_file", "");
    std::string val_csv = args.get("valdn_dataset_file", "");

    if (train_csv.empty() || val_csv.empty())
        throw_rte_with_backtrace("train_dataset_file and valdn_dataset_file must be provided");

    std::unique_ptr<Dataset> train, val;
    std::string task = args.get("task", "classification");

    std::unique_ptr<Word2VecBase> word2vec;

    // Actually create the datasets
    if (task == "classification")
    {
        train = std::make_unique<CSVDataset>(train_csv, DataMode::TRAIN, data, target, args);
        val = std::make_unique<CSVDataset>(val_csv, DataMode::VALIDATION, data, target, args);
    }
    else if (task == "text_classification")
    {
        std::string word2vec_file =
            args.get("word2vec_file", "datasets/GloVE/glove.42B.300d.txt.bin");
        word2vec = std::make_unique<Word2VecBase>(word2vec_file);
        train = std::make_unique<TextClassification>(train_csv, word2vec.get(), args, data, target,
                                                     DataMode::TRAIN);
        val = std::make_unique<TextClassification>(val_csv, word2vec.get(), args, data, target,
                                                   DataMode::VALIDATION);
    }

    // Setup the trainer
    NetworkTrainer trainer(graph, *train, *val, args);
    trainer_ptr = &trainer;

    struct sigaction sa = {};
    sa.sa_handler = sigaction_handler;
    for (int32 i : {SIGINT, SIGTERM, SIGKILL}) sigaction(i, &sa, nullptr);

    trainer.train(args);
}

int main(int argc, char** argv)
{
    // Get the config file
    std::string config_file = "emotion_training.config";
    if (argc >= 2) config_file = argv[1];
    std::ifstream train_args_file(config_file);
    VarArgs args(train_args_file);

    // make the run directory and latest
    std::string run_dir = "runs";
    std::string run_name = args.get("run_name", std::string(""));
    std::string uniq_name = run_name + "-" + Context::get().session_id;
    std::string uniq_run_dir = run_dir + "/" + uniq_name;

    std::string sym_link_dst = run_dir + "/latest";

    std::filesystem::create_directories(uniq_run_dir);
    if (std::filesystem::exists(sym_link_dst)) std::filesystem::remove(sym_link_dst);
    std::filesystem::create_symlink(uniq_name, sym_link_dst);
    LOG(MAGENTA, "Run directory: ", uniq_run_dir);

    // copy config file to run directory
    std::string config_file_path = uniq_run_dir + "/" + config_file;
    std::filesystem::copy(config_file, config_file_path);

    args.set("run_dir", uniq_run_dir);
    train_network(args);
    return 0;
}
