/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include <initializer_list>
#include "csv_dataset.hpp"
#include "emotion_data.hpp"
#include "network_graph.hpp"
#include "signal.h"
#include "string_utils.hpp"
#include "trainer.hpp"
#include "word2vec.hpp"

void test_empty_kernel();

NetworkTrainer* trainer_ptr = nullptr;

inline void sigaction_handler(int32)
{
    trainer_ptr->end_training();
    exit(0);
}

inline void train_network(std::string config_file)
{
    std::ifstream train_args_file(config_file);
    VarArgs args(train_args_file);

    uint32 batch = args.get<uint32>("batch", 32);
    NetworkGraph graph;
    graph.define_variable("$batch", batch);

    std::string graph_file = args.get("graph_file", "");
    std::ifstream net_desc(graph_file);

    graph.load_from_desc_stream(net_desc);
    graph.write_dotviz(args.get("dot_file", "abalone-model.dot"));
    graph.print_nodes();

    auto data = graph.get_typed_node<Input<FloatT>>("input");
    auto target = graph.get_typed_node<Input<FloatT>>("target");

    std::string train_csv = args.get("train_dataset_file", "");
    std::string val_csv = args.get("valdn_dataset_file", "");

    if (train_csv.empty() || val_csv.empty())
        throw std::runtime_error("train_dataset_file and valdn_dataset_file must be provided");

    CSVDataset train(train_csv, DataMode::TRAIN, data, target, args);
    CSVDataset val(val_csv, DataMode::VALIDATION, data, target, args);

    NetworkTrainer trainer(graph, train, val);
    trainer_ptr = &trainer;

    struct sigaction sa;
    sa.sa_handler = sigaction_handler;
    for (int32 i : {SIGINT, SIGTERM, SIGKILL}) sigaction(i, &sa, nullptr);

    LOG(GREEN, "Training...");

    trainer.train(args);
}

int main(int argc, char** argv)
{
    std::string config_file = "training.config";
    if (argc >= 2) config_file = argv[1];

    train_network(config_file);
}
