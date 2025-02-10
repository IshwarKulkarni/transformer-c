#include "datasets.hpp"
#include "emotion_data.hpp"
#include "nodes/loss.hpp"
#include "nodes/parameterized.hpp"

uint32 count_misses(const Matrix<FloatT>* softmax, const Matrix<FloatT>* target)
{
    uint32 misses = 0;
    for (uint32 b = 0; b < target->batch(); b++)
    {
        for (uint32 y = 0; y < target->height(); y++)
        {
            uint32 row_offset = softmax->shape.offset(b, y, 0);
            const auto* sm = softmax->begin() + row_offset;
            const auto* tg = target->begin() + row_offset;

            auto max_idx = std::max_element(sm, sm + softmax->width()) - sm;
            auto target_idx = std::max_element(tg, tg + target->width()) - tg;
            if (max_idx != target_idx) misses++;
        }
    }
    return misses;
}

template <typename DatasetT>
std::pair<FloatT, uint32> run_validation(Loss2Node<FloatT>& loss, DatasetT& dataset)
{
    const auto* softmax = loss.predictions;
    loss.set_is_training(false);
    float32 valdn_loss = 0;
    uint32 misses = 0;
    for (uint32 j = 0; j < dataset.num_batches(DataMode::TEST); j++)
    {
        dataset.load(DataMode::TEST, j);
        loss.compute();
        valdn_loss += loss.value();
        misses += count_misses(softmax, dataset.target());
    }
    valdn_loss /= dataset.num_batches(DataMode::TEST);
    loss.set_is_training(true);
    return {valdn_loss, misses};
}

std::pair<FloatT, uint32> run_validation_emo(Loss2Node<FloatT>& loss, EmotionData& dataset)
{
    const auto* softmax = loss.predictions;
    loss.set_is_training(false);
    float32 valdn_loss = 0;
    uint32 misses = 0;
    for (uint32 j = 0; j < dataset.num_batches(); j++)
    {
        loss.compute();
        valdn_loss += loss.value();
        misses += count_misses(softmax, dataset.target());
    }
    valdn_loss /= dataset.num_batches();
    loss.set_is_training(true);
    return {valdn_loss, misses};
}

int train_MLP_wine()
{
    Timer timer_data("Data Loading");
    ContinuousDataset dataset("static_data/wine-quality.csv", 32, .1f);
    dataset.normalize();
    timer_data.stop(true);

    using LinR = Linear<FloatT, Relu<FloatT>>;
    using LinI = Linear<FloatT, IActivation<FloatT>>;

    LinR l1(LinR::LinearInput{32, dataset.input(), true, "Lin1"});
    Dropout<FloatT> d1(0.5, &l1, "Dropout1");

    LinR l2(LinR::LinearInput{16, &d1, true, "Lin2"});
    Dropout<FloatT> d2(0.25, &l2, "Dropout2");

    LinR l3(LinR::LinearInput{12, &d2, true, "Lin2"});
    Dropout<FloatT> d3(0.25, &l3, "Dropout2");

    LinI lf(LinI::LinearInput{dataset.target_size(), &d3, true, "Final"});
    L1Loss<> loss({&lf, dataset.target()}, "L2Loss");

    std::ofstream dot("mlp.dot");
    graph_to_dot(&loss, dot);

    std::ofstream train_csv("train_losses.csv");
    train_csv << "batch,loss,misses\n";

    std::ofstream valdn_csv("valdn_losses.csv");
    valdn_csv << "batch,loss,misses\n";

    Timer timer("MLP_Train");

    uint32 epoch = 0;
    uint32 train_batches = dataset.num_batches(DataMode::TRAIN);
    for (uint32 batch = 0; batch < 400 or epoch < 25; batch++)
    {
        epoch = batch / train_batches;
        dataset.load(DataMode::TRAIN, batch % train_batches);

        loss.compute();
        loss.backward();
        loss.update_weights(3e-3);

        // run validation and count misses
        auto train_loss = loss.value();

        train_csv << batch << ',' << loss.value() << ',' << 0 << '\n';

        if (batch % 80 == 0) LOG("Epoch: ", epoch, " Batch: ", batch, " -  T: ", train_loss);
        if (batch % train_batches - 1 == 0)
        {
            auto [valdn_loss, misses] = run_validation(loss, dataset);
            valdn_csv << batch << ',' << valdn_loss << ',' << misses << '\n';
            LOG("Epoch: ", epoch, " Batch: ", batch, " -  V: ", valdn_loss);
        }
    }

    if (dataset.target_size() == 1)
    {
        loss.set_is_training(false);
        for (uint32 b = 0; b < lf.batch(); ++b)
        {
            auto out = lf(b, 0, 0);
            auto tgt = (*dataset.target())(b, 0, 0);
            printf("%d:\t %f\t%d\t %f\n", b, out, uint32(tgt), out - tgt);
        }
    }
    return 0;
}

int train_MLP_iris()
{
    Timer timer_data("Data Loading");
    CategoricalDataset dataset("static_data/iris.csv", 15, .1f);
    dataset.normalize();
    timer_data.stop(true);

    using LinR = Linear<FloatT, Relu<FloatT>>;
    using LinI = Linear<FloatT, IActivation<FloatT>>;

    LinR l1(LinR::LinearInput{64, dataset.input(), true, "Lin1"});
    Dropout<FloatT> d1(0.5, &l1, "D1");

    LinR l2(LinR::LinearInput{32, &d1, true, "Lin2"});
    Dropout<FloatT> d2(0.5, &l2, "D2");

    LinI lf(LinI::LinearInput{dataset.target_size(), &d2, true, "Lin2"});
    SoftmaxDim0<> softmax({&lf});
    L2Loss<> loss({&softmax, dataset.target()}, "L2Loss");

    std::ofstream dot("mlp.dot");
    graph_to_dot(&loss, dot);

    std::ofstream train_csv("train_losses.csv");
    train_csv << "batch,loss,misses\n";

    std::ofstream valdn_csv("valdn_losses.csv");
    valdn_csv << "batch,loss,misses\n";

    Timer timer("MLP_Train");

    for (uint32 batch = 0; batch < 500; batch++)
    {
        uint32 epoch = batch / dataset.num_batches(DataMode::TRAIN);
        dataset.load(DataMode::TRAIN, batch % dataset.num_batches(DataMode::TRAIN));

        loss.compute();
        loss.backward();
        loss.update_weights(3e-3);

        // run validation and count misses
        auto train_loss = loss.value();

        train_csv << batch << ',' << loss.value() << ',' << 0 << '\n';

        if (batch % 10 == 0) LOG("Epoch: ", epoch, " Batch: ", batch, " -  T: ", train_loss);
        if (batch % dataset.num_batches(DataMode::TRAIN) - 1 == 0)
        {
            auto [valdn_loss, misses] = run_validation(loss, dataset);
            valdn_csv << batch << ',' << valdn_loss << ',' << misses << '\n';
            LOG("Epoch: ", epoch, " Batch: ", batch, " -  V: ", valdn_loss);
        }
    }

    return 0;
}

int train_emotion()
{
    Timer timer_data("Data Loading");
    Word2Vec word2vec("static_data/glove.42B.300d.txt", 10000);

    uint32 nbatch = 4;
    EmotionData train_dataset("static_data/emotion_train.csv", nbatch, &word2vec);
    EmotionData valdn_dataset("static_data/emotion_validation.csv", nbatch, &word2vec);
    timer_data.stop(true);

    using LinR = Linear<FloatT, Relu<FloatT>>;
    using LinI = Linear<FloatT, IActivation<FloatT>>;

    LinR l1(LinR::LinearInput{64, train_dataset.features.get(), true, "Lin1"});
    Dropout<FloatT> d1(0.5, &l1, "Dropout1");

    SelfAttention<FloatT> sa1(32, &d1, "SelfAttention");
    Normalize<FloatT, 1> norm1(&sa1, "Norm1");
    Dropout<FloatT> d2(0.5, &norm1, "Dropout2");

    Mean<FloatT, 1> sum({&d2}, "Mean");
    LinI lf(LinI::LinearInput{train_dataset.target_size(), &sum, true, "Final"});
    LogSoftmaxCELoss<> loss({&lf, train_dataset.target()}, "CELoss");

    std::ofstream dot("emotion_train.dot");
    graph_to_dot(&loss, dot);

    std::ofstream train_csv("train_losses.csv");
    train_csv << "batch,loss,misses\n";

    std::ofstream valdn_csv("valdn_losses.csv");
    valdn_csv << "batch,loss,misses\n";

    uint32 epoch = 0;
    Timer timer("Emotion_Train");
    for (uint32 batch = 0; batch < 100 or epoch < 25; batch++)
    {
        epoch = batch / train_dataset.num_batches();
        train_dataset.load(DataMode::TRAIN, batch);

        loss.compute();
        loss.backward();
        cudaErrCheck(cudaDeviceSynchronize());
        loss.update_weights(3e-3);

        // run validation and count misses
        auto train_loss = loss.value();
        train_csv << batch << ',' << loss.value() << ',' << 0 << '\n';
        if (batch % 10 == 0) LOG("Epoch: ", epoch, " Batch: ", batch, " -  T: ", train_loss);
        if (std::isnan(train_loss))
        {
            LOG("NaN detected in training loss. Exiting.");
            NodePtr<FloatT> temp = &loss;
            std::set<NodePtr<FloatT>> visited;
            while (temp != nullptr)
            {
                if (visited.find(temp) == visited.end())
                {
                    if (temp) LOG(*temp);
                }
                visited.insert(temp);
                temp = temp->prev_nodes[0];
            }
            loss.debug_print();
            return 1;
        }
        if (batch % train_dataset.num_batches() - 1 == 0)
        {
            auto [valdn_loss, misses] = run_validation_emo(loss, valdn_dataset);
            valdn_csv << batch << ',' << valdn_loss << ',' << misses << '\n';
            LOG("Epoch: ", epoch, " Batch: ", batch, " -  V: ", valdn_loss);
        }
    }
    return 0;
}

int main()
{
    train_emotion();
    return 0;
}