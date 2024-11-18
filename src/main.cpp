#include "../headers/learning_nodes.hpp"
#include "../headers/logger.hpp"
#include "../headers/loss_nodes.hpp"
#include "../headers/matrix_ops.cuh"
#include "../headers/matrix_ops.hpp"
#include "../headers/word2vec.hpp"
#include "fstream"

struct EmotionData
{
    std::vector<std::vector<std::string>> sentences;
    std::vector<uint32> class_onehot;
    const Word2Vec& word2vec;
    static constexpr uint32 SEQ_LEN = 64;
    static constexpr uint32 EMBEDDING_DIM = N;
    static constexpr uint32 NUM_CLASSES = 6;

    std::array<FloatT, SEQ_LEN * EMBEDDING_DIM> temp_sample;

    Input<> sample{SEQ_LEN, EMBEDDING_DIM, "sample"};
    Input<> target{NUM_CLASSES, 1, "target"};

    std::set<std::string> missing;
    EmotionData(std::string emotion_csv, const Word2Vec* word2vec, uint32 max_samples = -1)
        : word2vec(*word2vec)
    {
        std::ifstream emotion_file(emotion_csv);
        std::string line;
        uint32 emotion;
        emotion_file >> line;  // skip header
        Timer timer("Reading " + emotion_csv);
        std::vector<std::string> words;
        while (emotion_file and sentences.size() < max_samples)
        {
            // read till comma:
            std::getline(emotion_file, line, ',');
            if (line.empty()) continue;
            emotion_file >> emotion;
            split_line_to_words(line, words);
            words.resize(std::min(words.size(), (size_t)64));
            sentences.push_back(words);
            class_onehot.push_back(emotion);
        }
        LOG(GREEN, "Read ", sentences.size(), " lines from ", emotion_csv, " in ", timer.stop(),
            "s.");
    }

    void split_line_to_words(std::string line, std::vector<std::string>& words)
    {
        words.clear();
        std::string word;
        for (char c : line)
        {
            if (c == ' ' or c == '\n')  // space or newlines
            {
                if (!word.empty()) words.push_back(word);
                word.clear();
            }
            else
            {
                word.push_back(c);
            }
        }
        if (!word.empty()) words.push_back(word);
    }

    uint32 size() { return sentences.size(); }

    void load(uint32 idx)
    {
        if (idx >= size())
            throw_rte_with_backtrace("Index out of bounds: " + std::to_string(idx) +
                                     " >= " + std::to_string(size()));
        std::fill(target.begin(), target.end(), 0);
        *(target.begin() + class_onehot[idx]) = 1;
        std::fill(temp_sample.begin(), temp_sample.end(), 0);
        auto offset = temp_sample.begin();
        Vec300 empty = {1};

        for (uint32 i = 0; i < sentences[idx].size(); i++)
        {
            auto node = word2vec[sentences[idx][i]];
            auto src = node ? node->vec : empty;
            offset = std::copy(src.begin(), src.end(), offset);
        }
        fill(sample, temp_sample.data());
    }
};

int train()
{
    Word2Vec word2vec("/home/ishwark/word2vecdata/wiki.multi.en.vec");
    EmotionData train_data("/home/ishwark/word2vecdata/emotion/split/train.csv", &word2vec);

    MultiHeadAttention<> MHA1(3, 64, {64, &train_data.sample, true, "In"}, "MHA1");

    MLP<FloatT, Relu<FloatT>, Sigmoid<FloatT>> MLP1(32, {64, &MHA1, true, "MLP1"}, 0.25, "MLP1");

    MultiHeadAttention<> MHA2(3, 16, {6, &MLP1, false, "In"}, "MHA2");

    MLP<FloatT, Relu<FloatT>, Sigmoid<FloatT>> MLP2(6, {6, &MHA2, true, "MLP2"}, 0.25, "MLP2");

    Transpose<> T1({&MLP2}, "T1");
    Linear<> output(1, &T1, true, "OutputLinear");
    SoftmaxDim0<FloatT> softmax(&output, "Softmax");
    CrossEntropyLoss<> celoss({&train_data.target, &softmax}, "CELoss");
    std::set<Parameter<FloatT, FloatT>*> seen;
    std::ofstream dot("emotion_classification.dot");
    graph_to_dot(&celoss, dot);

    std::cout << "Trainable params:\n"
              << celoss.n_trainable_params(seen) << std::endl
              << "Total alloced matrices: " << MatrixInitUitls::peek_id() << std::endl
              << " consuming " << MatrixInitUitls::get_alloced_bytes() << " bytes" << std::endl;

    train_data.load(0);
    Timer timer("Training");
    const uint32 max_iters = 100;
    for (uint32 i = 0; i < max_iters; i++)
    {
        train_data.load(i);
        celoss.compute();
        celoss.backward();
    }
    auto time = timer.stop();
    LOG(GREEN, "Training took ", time, "s. time per iteration: ", (1000 * time) / max_iters, "ms.");
    cudaErrCheck(cudaDeviceSynchronize());

    print(MHA1.heads[0]->Q.W.grads, "MHA1.heads[0]->W.grads");
    softmax.update_weights(0.1);
    return 0;
}

int main() { return train(); }