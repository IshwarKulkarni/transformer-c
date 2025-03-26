#ifndef EMOTION_DATA_HPP
#define EMOTION_DATA_HPP

#include <thread>
#include "datasets.hpp"
#include "nodes/unparameterized.hpp"
#include "word2vec.hpp"

struct EmotionDataInput;

struct FakeData
{
    static constexpr uint32 SEQ_LEN = 8;
    static constexpr uint32 EMBEDDING_DIM = 12;
    static constexpr uint32 NUM_CLASSES = 4;

    std::unique_ptr<Input<>> features;
    std::unique_ptr<Input<>> target_node;

    FakeData(std::string, uint32 batch, const Word2Vec*)
    {
        features = std::make_unique<Input<>>(batch, SEQ_LEN, EMBEDDING_DIM, "features");
        target_node = std::make_unique<Input<>>(batch, 1, NUM_CLASSES, "target");
    }

    uint32 num_batches(DataMode mode = DataMode::TRAIN) const
    {
        (void)(mode);
        return 1000;
    }

    void load(DataMode, uint32)
    {
        normal_init(*features, 0.0, 1.0);
        target_node->set_val(0.f);

        for (uint32 i = 0; i < target_node->batch(); i++)
            for (uint32 j = 0; j < target_node->height(); j++)
                (*target_node)(i, j, rdm::next_urand<uint32>(0, NUM_CLASSES)) = 1.0;
    }

    uint32 target_size() const { return NUM_CLASSES; }

    Input<>* target() { return target_node.get(); }
};

struct EmotionData
{
    std::vector<std::vector<std::string>> sentences;
    std::vector<uint32> class_onehot;
    const Word2Vec& word2vec;
    static constexpr uint32 SEQ_LEN = 64;
    static constexpr uint32 EMBEDDING_DIM = WORD2VEC_DIM;
    static constexpr uint32 NUM_CLASSES = 6;
    std::vector<uint32> index_swizzle;
    uint32 total_loaded = 0;
    uint32 last_fetched = UINT32_MAX;
    const uint32 batch = 0;

    std::vector<FloatT> temp_features;
    std::vector<FloatT> temp_target;

    std::unique_ptr<Input<>> features_node;
    std::unique_ptr<Input<>> target_node;

    std::mutex prefetching_mutex;

    EmotionData(std::string emotion_csv, uint32 batch, const Word2Vec* word2vec,
                uint32 max_featuress = std::numeric_limits<uint32>::max());

    void shuffle();

    bool split_line_to_words(std::string line, std::vector<std::string>& words, uint32 max_len);

    uint32 num_batches(DataMode mode = DataMode::TRAIN) const;

    void load(DataMode mode, uint32 batch);

    void prefetch(uint32 idx);

    uint32 target_size() const { return NUM_CLASSES; }

    Input<>* target() { return target_node.get(); }
};

#endif  // EMOTION_DATA_HPP
