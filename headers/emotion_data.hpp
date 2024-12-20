#ifndef EMOTION_DATA_HPP
#define EMOTION_DATA_HPP

#include <array>
#include <string>
#include <vector>
#include "../headers/word2vec.hpp"
#include "nodes.hpp"

struct EmotionData
{
    std::vector<std::vector<std::string>> sentences;
    std::vector<uint32> class_onehot;
    const Word2Vec& word2vec;
    static constexpr uint32 SEQ_LEN = 64;
    static constexpr uint32 EMBEDDING_DIM = N;
    static constexpr uint32 NUM_CLASSES = 6;
    std::vector<uint32> index_swizzle;
    uint32 loaded = 0;

    std::array<FloatT, SEQ_LEN * EMBEDDING_DIM> temp_sample;

    Input<> sample{SEQ_LEN, EMBEDDING_DIM, "sample"};
    Input<> target{NUM_CLASSES, 1, "target"};

    EmotionData(std::string emotion_csv, const Word2Vec* word2vec, uint32 max_samples = -1);

    void shuffle();

    void split_line_to_words(std::string line, std::vector<std::string>& words);

    uint32 size();

    void load(uint32 idx);
};

#endif  // EMOTION_DATA_HPP
