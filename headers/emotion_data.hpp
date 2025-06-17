/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef EMOTION_DATA_HPP
#define EMOTION_DATA_HPP

#include "dataset.hpp"
#include "nodes/unparameterized.hpp"
#include "word2vec.hpp"

struct EmotionDataInput;

struct EmotionData : public Dataset
{
    static constexpr uint32 EMBEDDING_DIM = WORD2VEC_DIM;
    static constexpr uint32 NUM_CLASSES = 6;

    std::vector<std::vector<std::string>> sentences;
    std::vector<uint32> m_emotion_id;
    const Word2VecBase& word2vec;
    const uint32 SEQ_LEN;

    std::vector<uint32> index_swizzle;
    uint32 total_loaded = 0;

    std::vector<FloatT> m_temp_features;
    std::vector<FloatT> m_temp_target;
    std::mutex prefetching_mutex;

    EmotionData(std::string emotion_csv, uint32 batch, const Word2VecBase* word2vec,
                Input<FloatT>* data = nullptr, Input<FloatT>* target = nullptr,
                DataMode mode = DataMode::TRAIN, bool shuffle = true,
                uint32 max_samples = std::numeric_limits<uint32>::max());

    void shuffle();

    bool split_line_to_words(std::string line, std::vector<std::string>& words, uint32 max_len);

    void load(uint32 batch_idx) override;

    void prefetch(uint32 idx);

    uint32 input_size() const { return EMBEDDING_DIM; }
    uint32 target_size() const { return NUM_CLASSES; }
};

#endif  // EMOTION_DATA_HPP
