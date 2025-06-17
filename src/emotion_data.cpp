/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#include "emotion_data.hpp"
#include <thread>
#include "dataset.hpp"

EmotionData::EmotionData(std::string emotion_csv, uint32 batch, const Word2VecBase* word2vec,
                         Input<FloatT>* data, Input<FloatT>* target, DataMode mode,
                         bool should_shuffle, uint32 max_samples)
    : Dataset(mode, should_shuffle, data, target), word2vec(*word2vec), SEQ_LEN(data->height())
{
    if (!data || !target)
    {
        throw_rte_with_backtrace("Data || target node not set");
    }

    if (data->batch() != target->batch())
    {
        throw_rte_with_backtrace("Data &&target batch sizes do not match");
    }

    std::ifstream emotion_file(emotion_csv);
    std::string line;
    uint32 emotion;
    emotion_file >> line;  // skip header
    uint32 colmn_count = std::count(line.begin(), line.end(), ',') + 1;

    LOG(GREEN, "Number of columns in the csv file: ", colmn_count);

    Timer timer("Reading " + emotion_csv);
    std::vector<std::string> words;
    uint32 num_sentences_too_long = 0;
    std::vector<uint32> one_hot;
    while (emotion_file && sentences.size() < max_samples)
    {
        std::getline(emotion_file, line, ',');
        if (line.empty()) continue;
        emotion_file >> emotion;
        num_sentences_too_long += split_line_to_words(line, words, SEQ_LEN);
        if (words.empty()) break;
        sentences.push_back(words);
        m_emotion_id.push_back(emotion);
    }

    index_swizzle.resize(sentences.size());
    std::iota(index_swizzle.begin(), index_swizzle.end(), 0);
    if (m_shuffle)
    {
        shuffle();
    }

    set_num_batches(sentences.size() / batch);

    LOG(GREEN, "Read ", sentences.size(), " sentences from ", emotion_csv, " for ", batches(),
        " batches in ", timer.stop(), ". ", num_sentences_too_long,
        " sentences too long for sequence length ", SEQ_LEN);
}

void EmotionData::shuffle()
{
    std::shuffle(std::begin(index_swizzle), std::end(index_swizzle), rdm::gen());
}

// split the line by space, &&insert them to `words` upto max of `max_len`, return if there were
// more words than `max_len`
bool EmotionData::split_line_to_words(std::string line, std::vector<std::string>& words,
                                      uint32 max_len)
{
    std::istringstream iss(line);
    std::string word;
    words.clear();
    uint32 word_count = 0;
    while (iss >> word)
    {
        if (word_count < max_len) words.push_back(word);
        word_count++;
    }
    return word_count > max_len;
}

void EmotionData::load(uint32 idx)
{
    // std::lock_guard<std::mutex> lock(prefetching_mutex);
    auto features = features_node();
    auto target = target_node();
    if (!features || !target)
    {
        throw_rte_with_backtrace("Data || target node not set");
    }

    uint32 batch_size = features->batch();

    WORDVEC empty = {1};

    m_temp_features.resize(batch_size * SEQ_LEN * EMBEDDING_DIM);
    m_temp_target.resize(batch_size * NUM_CLASSES);
    // set all values to nan
    auto nan = std::numeric_limits<FloatT>::quiet_NaN();
    std::fill(m_temp_features.begin(), m_temp_features.end(), nan);
    std::fill(m_temp_target.begin(), m_temp_target.end(), 0);

    for (uint32 b = 0; b < batch_size; b++)
    {
        uint32 data_idx = index_swizzle[(idx + b) % sentences.size()];
        uint32 offset = b * SEQ_LEN * EMBEDDING_DIM;
        const auto& sentence = sentences[data_idx];
        for (uint32 w = 0; w < sentence.size(); w++)
        {
            auto node = word2vec[sentence[w]];
            auto& src = node ? node->vec : empty;
            std::copy(src.begin(), src.end(), m_temp_features.begin() + offset);
            offset += EMBEDDING_DIM;
        }
        uint32 emotion_id = m_emotion_id[data_idx];
        m_temp_target[b * NUM_CLASSES + emotion_id] = 1;
        features->set_extent<HEIGHT_IDX>(b, std::min<uint32>(sentence.size(), SEQ_LEN));
        // set values after the sentence length to NaN
        for (uint32 i = sentence.size() * EMBEDDING_DIM; i < features->numels(); i++)
        {
            m_temp_features[i] = std::numeric_limits<FloatT>::quiet_NaN();
        }
    }

    features->copy(m_temp_features.data());
    target->copy(m_temp_target.data());
    if (total_loaded++ % batches() == 0)
    {
        if (m_shuffle) shuffle();
    }
}
