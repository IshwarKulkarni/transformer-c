#include "emotion_data.hpp"
#include <thread>

EmotionData::EmotionData(std::string emotion_csv, uint32 batch, const Word2Vec* word2vec,
                         uint32 max_samples)
    : word2vec(*word2vec),
      batch(batch),
      temp_features(batch * SEQ_LEN * EMBEDDING_DIM),
      temp_target(batch * 1 * NUM_CLASSES)
{
    std::ifstream emotion_file(emotion_csv);
    std::string line;
    uint32 emotion;
    emotion_file >> line;  // skip header
    Timer timer("Reading " + emotion_csv);
    std::vector<std::string> words;
    uint32 num_sentences_too_long = 0;
    while (emotion_file && sentences.size() < max_samples)
    {
        std::getline(emotion_file, line, ',');
        if (line.empty()) continue;
        emotion_file >> emotion;
        num_sentences_too_long += split_line_to_words(line, words, SEQ_LEN);
        if (words.empty()) break;
        sentences.push_back(words);
        class_onehot.push_back(emotion);
    }

    index_swizzle.resize(sentences.size());
    std::iota(index_swizzle.begin(), index_swizzle.end(), 0);
    shuffle();

    features_node = std::make_unique<Input<>>(batch, SEQ_LEN, EMBEDDING_DIM, "features");
    target_node = std::make_unique<Input<>>(batch, 1, NUM_CLASSES, "target");

    LOG(GREEN, "Read ", sentences.size(), " from ", emotion_csv, " for ", num_batches(),
        " batches in ", timer.stop(), "s. ", num_sentences_too_long, " sentences too long.");
}

void EmotionData::shuffle()
{
    std::shuffle(std::begin(index_swizzle), std::end(index_swizzle), rdm::gen());
}

// split the line by space, and insert them to `words` upto max of `max_len`, return if there were
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

uint32 EmotionData::num_batches(DataMode) const
{
    return sentences.size() / features_node->batch();
}

void EmotionData::load(DataMode, uint32 batch)
{
    if (batch != last_fetched) prefetch(batch);
    features_node->copy(temp_features.data());
    target_node->copy(temp_target.data());
}

void EmotionData::prefetch(uint32 fetch_index)
{
    // std::lock_guard<std::mutex> lock(prefetching_mutex);
    uint32 idx = fetch_index * features_node->batch();
    idx %= index_swizzle.size();

    std::fill(temp_features.begin(), temp_features.end(), 0.f);
    std::fill(temp_target.begin(), temp_target.end(), 0.f);

    Vec300 empty = {1};
    for (uint32 s = 0; s < features_node->batch(); s++)
    {
        const auto& sentence = sentences[index_swizzle[idx + s]];
        uint32 offset = s * EMBEDDING_DIM * SEQ_LEN;
        for (uint32 w = 0; w < sentence.size(); w++)
        {
            auto node = word2vec[sentence[w]];
            auto& src = node ? node->vec : empty;
            std::copy(src.begin(), src.end(), temp_features.begin() + offset);
            offset += EMBEDDING_DIM;
        }
        temp_target[s * NUM_CLASSES + class_onehot[idx + s]] = 1.f;
    }
    last_fetched = fetch_index;
    if (total_loaded++ % num_batches() == 0)
    {
        last_fetched = UINT32_MAX;
        shuffle();
    }
}
