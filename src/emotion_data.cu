#include <thread>
#include "emotion_data.hpp"

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
        split_line_to_words(line, words);
        if (words.size() > SEQ_LEN)
        {
            num_sentences_too_long++;
        }
        words.resize(std::min(words.size(), (size_t)SEQ_LEN));
        sentences.push_back(words);
        class_onehot.push_back(emotion);
    }

    index_swizzle.resize(sentences.size());
    std::iota(index_swizzle.begin(), index_swizzle.end(), 0);
    shuffle();

    features = std::make_unique<Input<>>(batch, SEQ_LEN, EMBEDDING_DIM, "features");
    target_node = std::make_unique<Input<>>(batch, 1, NUM_CLASSES, "target");

    LOG(GREEN, "Read ", sentences.size(), " from ", emotion_csv, " for ", num_batches(),
        " batches in ", timer.stop(), "s. ", num_sentences_too_long, " sentences too long.");

    prefect(0);
}

void EmotionData::shuffle()
{
    std::shuffle(std::begin(index_swizzle), std::end(index_swizzle), rdm::gen());
}

void EmotionData::split_line_to_words(std::string line, std::vector<std::string>& words)
{
    words.clear();
    std::string word;
    for (char c : line)
    {
        if (c == ' ' || c == '\n')
        {
            if (!word.empty())
            {
                words.push_back(word);
                word.clear();
            }
        }
        else
        {
            word.push_back(c);
        }
    }
}

uint32 EmotionData::num_batches(DataMode) const { return sentences.size() / features->batch(); }

void EmotionData::load(DataMode, uint32 batch)
{
    if (batch != last_fetched) prefect(batch);
    features->copy(temp_features.data());
    target_node->copy(temp_target.data());

    std::thread([this, batch]() { prefect((batch + 1) % num_batches()); }).detach();
}

void EmotionData::prefect(uint32 batch)
{
    std::lock_guard<std::mutex> lock(prefetching_mutex);
    uint32 idx = batch * features->batch();
    idx %= index_swizzle.size();

    idx = index_swizzle[idx];

    std::fill(temp_features.begin(), temp_features.end(), 0.f);
    std::fill(temp_target.begin(), temp_target.end(), 0.f);

    Vec300 empty = {1};

    for (uint32 b = 0; b < features->batch(); b++)
    {
        const auto& sentence = sentences[idx + b];
        auto offset = temp_features.begin() + b * SEQ_LEN * EMBEDDING_DIM;
        for (uint32 i = 0; i < sentence.size(); i++)
        {
            auto node = word2vec[sentence[i]];
            auto src = node ? node->vec : empty;
            offset = std::copy(src.begin(), src.end(), offset);
        }

        temp_target[b * NUM_CLASSES + class_onehot[idx + b]] = 1.f;
    }
    last_fetched = batch;
    if (total_loaded++ % num_batches() == 0)
    {
        last_fetched = UINT32_MAX;
        shuffle();
    }
}
