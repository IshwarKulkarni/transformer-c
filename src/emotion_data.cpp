#include "../headers/emotion_data.hpp"

EmotionData::EmotionData(std::string emotion_csv, const Word2Vec* word2vec, uint32 max_samples)
    : word2vec(*word2vec)
{
    std::ifstream emotion_file(emotion_csv);
    std::string line;
    uint32 emotion;
    emotion_file >> line;  // skip header
    Timer timer("Reading " + emotion_csv);
    std::vector<std::string> words;
    while (emotion_file && sentences.size() < max_samples)
    {
        std::getline(emotion_file, line, ',');
        if (line.empty()) continue;
        emotion_file >> emotion;
        split_line_to_words(line, words);
        words.resize(std::min(words.size(), (size_t)SEQ_LEN));
        sentences.push_back(words);
        class_onehot.push_back(emotion);
    }
    LOG(GREEN, "Read ", sentences.size(), " from ", emotion_csv, " in ", timer.stop(), "s.");
    index_swizzle.resize(sentences.size());
    std::iota(index_swizzle.begin(), index_swizzle.end(), 0);
    shuffle();
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

uint32 EmotionData::size() { return sentences.size(); }

void EmotionData::load(uint32 idx)
{
    loaded++;
    if (loaded % size() == 0)
    {
        shuffle();
    }
    idx = index_swizzle[idx % index_swizzle.size()];
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