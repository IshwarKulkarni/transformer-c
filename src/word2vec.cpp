#include "word2vec.hpp"
#include <fstream>
#include <iostream>
#include "logger.hpp"
#include "matrix_ops_cpu.hpp"
#include "utils.hpp"

using Node = WordVecNode;

std::ostream& operator<<(std::ostream& os, const Vec300& arr)
{
    for (uint32 i = 0; i < WORD2VEC_DIM; i++)
    {
        os << arr[i] << ", ";
    }
    return os;
}

std::istream& operator>>(std::istream& is, Vec300& arr)
{
    for (uint32 i = 0; i < WORD2VEC_DIM; i++)
    {
        is >> arr[i];
    }
    return is;
}

std::ostream& operator<<(std::ostream& os, const WordVecNode& node)
{
    char line[256] = {0};
    bool left = node.parent and node.parent->left and node.parent->left->id == node.id;
    snprintf(line, 256, "%18s [%8d/%3d %c] ", node.word.c_str(), node.id, node.depth,
             left ? 'L' : 'R');
    os << line;
    return os;
}

FloatT add_noise(Vec300& vec, FloatT mean, FloatT std)
{
    Vec300 orig = vec;
    auto& gen = rdm::gen();
    std::normal_distribution<FloatT> d(mean, std);
    for (uint32 i = 0; i < WORD2VEC_DIM; i++) vec[i] += d(gen);

    return l2_dist2(orig, vec);
}

FloatT add_noise_normal(Vec300& vec, FloatT mean, FloatT std)
{
    Vec300 orig = vec;
    add_noise(vec, mean, std);
    FloatT norm = 0;
    for (uint32 i = 0; i < WORD2VEC_DIM; i++) norm += vec[i] * vec[i];
    norm = std::sqrt(norm);
    for (uint32 i = 0; i < WORD2VEC_DIM; i++) vec[i] /= norm;
    return cos_sim(orig, vec);
}

std::array<std::pair<float32, bool>, WORD2VEC_DIM> diff(const Vec300& a, const Vec300& b,
                                                        float32 eps, uint32* count)
{
    std::array<std::pair<float32, bool>, WORD2VEC_DIM> diffs;
    for (uint32 i = 0; i < WORD2VEC_DIM; i++)
    {
        diffs[i].first = a[i] - b[i];
        diffs[i].second = std::abs(diffs[i].first) > eps;
        if (count && diffs[i].second) (*count)++;
    }
    return diffs;
}

void print_path(const WordVecNode* node)
{
    auto path = get_ancestors(node, true);
    for (auto an : path)
        printf("%2d>%6s: %2.2f%c| ", an->depth, an->word.c_str(), an->dist2(node),
               is_left_child(an) ? 'L' : 'R');
    printf("\n");
}

static void write_binary(const std::string& filename, const std::vector<WordVecPair>& wordVecPairs)
{
    Timer timer("Writing Binary");
    uint32 max_word_len = 0;
    for (const WordVecPair& pair : wordVecPairs)
    {
        max_word_len = std::max(max_word_len, (uint32)pair.first.size());
    }
    std::ofstream file(filename, std::ios::binary);
    uint32 n_words = wordVecPairs.size();
    uint32 vec_size = WORD2VEC_DIM;
    file.write((char*)&n_words, sizeof(uint32));
    file.write((char*)&vec_size, sizeof(uint32));
    file.write((char*)&max_word_len, sizeof(uint32));

    char* word_str = new char[max_word_len];
    for (const WordVecPair& pair : wordVecPairs)
    {
        memset(word_str, 0, max_word_len);
        snprintf(word_str, max_word_len - 1, "%s", pair.first.c_str());

        file.write(word_str, max_word_len);
        file.write((char*)&pair.second[0], WORD2VEC_DIM * sizeof(float32));
    }
    delete[] word_str;
}

static bool read_binary(const std::string& filename, std::vector<WordVecPair>& wordVecPairs)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }
    Timer timer("Reading Word2Vec Binary");
    uint32 n_words, vec_size, max_word_len;
    file.read((char*)&n_words, sizeof(uint32));
    file.read((char*)&vec_size, sizeof(uint32));
    file.read((char*)&max_word_len, sizeof(uint32));
    if (vec_size != WORD2VEC_DIM)
    {
        LOG(RED, "Vector size mismatch: ", vec_size, " != ", WORD2VEC_DIM);
        throw_rte_with_backtrace("Vector size mismatch");
    }
    wordVecPairs.reserve(n_words);

    std::vector<char> word_str(max_word_len);
    Vec300 vec;
    for (uint32 i = 0; i < n_words; i++)
    {
        word_str.clear();
        file.read(word_str.data(), max_word_len);
        file.read((char*)&vec[0], WORD2VEC_DIM * sizeof(float32));
        wordVecPairs.push_back({word_str.data(), vec});
    }
    return true;
}

static bool read_text(const std::string& filename, std::vector<WordVecPair>& wordVecPairs,
                      uint32 max_dict_size)
{
    if (endswith(filename, ".bin") && !read_binary(filename, wordVecPairs))
    {
        throw std::runtime_error("Binary failed to load");
    }
    if (read_binary(filename + ".bin", wordVecPairs))
    {
        LOG(YELLOW, "Loaded ", wordVecPairs.size(),
            " words from binary, check number of words read");
        return true;
    }
    Timer timer("Reading Text");
    std::ifstream file(filename, std::ios::binary);
    uint32 n_words, vec_size;
    file >> n_words >> vec_size;
    if (vec_size != WORD2VEC_DIM)
    {
        LOG(RED, "Vector size mismatch: ", vec_size, " != ", WORD2VEC_DIM);
        throw_rte_with_backtrace("Vector size mismatch");
    }
    if (n_words > max_dict_size)
    {
        LOG(YELLOW, "Truncating dictionary from ", n_words, " to ", max_dict_size);
        n_words = max_dict_size;
    }
    wordVecPairs.reserve(n_words);
    for (uint32 i = 0; i < n_words && file; ++i)
    {
        std::string word;
        Vec300 vec;
        file >> word >> vec;
        wordVecPairs.push_back(WordVecPair(word, vec));
        if (i % 1000 == 0) progress_bar(i, n_words);
    }
    write_binary(filename + ".bin", wordVecPairs);
    return true;
}

Word2Vec::Word2Vec(const std::string& filename, uint32 max_dict_size)
{
    std::vector<WordVecPair> wordVecPairs;
    read_text(filename, wordVecPairs, max_dict_size);

    Timer timer("Tree Building ");

    nodes.reserve(wordVecPairs.size() + 1);
    root = build(wordVecPairs.begin(), wordVecPairs.end(), 0);
    for (const auto* node : nodes)
    {
        word2Node[node->word] = node;
    }

    std::sort(nodes.begin(), nodes.end(),
              [](const WordVecNode* a, const WordVecNode* b) { return a->id < b->id; });

    LOG(GREEN, "Tree with ", nodes.size(), " nodes built in ", timer.stop(), "s.");
}

void dfs(const WordVecNode* start_node, NodeVector& out)  // infix traversal
{
    if (start_node->left)
    {
        dfs(start_node->left, out);
    }
    out.push_back(start_node);
    if (start_node->right)
    {
        dfs(start_node->right, out);
    }
}

std::vector<const WordVecNode*> get_ancestors(const WordVecNode* node, bool root_first)
{
    std::vector<const WordVecNode*> out;
    while (node->parent != node)
    {
        out.push_back(node);
        node = node->parent;
    }
    out.push_back(node);
    if (root_first) std::reverse(out.begin(), out.end());
    return out;
}

const WordVecNode* get_ancestor(const WordVecNode* node, uint32 num_level_up)
{
    for (uint32 i = 0; i < num_level_up; i++)
    {
        if (node->parent == nullptr) return node;
        node = node->parent;
        if (node->parent == node) break;
    }
    return node;
}

WordVecNode* Word2Vec::build(WordVecPairIter begin, WordVecPairIter end, int depth)
{
    if (begin == end)
    {
        return nullptr;
    }
    uint32 axis = get_axis(depth);
    std::sort(begin, end, [axis](const WordVecPair& a, const WordVecPair& b) {
        return a.second[axis] < b.second[axis];
    });

    WordVecPairIter mid = begin + (end - begin) / 2;
    WordVecNode* node = new WordVecNode(*mid, depth);
    node->left = build(begin, mid, depth + 1);
    node->right = build(mid + 1, end, depth + 1);
    if (node->left) node->left->parent = node;
    if (node->right) node->right->parent = node;
    node->parent = depth == 0 ? node : node->parent;
    nodes.push_back(node);
    return node;
}

void Word2Vec::nearest(const Vec300& vec, const WordVecNode* node, NodeDist2& best, FloatT thresh,
                       uint32* count, uint32 count_threshold, uint32 maxDepthForExact)
{
    if (!node) return;
    nearest_count++;
    if (count and *count > count_threshold) return;
    uint32 axis = get_axis(node->depth);
    if (best.second < thresh) return;
    FloatT dist2 =
        node->depth < maxDepthForExact ? node->dist2(vec) : l2_dist2_pca(node->vec, vec, 50);
    if (dist2 < best.second)
    {
        best = {node, dist2};
    }
    auto *pursure = node->right, *other = node->left;
    if (node->left and vec[axis] < node->vec[axis]) std::swap(pursure, other);
    nearest(vec, pursure, best, thresh, count, count_threshold, maxDepthForExact);
    if (best.second < thresh or !other) return;

    FloatT aa_dist = (node->vec[axis] - vec[axis]) * (node->vec[axis] - vec[axis]);
    if (aa_dist < best.second)
    {
        if (count and (*count)++ > count_threshold) return;
        nearest(vec, other, best, thresh, count, count_threshold, maxDepthForExact);
    }
}

const WordVecNode* Word2Vec::operator()(const Vec300& vec, SearchOption option)
{
    NodeDist2 best{root, 3};  // embeddings are normalized so largest possible distance is 2
    uint32 count = 0;
    switch (option)
    {
        case SearchOption::FAST:
            nearest(vec, this->root, best, 1e-2, &count, 25'000, 5);
            break;
        case SearchOption::ACCURATE:
            nearest(vec, this->root, best, 1e-2, &count, 60'000, 10);
            if (best.first->cos_sim(vec) < 0.3)
            {
                nearest(vec, best.first->parent, best, 1e-3, &count, 120'000, 10);
                if (best.first->cos_sim(vec) < 0.5)
                {
                    nearest(vec, best.first->parent, best, 1e-4, nullptr, 0, 17);
                }
            }
            break;
        case SearchOption::EXACT:
            for (uint32 i = 0; i < size(); ++i)
            {
                auto dist2 = nodes[i]->dist2(vec);
                if (dist2 < best.second)
                {
                    best = {nodes[i], dist2};
                }
            }
            break;
    }
    return best.first;
}
