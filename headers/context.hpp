/*
 * Author: Ishwar Kulkarni
 * This file is distributed under the MIT license.
 * See: https://mit-license.org
 */

#ifndef CONTEXT_HPP
#define CONTEXT_HPP

#include <chrono>
#include <map>
#include <random>
#include <string>
#include <vector>

inline std::string get_readable_id()
{
    static std::vector<std::string> adj1 = {"Happy",  "Sad",     "Angry", "Excited", "Bored",
                                            "Hungry", "Thirsty", "Tired", "Sleepy",  "Sick",
                                            "Happy",  "Sad",     "Angry", "Excited", "Bored",
                                            "Hungry", "Thirsty", "Tired", "Sleepy",  "Sick"};
    static std::vector<std::string> adj2 = {
        "Red",   "Blue", "Green", "Yellow", "Purple", "Orange",  "Pink", "Brown",     "Black",
        "White", "Gray", "Gold",  "Silver", "Cyan",   "Magenta", "Teal", "Turquoise", "Maroon"};
    static std::vector<std::string> animal = {"Dog",    "Cat",   "Tiger",   "Lion",    "Viper",
                                              "Lizard", "Bear",  "Wolf",    "Fox",     "Rabbit",
                                              "Panda",  "Koala", "Penguin", "Dolphin", "Whale",
                                              "Shark",  "Eagle", "Pigeon",  "Duck",    "Goose"};

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    char ddmmyy[10];
    std::strftime(ddmmyy, sizeof(ddmmyy), "%d%m%y", std::localtime(&now_time));
    // random choice of adj1, adj2, animal:
    std::random_device rd;
    std::mt19937 gen(rd());
    using dist = std::uniform_int_distribution<>;
    dist dis(0, adj1.size() - 1);
    std::string adj1_choice = adj1[dis(gen)];
    dist dis2(0, adj2.size() - 1);
    std::string adj2_choice = adj2[dis2(gen)];
    dist dis3(0, animal.size() - 1);
    std::string animal_choice = animal[dis3(gen)];
    return adj1_choice + "-" + adj2_choice + "-" + animal_choice + "-" + ddmmyy;
}

class Context
{
 public:
    const std::string session_id;

    static Context& get()
    {
        static Context ctx;
        return ctx;
    }

    unsigned forward_pass() { return forward_pass_count++; }

    unsigned backward_pass() { return backward_pass_count++; }

    unsigned weight_update() { return weight_update_count++; }

    unsigned get_forward_pass_count() { return forward_pass_count; }

    unsigned get_backward_pass_count() { return backward_pass_count; }

    unsigned get_weight_update_count() { return weight_update_count; }

    static void reset()
    {
        get().depth = 0;
        get().epoch = 0;
        get().forward_pass_count = 0;
        get().backward_pass_count = 0;
        get().weight_update_count = 0;
    }

    std::vector<std::string> node_names;

 private:
    Context() : session_id(get_readable_id()) {}
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    unsigned depth = 0;
    unsigned epoch = 0;
    unsigned forward_pass_count = 0;
    unsigned backward_pass_count = 0;
    unsigned weight_update_count = 0;
};

#endif  // CONTEXT_HPP
