#include "datasets.hpp"
#include "emotion_data.hpp"
#include "network_builder.hpp"
#include "nodes/loss.hpp"
#include "nodes/parameterized.hpp"
#include "signal.h"

inline void segfault_sigaction(int, siginfo_t*, void*)
{
    print_backtrace();
    exit(1);
}

int main()
{
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = segfault_sigaction;
    sigaction(SIGSEGV, &sa, nullptr);

    NetworkBuilder builder("static_data/network.nd");
    // print all nodes in the network
    for (const auto& [name, node] : builder.get_nodes())
    {
        LOG(YELLOW, name, " - ", node->shape);
    }
    return 0;
}
