#include "network_graph.hpp"
#include "signal.h"

void test_empty_kernel();

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

    return 0;
}
