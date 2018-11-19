#include <execinfo.h>
#include <signal.h>
#include <string.h>
#include <cstring>

#include <iostream>
#include <cstdlib>
#include <stdexcept>

void my_terminate(void);

namespace {
    // invoke set_terminate as part of global constant initialization
    static const bool SET_TERMINATE = std::set_terminate(my_terminate);
}

extern struct sigaction sa;
extern struct sigaction sigact;


void segfault_sigaction(int sig, siginfo_t *info, void *c);

void abort_sigaction(int sig_num, siginfo_t * info, void * ucontext);

void set_error_handlers();


