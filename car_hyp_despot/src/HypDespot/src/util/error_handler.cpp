/*
 * error_handller.cpp
 *
 *  Created on: 5 Oct, 2018
 *      Author: panpan
 */

// #include <libunwind.h>

#include <despot/util/error_handler.h>


//Sigfault handling
struct sigaction sa;
struct sigaction sigact;
struct sigaction fpeact;


static const char *gregs[] = {
	"GS",
	"FS",
	"ES",
	"DS",
	"EDI",
	"ESI",
	"EBP",
	"ESP",
	"EBX",
	"EDX",
	"ECX",
	"EAX",
	"TRAPNO",
	"ERR",
	"EIP",
	"CS",
	"EFL",
	"UESP",
	"SS"
};

void segfault_sigaction(int sig, siginfo_t *info, void *c)
{
	ucontext_t *context = (ucontext_t *)c;

	fprintf(stderr,
		"si_signo:  %d\n"
		"si_code:   %s\n"
		"si_errno:  %d\n"
		"si_pid:    %d\n"
		"si_uid:    %d\n"
		"si_addr:   %p\n"
		"si_status: %d\n"
		"si_band:   %ld\n",
		info->si_signo,
		(info->si_code == SEGV_MAPERR) ? "SEGV_MAPERR" : "SEGV_ACCERR",
		info->si_errno, info->si_pid, info->si_uid, info->si_addr,
		info->si_status, info->si_band
	);

	fprintf(stderr,
		"uc_flags:  0x%x\n"
		"ss_sp:     %p\n"
		"ss_size:   %d\n"
		"ss_flags:  0x%X\n",
		context->uc_flags,
		context->uc_stack.ss_sp,
		context->uc_stack.ss_size,
		context->uc_stack.ss_flags
	);

	fprintf(stderr, "General Registers:\n");
	for(int i = 0; i < 19; i++)
		fprintf(stderr, "\t%7s: 0x%x\n", gregs[i], context->uc_mcontext.gregs[i]);

    raise(SIGABRT);

	// std::terminate();
	// exit(-1);
}


void backtrace();


void abort_sigaction(int sig_num, siginfo_t * info, void * ucontext) {

	ucontext_t * uc = (ucontext_t *)ucontext;

    // Get the address at the time the signal was raised from the EIP (x86)
    void * caller_address = (void *) uc->uc_mcontext.gregs[14];

    std::cerr << "signal " << sig_num
              << " (" << strsignal(sig_num) << "), address is "
              << info->si_addr << " from "
              << caller_address << std::endl;

    void * array[50];
    int size = backtrace(array, 50);

    std::cerr << __FUNCTION__ << " backtrace returned "
              << size << " frames\n\n";

    // overwrite sigaction with caller's address
    array[1] = caller_address;

    char ** messages = backtrace_symbols(array, size);

    // skip first stack frame (points here)
    for (int i = 1; i < size && messages != NULL; ++i) {
        std::cerr << "[bt]: (" << i << ") " << messages[i] << std::endl;
    }
    std::cerr << std::endl;

    free(messages);

    // backtrace();

    exit(EXIT_FAILURE);
}


// Call this function to get a backtrace.
// void backtrace() {
//   printf("\n ======================= back tracing ==========================\n");

//   unw_cursor_t cursor;
//   unw_context_t context;

//   // Initialize cursor to current frame for local unwinding.
//   unw_getcontext(&context);
//   unw_init_local(&cursor, &context);

//   // Unwind frames one by one, going up the frame stack.
//   while (unw_step(&cursor) > 0) {
//     unw_word_t offset, pc;
//     unw_get_reg(&cursor, UNW_REG_IP, &pc);
//     if (pc == 0) {
//       break;
//     }
//     printf("0x%lx:", pc);

//     char sym[256];
//     if (unw_get_proc_name(&cursor, sym, sizeof(sym), &offset) == 0) {
//       printf(" (%s+0x%lx)\n", sym, offset);
//     } else {
//       printf(" -- error: unable to obtain symbol name for this frame\n");
//     }
//   }
//   printf(" ======================= back tracing ==========================\n");

// }

char** print_exeption() {
	void* array[50];
	int size = backtrace(array, 50);
	std::cerr << __FUNCTION__ << " backtrace returned " << size
			<< " frames\n\n";
	char** messages = backtrace_symbols(array, size);
	for (int i = 0; i < size && messages != NULL; ++i) {
		std::cerr << "[bt]: (" << i << ") " << messages[i] << std::endl;
	}
	std::cerr << std::endl;
	return messages;
}

void my_terminate() {

    // backtrace();

    static bool tried_throw = false;
    printf(" my_terminate \n");

    try {
        // try once to re-throw currently active exception
        if (!tried_throw++) throw;
    }
    catch (const std::exception &e) {
        std::cerr << __FUNCTION__ << " caught unhandled exception. what(): "
                  << e.what() << std::endl;
    }
    catch (...) {

        std::cerr << __FUNCTION__ << " caught unknown/unhandled exception."
                  << std::endl;
    }

    char** messages = print_exeption();
    free(messages);

    // abort();
    raise(SIGABRT);
}


void set_error_handlers() {
	std::memset((void*) (&sa), 0, sizeof(struct sigaction));
	sigemptyset(&sa.sa_mask);
	sa.sa_sigaction = abort_sigaction;
	sa.sa_flags = SA_SIGINFO;
	sigaction(SIGSEGV, &sa, NULL);

	std::memset((void*) (&sigact), 0, sizeof(struct sigaction));
	sigemptyset(&sigact.sa_mask);
	sigact.sa_sigaction = abort_sigaction;
	sigact.sa_flags = SA_RESTART | SA_SIGINFO;
	sigaction(SIGABRT, &sigact, NULL);

	std::memset((void*) (&fpeact), 0, sizeof(struct sigaction));
	sigemptyset(&fpeact.sa_mask);
	fpeact.sa_sigaction = abort_sigaction;
	fpeact.sa_flags = SA_RESTART | SA_SIGINFO;
	sigaction(SIGFPE, &fpeact, NULL);


	std::set_unexpected(my_terminate);
}
