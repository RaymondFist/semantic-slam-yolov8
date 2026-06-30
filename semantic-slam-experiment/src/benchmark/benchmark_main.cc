// ============================================================================
// Unified Benchmark Dispatcher
// Combines KITTI, EuRoC, TUM, and Ablation benchmarks into a single executable.
// Usage: semantic_slam_benchmark <mode> [args...]
//   mode: kitti | euroc | tum | ablation
// ============================================================================

#include <iostream>
#include <cstring>
#include <csignal>
#include <cstdio>

#ifdef __linux__
#include <execinfo.h>
#endif

extern int run_kitti_benchmark(int argc, char** argv);
extern int run_euroc_benchmark(int argc, char** argv);
extern int run_tum_benchmark(int argc, char* argv[]);
extern int run_ablation(int argc, char** argv);

static void crashHandler(int sig) {
    fprintf(stderr, "\n[CRASH] Signal %d (%s) received\n", sig,
            sig == SIGSEGV ? "SIGSEGV" :
            sig == SIGABRT ? "SIGABRT" :
            sig == SIGFPE  ? "SIGFPE"  : "UNKNOWN");
#ifdef __linux__
    void* buffer[32];
    int nptrs = backtrace(buffer, 32);
    char** symbols = backtrace_symbols(buffer, nptrs);
    if (symbols) {
        fprintf(stderr, "[CRASH] Backtrace (%d frames):\n", nptrs);
        for (int i = 0; i < nptrs; ++i) {
            fprintf(stderr, "  #%d %s\n", i, symbols[i]);
        }
        free(symbols);
    }
#endif
    fflush(stderr);
    // Re-raise the signal with default handler to produce core dump
    signal(sig, SIG_DFL);
    raise(sig);
}

static void printUsage() {
    std::cerr << "Usage: semantic_slam_benchmark <mode> [args...]\n\n"
              << "Modes:\n"
              << "  kitti    — KITTI odometry benchmark\n"
              << "  euroc    — EuRoC MAV benchmark\n"
              << "  tum      — TUM RGB-D benchmark\n"
              << "  ablation — Ablation study runner\n\n"
              << "Examples:\n"
              << "  semantic_slam_benchmark kitti /data/KITTI ./ORBvoc.txt ./KITTI00.yaml ./output\n"
              << "  semantic_slam_benchmark euroc /data/EuRoC ./ORBvoc.txt ./config/EuRoC.yaml MH_01_easy\n"
              << "  semantic_slam_benchmark tum /data/TUM/fr1_xyz ./ORBvoc.txt ./TUM1.yaml ./results\n"
              << "  semantic_slam_benchmark ablation /kitti /euroc /tum vocab.txt k.yaml e.yaml t.yaml\n";
}

int main(int argc, char** argv) {
    // Install crash handler for debugging segfaults
    signal(SIGSEGV, crashHandler);
    signal(SIGABRT, crashHandler);
    signal(SIGFPE,  crashHandler);

    if (argc < 2) {
        printUsage();
        return 1;
    }

    const char* mode = argv[1];

    // Shift argv to remove the first argument (the mode), so each function
    // receives arguments starting from the original argv[1] position.
    // For compatibility, we pass the shifted argv with argc-1.
    if (std::strcmp(mode, "kitti") == 0) {
        return run_kitti_benchmark(argc - 1, argv + 1);
    } else if (std::strcmp(mode, "euroc") == 0) {
        return run_euroc_benchmark(argc - 1, argv + 1);
    } else if (std::strcmp(mode, "tum") == 0) {
        return run_tum_benchmark(argc - 1, argv + 1);
    } else if (std::strcmp(mode, "ablation") == 0) {
        return run_ablation(argc - 1, argv + 1);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        printUsage();
        return 1;
    }
}