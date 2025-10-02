#ifndef _SRC_CORE_TENSOR_DEBUG_UTILS_H_
#define _SRC_CORE_TENSOR_DEBUG_UTILS_H_

/**
 * @file debug_utils.h
 * @brief Lightweight debug utilities for Tensor system
 *
 * Provides compile-time-optimized debug assertions using templates and constexpr.
 * Unlike macro-based assertions, these avoid evaluating format strings in release builds.
 */

#include <cstdio>
#include <cstdlib>

namespace cistem {
namespace tensor {

#ifdef DEBUG
/**
 * @brief Debug assertion that only evaluates message in debug builds
 *
 * Uses if constexpr to completely eliminate formatting overhead in release builds.
 *
 * @param condition Condition to check
 * @param format Printf-style format string
 * @param args Format arguments
 */
template <typename... Args>
inline void DebugAssert(bool condition, const char* file, int line,
                        const char* function, const char* format, Args&&... args) {
    if ( ! condition ) {
        std::fprintf(stderr, "\n");
        std::fprintf(stderr, format, std::forward<Args>(args)...);
        std::fprintf(stderr, "\nFailed Assert at %s:%d\n%s\n", file, line, function);
        std::abort( );
    }
}

// Overload for no-argument messages (avoid va_args issues)
inline void DebugAssert(bool condition, const char* file, int line,
                        const char* function, const char* message) {
    if ( ! condition ) {
        std::fprintf(stderr, "\n%s\nFailed Assert at %s:%d\n%s\n",
                     message, file, line, function);
        std::abort( );
    }
}

#define TENSOR_DEBUG_ASSERT(cond, msg, ...) \
    cistem::tensor::DebugAssert((cond), __FILE__, __LINE__, __PRETTY_FUNCTION__, msg, ##__VA_ARGS__)

#else
// Release build: completely eliminate assertion code
#define TENSOR_DEBUG_ASSERT(cond, msg, ...) ((void)0)
#endif

} // namespace tensor
} // namespace cistem

#endif // _SRC_CORE_TENSOR_DEBUG_UTILS_H_
