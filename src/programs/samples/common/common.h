#ifndef SRC_PROGRAMS_SAMPLES_COMMON_COMMON_H_
#define SRC_PROGRAMS_SAMPLES_COMMON_COMMON_H_

#include "helper_functions.h"
#include "embedded_test_file.h"
#include "numeric_test_file.h"

extern bool samples_tests_have_all_passed;

inline void TEST(bool result) {
    if ( ! result ) {
        samples_tests_have_all_passed = false;
    }
}
#endif
