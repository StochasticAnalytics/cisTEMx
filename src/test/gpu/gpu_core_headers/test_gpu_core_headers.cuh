/*
 * Original Copyright (c) 2017, Howard Hughes Medical Institute
 * Licensed under Janelia Research Campus Software License 1.2
 * See license_details/LICENSE-JANELIA.txt
 *
 * Modifications Copyright (c) 2025, Stochastic Analytics, LLC
 * Modifications licensed under MPL 2.0 for academic use; 
 * commercial license required for commercial use.
 * See LICENSE.md for details.
 */

#ifndef _src_test_gpu_test_gpu_core_headers_h
#define _src_test_gpu_test_gpu_core_headers_h

void    test_complex_add(Complex* a, Complex* b, Complex* output);
void    test_complex_scale(Complex* a, float output);
Complex test_complex_scale(Complex& a, float output);

#endif