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

#include "core_headers.h"

RandomNumberGenerator global_random_number_generator(-1);

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(wxArrayFloat);
WX_DEFINE_OBJARRAY(wxArrayBool);
