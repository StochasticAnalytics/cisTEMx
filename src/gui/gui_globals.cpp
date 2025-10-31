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

#include "../core/gui_core_headers.h"

wxDEFINE_EVENT(RETURN_PROCESSED_IMAGE_EVT, ReturnProcessedImageEvent);
wxDEFINE_EVENT(RETURN_SHARPENING_RESULTS_EVT, ReturnSharpeningResultsEvent);

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayofColors);

ArrayofColors default_colormap;
ArrayofColors default_colorbar;
