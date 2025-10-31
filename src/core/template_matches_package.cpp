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
#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
//WX_DEFINE_OBJARRAY(ArrayOfRefinmentPackageParticleInfos);
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchesPackages);

//WX_DEFINE_OBJARRAY(ArrayofSingleRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofMultiClassRefinementResults);
//WX_DEFINE_OBJARRAY(ArrayofWholeRefinementResults);

/* RefinementPackageParticleInfo::RefinementPackageParticleInfo( ) {
    parent_image_id                     = -1;
    original_particle_position_asset_id = -1;
    position_in_stack                   = -1;
    x_pos                               = -1;
    y_pos                               = -1;
    pixel_size                          = -1;
    defocus_1                           = 0;
    defocus_2                           = 0;
    defocus_angle                       = 0;
    phase_shift                         = 0;
    spherical_aberration                = 0;
    microscope_voltage                  = 0;
    amplitude_contrast                  = 0.07;
    assigned_subset                     = -1;
}

RefinementPackageParticleInfo::~RefinementPackageParticleInfo( ) {
} */

TemplateMatchesPackage::TemplateMatchesPackage( ) {
    asset_id              = -1;
    starfile_filename     = "";
    name                  = "";
    contained_match_count = 0;
}

TemplateMatchesPackage::~TemplateMatchesPackage( ) {
}

/* long TemplateMatchesPackage::ReturnLastRefinementID( ) {
    return refinement_ids.Item(refinement_ids.GetCount( ) - 1);
} */

/* RefinementPackageParticleInfo RefinementPackage::ReturnParticleInfoByPositionInStack(long wanted_position_in_stack) {
    for ( long counter = wanted_position_in_stack - 1; counter < contained_particles.GetCount( ); counter++ ) {
        if ( contained_particles.Item(counter).position_in_stack == wanted_position_in_stack )
            return contained_particles.Item(counter);
    }

    for ( long counter = 0; counter < wanted_position_in_stack; counter++ ) {
        if ( contained_particles.Item(counter).position_in_stack == wanted_position_in_stack )
            return contained_particles.Item(counter);
    }

    MyDebugPrintWithDetails("Shouldn't get here, means i didn't find the particle");
    DEBUG_ABORT;
} */
