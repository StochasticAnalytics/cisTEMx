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

/* class RefinementPackageParticleInfo {

  public:
    RefinementPackageParticleInfo( );
    ~RefinementPackageParticleInfo( );

    long  parent_image_id;
    long  position_in_stack;
    long  original_particle_position_asset_id;
    float x_pos;
    float y_pos;
    float pixel_size;
    float defocus_1;
    float defocus_2;
    float defocus_angle;
    float phase_shift;
    float spherical_aberration;
    float amplitude_contrast;
    float microscope_voltage;
    int   assigned_subset;
};

WX_DECLARE_OBJARRAY(RefinementPackageParticleInfo, ArrayOfRefinmentPackageParticleInfos); */

class TemplateMatchesPackage {

  public:
    TemplateMatchesPackage( );
    ~TemplateMatchesPackage( );

    long     asset_id;
    wxString starfile_filename;
    wxString name;
    long     contained_match_count;

    wxArrayLong match_template_result_ids;
};

WX_DECLARE_OBJARRAY(TemplateMatchesPackage, ArrayOfTemplateMatchesPackages);
