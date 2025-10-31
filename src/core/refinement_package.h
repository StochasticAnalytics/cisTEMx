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

class RefinementPackageParticleInfo {

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

WX_DECLARE_OBJARRAY(RefinementPackageParticleInfo, ArrayOfRefinmentPackageParticleInfos);

class RefinementPackage {

  public:
    RefinementPackage( );
    ~RefinementPackage( );

    long     asset_id;
    wxString stack_filename;
    wxString name;
    int      stack_box_size;
    float    output_pixel_size;

    int number_of_classes;

    wxString symmetry;
    double   estimated_particle_size_in_angstroms;
    double   estimated_particle_weight_in_kda;
    double   lowest_resolution_of_intial_parameter_generated_3ds;

    bool stack_has_white_protein;

    int  number_of_run_refinments;
    long last_refinment_id;

    wxArrayLong references_for_next_refinement;
    wxArrayLong refinement_ids;
    wxArrayLong classification_ids;

    ArrayOfRefinmentPackageParticleInfos contained_particles;

    RefinementPackageParticleInfo ReturnParticleInfoByPositionInStack(long wanted_position_in_stack);

    long ReturnLastRefinementID( );
};

WX_DECLARE_OBJARRAY(RefinementPackage, ArrayOfRefinementPackages);
