#include "core_headers.h"

#include <wx/arrimpl.cpp> // this is a magic incantation which must be done!
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchFoundPeakInfos);
WX_DEFINE_OBJARRAY(ArrayOfTemplateMatchJobResults);

TemplateMatchJobResults::TemplateMatchJobResults( ) {
    job_name                     = "";
    db.search_type_code          = -1;
    db.parent_search_id          = -1;
    db.search_id                 = -1;
    db.template_match_id         = -1;
    db.datetime_of_run           = 0;
    db.image_asset_id            = -1;
    db.reference_volume_asset_id = -1;
    symmetry                     = "C1";
    db.pixel_size                = 0.0f;
    db.voltage                   = 0.0f;
    db.spherical_aberration      = 0.0f;
    db.amplitude_contrast        = 0.0f;
    db.defocus1                  = 0.0f;
    db.defocus2                  = 0.0f;
    db.defocus_angle             = 0.0f;
    db.phase_shift               = 0.0f;
    low_res_limit                = 0.0f;
    high_res_limit               = 0.0f;
    out_of_plane_step            = 0.0f;
    in_plane_step                = 0.0f;
    defocus_search_range         = 0.0f;
    defocus_step                 = 0.0f;
    pixel_size_search_range      = 0.0f;
    pixel_size_step              = 0.0f;
    mask_radius                  = 0.0f;
    min_peak_radius              = 0.0f;
    xy_change_threshold          = 0.0f;
    exclude_above_xy_threshold   = false;

    db.output_filename_base = "";
    db.elapsed_time_seconds = 0;

    refinement_threshold            = 0.0f;
    used_threshold                  = 0.0f;
    reference_box_size_in_angstroms = 0;
}
