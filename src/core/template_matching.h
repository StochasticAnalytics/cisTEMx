#include "database/typesafe_database_schema.h"

class Image;
class ImageFile;

class TemplateMatchFoundPeakInfo {
  public:
    int   peak_number;
    float x_pos;
    float y_pos;
    float psi;
    float theta;
    float phi;
    float defocus;
    float pixel_size;
    float peak_height;
    int   original_peak_number;
    int   new_peak_number;
};

WX_DECLARE_OBJARRAY(TemplateMatchFoundPeakInfo, ArrayOfTemplateMatchFoundPeakInfos);

class TemplateMatchJobResults {
  public:
    TemplateMatchJobResults( );

    // Database record for TEMPLATE_MATCH_LIST table
    template_match_list db;

    wxString job_name;
    wxString symmetry;
    float    low_res_limit;
    float    high_res_limit;
    float    out_of_plane_step;
    float    in_plane_step;
    float    defocus_search_range;
    float    defocus_step;
    float    pixel_size_search_range;
    float    pixel_size_step;
    float    mask_radius;
    float    min_peak_radius;
    float    xy_change_threshold;
    bool     exclude_above_xy_threshold;

    // Helper methods to generate individual output filenames from db.output_filename_base
    // Format: <base>_<type>_<template_match_id>_<search_id>.mrc
    wxString GetMipFilename( ) const {
        return db.output_filename_base + wxString::Format("_mip_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetScaledMipFilename( ) const {
        return db.output_filename_base + wxString::Format("_scaled_mip_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetAvgFilename( ) const {
        return db.output_filename_base + wxString::Format("_avg_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetStdFilename( ) const {
        return db.output_filename_base + wxString::Format("_std_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetPsiFilename( ) const {
        return db.output_filename_base + wxString::Format("_psi_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetThetaFilename( ) const {
        return db.output_filename_base + wxString::Format("_theta_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetPhiFilename( ) const {
        return db.output_filename_base + wxString::Format("_phi_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetDefocusFilename( ) const {
        return db.output_filename_base + wxString::Format("_defocus_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetPixelSizeFilename( ) const {
        return db.output_filename_base + wxString::Format("_pixel_size_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    wxString GetHistogramFilename( ) const {
        return db.output_filename_base + wxString::Format("_histogram_%ld_%ld.txt", db.template_match_id, db.search_id);
    }

    wxString GetProjectionResultFilename( ) const {
        return db.output_filename_base + wxString::Format("_projection_result_%ld_%ld.mrc", db.template_match_id, db.search_id);
    }

    float refinement_threshold;
    float used_threshold;
    float reference_box_size_in_angstroms;

    ArrayOfTemplateMatchFoundPeakInfos found_peaks;
    ArrayOfTemplateMatchFoundPeakInfos peak_changes;
};

WX_DECLARE_OBJARRAY(TemplateMatchJobResults, ArrayOfTemplateMatchJobResults);
