#ifndef _SRC_CORE_DATABASE_TYPESAFE_DATABASE_SCHEMA_H_
#define _SRC_CORE_DATABASE_TYPESAFE_DATABASE_SCHEMA_H_

#include <string>
#include <wx/string.h>
#include "typesafe_database_helpers.h"

// Schema type that carries column names as a constexpr array
template <typename T, std::size_t N>
struct TableSchema {
    using table_type                          = T;
    static constexpr std::size_t column_count = N;
    std::array<const char*, N>   columns;

    constexpr TableSchema(std::array<const char*, N> cols) : columns(cols) {}

    // Compile-time membership check
    constexpr bool contains(const char* field) const {
        for ( std::size_t i = 0; i < N; ++i ) {
            if ( str_equal(columns[i], field) )
                return true;
        }
        return false;
    }
};

// Type-safe database table structures
struct template_match_list {
    static constexpr const char* table_name = "TEMPLATE_MATCH_LIST";

    long     template_match_id; // p - PRIMARY KEY
    long     datetime_of_run; // l
    long     elapsed_time_seconds; // l
    long     search_id; // l
    int      search_type_code; // i
    long     parent_search_id; // i
    long     image_asset_id; // l
    long     reference_volume_asset_id; // i
    int      is_active; // i
    wxString output_filename_base; // t - TEXT
    float    pixel_size; // r - REAL
    float    voltage; // r
    float    spherical_aberration; // r
    float    amplitude_contrast; // r
    float    defocus1; // r
    float    defocus2; // r
    float    defocus_angle; // r
    float    phase_shift; // r
    float    future_float_1; // r
    float    future_float_2; // r
};

// Create the schema for template_match_list
using template_match_list_schema = TableSchema<template_match_list, 20>;
constexpr template_match_list_schema template_match_list_columns{{"template_match_id",
                                                                  "datetime_of_run",
                                                                  "elapsed_time_seconds",
                                                                  "search_id",
                                                                  "search_type_code",
                                                                  "parent_search_id",
                                                                  "image_asset_id",
                                                                  "reference_volume_asset_id",
                                                                  "is_active",
                                                                  "output_filename_base",
                                                                  "pixel_size",
                                                                  "voltage",
                                                                  "spherical_aberration",
                                                                  "amplitude_contrast",
                                                                  "defocus1",
                                                                  "defocus2",
                                                                  "defocus_angle",
                                                                  "phase_shift",
                                                                  "future_float_1",
                                                                  "future_float_2"}};

// Compile-time verification that schema column count is correct
// Note: std::array constructor already enforces that exactly 20 elements are provided above
static_assert(template_match_list_schema::column_count == 20,
              "template_match_list schema column count mismatch - update both the schema and this assertion when modifying the table");

#endif // _SRC_CORE_DATABASE_TYPESAFE_DATABASE_SCHEMA_H_