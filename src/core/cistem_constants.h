#ifndef _src_core_cistem_constants_h_
#define _src_core_cistem_constants_h_

#include <array>

// Place system wide constants and enums here. Gradually, we would like to replace the many defines.
namespace cistem {

// The default border to exclude when choosing peaks, e.g. in match_template, refine_template, prepare_stack_matchtemplate, make_template_result.
constexpr const int fraction_of_box_size_to_exclude_for_border = 4;
constexpr const int maximum_number_of_detections               = 1000;

/*
    SCOPED ENUMS:
        Rather than specifying a scoped enum as enum class, we use the following technique to define scoped enums while
        avoiding the need for static_cast<type>(enum) anywhere we want to do an assignment or comparison.      
*/

// To ensure data base type parity, force int type (even though this should be the default).
namespace job_type {
enum Enum : int {
    // TODO: extend this to remove other existing job_type defines.
    template_match_full_search,
    template_match_refinement
};

} // namespace job_type

namespace workflow {
enum Enum : int { single_particle,
                  template_matching };
} // namespace workflow

namespace PCOS_image_type {
enum Enum : int { reference_volume_t,
                  particle_image_t,
                  ctf_image_t,
                  beamtilt_image_t };
}

namespace gpu {

constexpr int warp_size             = 32;
constexpr int min_threads_per_block = warp_size;
constexpr int max_threads_per_block = 1024;

// Currently we just support up to 3d tensors to match the Image class
constexpr int max_tensor_manager_dimensions = 3;
constexpr int max_tensor_manager_tensors    = 4;

namespace tensor_op {
enum Enum : int {
    reduction,
    contraction,
    binary,
    ternary,
};
} // namespace tensor_op

namespace tensor_id {
enum Enum : int {
    A,
    B,
    C,
    D,
};

// must match the above enum tensor_id
constexpr std::array<char, 4> tensor_names = {'A', 'B', 'C', 'D'};
} // namespace tensor_id

} // namespace gpu

} // namespace cistem

#endif
