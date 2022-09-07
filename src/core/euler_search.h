#ifndef _SRC_CORE_EULER_SEARCH_H_
#define _SRC_CORE_EULER_SEARCH_H_
#include <cistem_config.h>

#include "../constants/cistem_numbers.h"

class GpuImage;

class EulerSearch {

  private:
    bool                  _is_symmetry_limit_set;
    RandomNumberGenerator _my_rand{pi_v<float>};

    // Brute-force search to find matching projections

  public:
    int     number_of_search_dimensions;
    int     refine_top_N;
    int     number_of_search_positions;
    int     best_parameters_to_keep;
    float   angular_step_size;
    float   max_search_x;
    float   max_search_y;
    float   phi_max;
    float   phi_start;
    float   theta_max;
    float   theta_start;
    float   psi_max;
    float   psi_step;
    float   psi_start;
    float** list_of_search_parameters;
    float** list_of_best_parameters;
    //	Kernel2D	 **kernel_index;
    //	float		 *best_values;
    //	float		 best_score;
    float        resolution_limit;
    ParameterMap parameter_map;
    bool         test_mirror;
    wxString     symmetry_symbol;

#ifdef CISTEM_PROFILING
    cistem_timer::StopWatch timer;
#else
    cistem_timer_noop::StopWatch timer;
#endif

    // Constructors & destructors
    EulerSearch( );
    ~EulerSearch( );

    EulerSearch(const EulerSearch& other_search);

    EulerSearch& operator=(const EulerSearch& t);
    EulerSearch& operator=(const EulerSearch* t);

    // Methods
    void Init(float wanted_resolution_limit, ParameterMap& wanted_parameter_map, int wanted_parameters_to_keep);
    void InitGrid(wxString wanted_symmetry_symbol, float angular_step_size, float wanted_phi_start, float wanted_theta_start, float wanted_psi_max, float wanted_psi_step, float wanted_psi_start, float wanted_resolution_limit, ParameterMap& parameter_map, int wanted_parameters_to_keep);
    void InitRandom(wxString wanted_symmetry_symbol, float wanted_psi_step, int wanted_number_of_search_positions, float wanted_resolution_limit, ParameterMap& wanted_parameter_map, int wanted_parameters_to_keep);

    // batch_size is ignored in the CPU code path and used to accelerate the inner loop over in-plane angles for the GPU search
    template <class ImageType>
    void Run(Particle& particle, Image& input_3d, ImageType* projections, int batch_size = 1);

#ifdef CISTEM_DETERMINISTIC_OUTCOME
    // WE also override in the function itself, but macro here for clarity on the default value.
    void CalculateGridSearchPositions(bool random_start_angle = false);
#else
    void                         CalculateGridSearchPositions(bool random_start_angle = true);
#endif

    // TODO: make this use the newer GetRandomEulerAngles() which evenly sample the euler sphere
    void CalculateRandomSearchPositions( );
    void SetSymmetryLimits( );
    void GetRandomEulerAngles(float& phi, float& theta, float& psi);

    //	void RotateFourier2DFromIndex(Image &image_to_rotate, Image &rotated_image, Kernel2D &kernel_index);
    //	void RotateFourier2DIndex(Image &image_to_rotate, Kernel2D &kernel_index, AnglesAndShifts &rotation_angle, float resolution_limit = 1.0, float padding_factor = 1.0);
    Kernel2D ReturnLinearInterpolatedFourierKernel2D(Image& image_to_rotate, float& x, float& y);
};

#endif