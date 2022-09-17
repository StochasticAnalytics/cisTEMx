#include "../constants/constants.h"

class ElectronDose {
  private:

  public:
    float acceleration_voltage;

    float voltage_scaling_factor;

    float pixel_size;

    ElectronDose( );
    ElectronDose(float wanted_acceleration_voltage, float wanted_pixel_size);

    void  Init(float wanted_acceleration_voltage, float wanted_pixel_size);
    float ReturnCriticalDose(float spatial_frequency);
    float ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose);
    float ReturnCummulativeDoseFilter(float dose_at_start_of_exposure, float dose_at_end_of_exosure, float critical_dose);
    void  CalculateCummulativeDoseFilterAs1DArray(Image* ref_image, float* filter_array, float dose_start, float dose_finish);

    // Image defin in electron_dose.cpp, GpuImage in src/gpu/core_extensions/electron_dose_gpu
    template <class ImageType>
    void CalculateDoseFilterAs1DArray(ImageType* ref_image, float* filter_array, float dose_start, float dose_finish, int n_images = 1, bool restore_power = false);
};

inline float ElectronDose::ReturnCriticalDose(float spatial_frequency) {
    return (cistem::electron_dose::critical_dose_a * powf(spatial_frequency, cistem::electron_dose::reduced_critical_dose_b) + cistem::electron_dose::critical_dose_c) * voltage_scaling_factor;
}

inline float ElectronDose::ReturnDoseFilter(float dose_at_end_of_frame, float critical_dose) {
    return expf((-0.5 * dose_at_end_of_frame) / critical_dose);
}

inline float ElectronDose::ReturnCummulativeDoseFilter(float dose_at_start_of_exposure, float dose_at_end_of_exosure, float critical_dose) {
    // The integrated exposure. Included in particular for the matched filter.
    // Calculated on Wolfram Alpha = integrate exp[ -0.5 * (x/a) ] from x=0 to x=t
    return 2.0f * critical_dose * (exp((-0.5 * dose_at_start_of_exposure) / critical_dose) - exp((-0.5 * dose_at_end_of_exosure) / critical_dose)) / dose_at_end_of_exosure;
}
