#include "../../core/core_headers.h"

class
        NikoTestApp : public MyApp {

  public:
    bool DoCalculation( );
    void DoInteractiveUserInput( );

  private:
};

IMPLEMENT_APP(NikoTestApp)

// override the DoInteractiveUserInput

void NikoTestApp::DoInteractiveUserInput( ) {
}

// override the do calculation method which will be what is actually run..

bool NikoTestApp::DoCalculation( ) {

    std::string filename = "a06_00321_UnderDefocus1.4um_frameImage_frames.mrc";

    ImageFile input_file(filename);

    Curve average;
    Curve number_of_values;

    float pixel_size = 0.822;
    Image aligned_frames;
    average.SetupXAxis(0.0, sqrt(2.0) * 0.5, input_file.ReturnXSize( ));

    number_of_values = average;

    // Read in the full movie stack for a 3d transofrm.
    aligned_frames.ReadSlices(&input_file, 1, input_file.ReturnNumberOfSlices( ));
    aligned_frames.ForwardFFT(false);
    wxPrintf("Size of aligned frames is %i %i %i\n", aligned_frames.logical_x_dimension, aligned_frames.logical_y_dimension, aligned_frames.logical_z_dimension);
    wxPrintf("Is in real space %i\n", aligned_frames.is_in_real_space);

    Image slice_from_3d;
    slice_from_3d.Allocate(aligned_frames.logical_x_dimension, aligned_frames.logical_y_dimension, 1, false);
    int stride = slice_from_3d.real_memory_allocated / 2;
    for ( int i = 0; i < input_file.ReturnNumberOfSlices( ); i++ ) {
        slice_from_3d.real_values    = (float*)&aligned_frames.complex_values[i * stride];
        slice_from_3d.complex_values = &aligned_frames.complex_values[i * stride];
        slice_from_3d.Compute1DRotationalAverage(average, number_of_values);
        average.SquareRoot( );
        std::string output_name = "rot_ps_" + std::to_string(i) + ".txt";
        average.WriteToFile(output_name);
    }

    slice_from_3d.real_values = nullptr;

    slice_from_3d.complex_values = nullptr;

    return true;
}
