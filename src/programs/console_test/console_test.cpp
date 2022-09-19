#include <wx/wx.h>
#include <wx/app.h>
#include <wx/cmdline.h>
#include <cstdio>
#include <unordered_map>
#include "wx/socket.h"

#ifdef ENABLEGPU
#warning "GPU enabled in refine3d"
#include "../../gpu/gpu_core_headers.h"
#include "../../gpu/GpuImage.h"
#else
#include "../../core/core_headers.h"
#endif

#ifdef _CISTEM_MODULES
import hello;
import hello2;
#endif

#ifdef ENABLE_FastFFT
#include <FastFFT.h>
#endif

// embedded images..

#include "hiv_image_80x80x1.cpp"
#include "hiv_images_shift_noise_80x80x10.cpp"
#include "sine_128x128x1.cpp"

#define PrintResult(result) PrintResultWorker(result, __LINE__);
#define FailTest                                \
    {                                           \
        if ( test_has_passed == true )          \
            PrintResultWorker(false, __LINE__); \
        test_has_passed = false;                \
    }

// TODO //
// TEST 3D's
// ODD Sized IMAGES
// NON SQUARE IMAGES

void print2DArray(Image& image) {
    int width           = 10;
    int padding_counter = 1;
    wxPrintf("\n");
    for ( int x = -1; x < image.logical_x_dimension + image.padding_jump_value; x++ ) {
        wxPrintf("[%-4d]   ", x);
        for ( int y = 0; y < image.logical_y_dimension; y++ ) {
            if ( x == -1 ) {
                wxPrintf("[%*.3d]", width, y);
            }
            else if ( x < image.logical_x_dimension ) {
                wxPrintf("%*.3f", width, image.real_values[image.ReturnReal1DAddressFromPhysicalCoord(x, y, 0)]);
            }
            else {
                wxPrintf("%*.3f", width, image.real_values[padding_counter + image.ReturnReal1DAddressFromPhysicalCoord(x - padding_counter, y, 0)]);
            }
        }
        if ( x >= image.logical_x_dimension )
            padding_counter++;
        wxPrintf("\n");
    }
    wxPrintf("\n");
}

class MyTestApp : public MyApp {
    wxString hiv_image_80x80x1_filename;
    wxString hiv_images_80x80x10_filename;
    wxString sine_wave_128x128x1_filename;
    wxString numeric_text_filename;
    wxString temp_directory;

  public:
    // We need DoCalculation so we can have a bool return type for automated testing and a noop DoInteractiveUserInput to allow it to run from the console.
    bool DoCalculation( );
    void DoInteractiveUserInput( );

    bool test_has_passed;
    bool all_tests_have_passed;

    /*
      The tests build in complexity, such that the validity of anyone one test may depend on any number of the others.
      Rather than rely on the tests being run in any given order, we instead, at the time the test is written,  
      prescribe any functions used in that test should be checked against the list of tests that have passed.

      When BeginTest() is called, the string of the test name is added to "test_results" and set to false, and current_test_name is set to test_name.
      When EndTest() is called, the current_test_name is set to "test_has_passed".

      To check the results, a list of strings is passed to the function CheckDependencies( { "Image::ClipIntoFourier", "Test Empirical Distribution" } ).
      If the string test name is not found indicating a typo or the test is not run, or the value is false, the test fails.
    */
    std::unordered_map<std::string, bool> test_results;
    bool                                  CheckDependencies(std::initializer_list<std::string> list);
    std::string                           current_test_name;

    void TestMRCFunctions( );
    void TestAssignmentOperatorsAndFunctions( );
    void TestFFTFunctions( );
    void TestScalingAndSizingFunctions( );
    void TestFilterFunctions( );
    void TestAlignmentFunctions( );
    void TestImageArithmeticFunctions( );
    void TestSpectrumBoxConvolution( );
    void TestImageLoopingAndAddressing( );
    void TestNumericTextFiles( );
    void TestClipIntoFourier( );
    void TestMaskCentralCross( );
    void TestStarToBinaryFileConversion( );
    void TestElectronExposureFilter( );
#ifdef ENABLEGPU
    void TestElectronExposureFilterGPU( );
#endif
    void TestEmpiricalDistribution( );
    void TestSumOfSquaresFourierAndFFTNormalization( );
    void TestRandomVariableFunctions( );
    void TestIntegerShifts( );
    void TestDatabase( );

    void BeginTest(const char* test_name);
    void EndTest( );
    void PrintTitle(const char* title);
    void PrintResultWorker(bool passed, int line);
    void WriteEmbeddedFiles( );
    void WriteEmbeddedArray(const char* filename, const unsigned char* array, long length);
    void WriteNumericTextFile(const char* filename);
    void WriteDatabase(const char* dir, const char* filename);

#ifdef _CISTEM_MODULES
    void TestModules( );
#endif

#ifdef ENABLE_FastFFT
    void TestFastFFT( );
#endif
};

IMPLEMENT_APP(MyTestApp)

void MyTestApp::DoInteractiveUserInput( ) {
    // noop
}

bool MyTestApp::DoCalculation( ) {
    wxPrintf("\n\n\n     **   ");
    if ( OutputIsAtTerminal( ) == true )
        wxPrintf(ANSI_UNDERLINE "ProjectX Library Tester" ANSI_UNDERLINE_OFF);
    else
        wxPrintf("ProjectX Library Tester");
    wxPrintf("   **\n");

    //wxPrintf("")

    WriteEmbeddedFiles( );
    wxPrintf("\n");

    // Do tests..

    //PrintTitle("Basic I/O Functions");

    all_tests_have_passed = true;

    TestMRCFunctions( );
    TestAssignmentOperatorsAndFunctions( );
    TestImageArithmeticFunctions( );
    TestFFTFunctions( );
    TestScalingAndSizingFunctions( );
    TestFilterFunctions( );
    TestAlignmentFunctions( );
    TestSpectrumBoxConvolution( );
    TestImageLoopingAndAddressing( );
    TestNumericTextFiles( );
    TestClipIntoFourier( );
    TestMaskCentralCross( );
    TestStarToBinaryFileConversion( );
    TestElectronExposureFilter( );
#ifdef ENABLEGPU
    TestElectronExposureFilterGPU( );
#endif
    TestDatabase( );
    TestEmpiricalDistribution( );
    TestSumOfSquaresFourierAndFFTNormalization( );
    TestRandomVariableFunctions( );
    TestIntegerShifts( );

#ifdef _CISTEM_MODULES
    TestModules( );
#endif

#ifdef ENABLE_FastFFT
    TestFastFFT( );
#endif

    wxPrintf("\n\n\n");

    if ( ! all_tests_have_passed )
        std::exit(-1);
    else
        return 0;
}

void MyTestApp::TestAssignmentOperatorsAndFunctions( ) {

    BeginTest("Memory Assignment Ops and Funcs");

    // Test for even and odd sized square
    const int   wanted_size = 16;
    const float test_value  = 1.234f;

    Image ref_image[2];

    ref_image[0].Allocate(wanted_size, wanted_size, 1);
    ref_image[1].Allocate(wanted_size + 1, wanted_size + 1, 1);

    // Set the initial values of the reference images, that we will use to check the results of the assignment ops.
    for ( int pixel_counter = 0; pixel_counter < ref_image[0].real_memory_allocated; pixel_counter++ ) {
        ref_image[0].real_values[pixel_counter] = test_value;
    }

    for ( int pixel_counter = 0; pixel_counter < ref_image[1].real_memory_allocated; pixel_counter++ ) {
        ref_image[1].real_values[pixel_counter] = test_value;
    }

    // First test the assignment operator Image = *Image under several conditions.

    // Condition 1: test image is not allogated.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image = &ref_image[iTest];
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Condition 2: test image is allocated but not the same size.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image.Allocate(wanted_size + 3, wanted_size + 3, 1);
        test_image = &ref_image[iTest];
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Image::CopyFrom should just use the underlying method tested.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image.CopyFrom(&ref_image[iTest]);
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Assignment by reference should also then just call the pointer based assignment.
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image = ref_image[iTest]; // This line is different (i.e. assign by reference.)
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;
        for ( int pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
            if ( test_image.real_values[pixel_counter] != ref_image[iTest].real_values[pixel_counter] )
                FailTest;
        }
    }

    // Finally we check the Image::Consume method, which differs from the above test in that the resulting pointers should have the same address
    for ( int iTest = 0; iTest < 2; iTest++ ) {
        Image test_image;
        test_image.Consume(&ref_image[iTest]); // This line is different (i.e. assign by reference.)
        // The memory should be in different places and the values of test should be equal.
        if ( test_image.real_memory_allocated != ref_image[iTest].real_memory_allocated )
            FailTest;
        if ( &test_image.real_values == &ref_image[iTest].real_values )
            FailTest;

        // Because the data array is "stolen" we obvi cannot compare to the reference as this would give a segfault.
    }

    EndTest( );
}

// A partial test for the ClipInto method, specifically when clipping a Fourier transform into
// a larger volume
void MyTestApp::TestClipIntoFourier( ) {
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT", "Memory Assignment Ops and Funcs", "Image::SetToConstant"});

    BeginTest("Image::ClipIntoFourier");

    Image       test_image_original;
    Image       test_image;
    Image       test_image_ampl;
    Image       big_image;
    Image       big_image_ampl;
    int         address;
    int         i, j, k;
    int         i_logi, j_logi, k_logi;
    const float error_tolerance = 0.0001;
    const bool  write_files_out = false;

    // Create a test image
    test_image.Allocate(4, 4, 4, true);
    address = 0;
    for ( k = 0; k < 4; k++ ) {
        for ( j = 0; j < 4; j++ ) {
            for ( i = 0; i < 4; i++ ) {
                test_image.real_values[address] = float(address);
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }
    global_random_number_generator.SetSeed(0);
    test_image.AddGaussianNoise(100.0); // TODO: Swap this for the STD functions and note that dependency.

    // Keep a copy
    test_image_original.CopyFrom(&test_image);

    // Write test image to disk
    if ( write_files_out )
        test_image.QuickAndDirtyWriteSlices("dbg_start.mrc", 1, 4);

    // Clip into a larger image when in Fourier space
    big_image.Allocate(8, 8, 8, false);
    test_image.ForwardFFT( );
    test_image.ComputeAmplitudeSpectrum(&test_image_ampl);
    if ( write_files_out )
        test_image_ampl.QuickAndDirtyWriteSlices("dbg_start_ampl.mrc", 1, 4);
    test_image.ClipInto(&big_image);
    big_image.ComputeAmplitudeSpectrum(&big_image_ampl);
    if ( write_files_out )
        big_image_ampl.QuickAndDirtyWriteSlices("dbg_big_ampl.mrc", 1, 8);

    //wxPrintf("%f %f %f %f\n",cabsf(test_image.complex_values[2]),cabsf(big_image.complex_values[2]),test_image_ampl.real_values[2], big_image_ampl.real_values[2]);

    // Do a few checks of the pixel amplitudes
    if ( abs(abs(test_image.complex_values[0]) - abs(big_image.complex_values[0])) > error_tolerance )
        FailTest;
    if ( abs(abs(test_image.complex_values[1]) - abs(big_image.complex_values[1])) > error_tolerance )
        FailTest;
    if ( abs(abs(test_image.complex_values[2]) - abs(big_image.complex_values[2])) > error_tolerance )
        FailTest;

    for ( k_logi = -2; k_logi <= 2; k_logi++ ) {
        for ( j_logi = -2; j_logi <= 2; j_logi++ ) {
            for ( i_logi = 0; i_logi <= 2; i_logi++ ) {
                if ( big_image.complex_values[big_image.ReturnFourier1DAddressFromLogicalCoord(i_logi, j_logi, k_logi)] == 0.0f ) {
                    wxPrintf("\nComplex pixel with logical coords %i %i %i was not set!\n", i_logi, j_logi, k_logi);
                    FailTest;
                }
            }
        }
    }

    // Clip back into smaller image - should get back to where we started
    big_image.BackwardFFT( );
    if ( write_files_out )
        big_image.QuickAndDirtyWriteSlices("dbg_big_real_space.mrc", 1, 8);
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );

    // Write output image to disk
    if ( write_files_out )
        test_image.QuickAndDirtyWriteSlices("dbg_finish.mrc", 1, 4);

    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 4; k++ ) {
        for ( j = 0; j < 4; j++ ) {
            for ( i = 0; i < 4; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    wxPrintf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
                    FailTest;
                }
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }

    /*
   *  Now test with odd dimensions into odd dimensions
   */
    test_image.Allocate(5, 5, 5, true);
    test_image.SetToConstant(0.0);
    test_image.AddGaussianNoise(100.0);
    test_image_original.CopyFrom(&test_image);
    test_image.ForwardFFT( );
    big_image.Allocate(9, 9, 9, false);
    test_image.ClipInto(&big_image);
    big_image.BackwardFFT( );
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );
    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 5; k++ ) {
        for ( j = 0; j < 5; j++ ) {
            for ( i = 0; i < 5; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    wxPrintf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
                    FailTest;
                }
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }

    /*
   *  Now test with odd dimensions into even dimensions
   */
    test_image.Allocate(5, 5, 5, true);
    test_image.SetToConstant(0.0);
    test_image.AddGaussianNoise(100.0);
    test_image_original.CopyFrom(&test_image);
    test_image.ForwardFFT( );
    big_image.Allocate(8, 8, 8, false);
    test_image.ClipInto(&big_image);
    big_image.BackwardFFT( );
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );
    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 5; k++ ) {
        for ( j = 0; j < 5; j++ ) {
            for ( i = 0; i < 5; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    wxPrintf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
                    FailTest;
                }
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }

    /*
   *  Now test with even dimensions into odd dimensions
   */
    test_image.Allocate(4, 4, 4, true);
    test_image.SetToConstant(0.0);
    test_image.AddGaussianNoise(100.0);
    test_image_original.CopyFrom(&test_image);
    test_image.ForwardFFT( );
    big_image.Allocate(9, 9, 9, false);
    test_image.ClipInto(&big_image);
    big_image.BackwardFFT( );
    big_image.ForwardFFT( );
    big_image.ClipInto(&test_image);
    test_image.BackwardFFT( );
    // Check we still have the same values
    address = 0;
    for ( k = 0; k < 4; k++ ) {
        for ( j = 0; j < 4; j++ ) {
            for ( i = 0; i < 4; i++ ) {
                if ( abs(test_image.real_values[address] - test_image_original.real_values[address]) > 0.001 ) {
                    wxPrintf("Voxel at address %i use to have value %f, now is %f\n", address, test_image_original.real_values[address], test_image.real_values[address]);
                    FailTest;
                }
                address++;
            }
            address += test_image.padding_jump_value;
        }
    }

    EndTest( );
}

void MyTestApp::TestDatabase( ) {
    BeginTest("Database");

    temp_directory             = wxFileName::GetTempDir( );
    wxString database_filename = temp_directory + "/1_0_test/1_0_test.db";
    Project  project;
    Database database;
    database.Open(database_filename);
    auto schema_result = database.CheckSchema( );
    if ( schema_result.first.size( ) < 1 || schema_result.second.size( ) < 1 ) {
        wxPrintf("Check Schema did not detect missing tables/columns\n");
        FailTest;
    }
    database.UpdateSchema(schema_result.second);
    schema_result = database.CheckSchema( );
    if ( schema_result.first.size( ) > 0 || schema_result.second.size( ) > 0 ) {
        wxPrintf("Update Schema did not fix missing tables/columns\n");
        FailTest;
    }
    database.Close( );
    project.OpenProjectFromFile(database_filename);
    if ( project.cistem_version_text != CISTEM_VERSION_TEXT ) {
        wxPrintf("New database does not have right version text\n");
        FailTest;
    }
    project.Close(false, true);

    EndTest( );
}

void MyTestApp::TestStarToBinaryFileConversion( ) {
    BeginTest("Star File To Binary Conversion");
    // generate set of 10k random parameters

    cisTEMParameters    test_parameters;
    cisTEMParameterLine temp_line;

    for ( unsigned int counter = 0; counter < 1000; counter++ ) {
        temp_line.amplitude_contrast                 = global_random_number_generator.GetUniformRandom( ) * 1;
        temp_line.assigned_subset                    = myroundint(global_random_number_generator.GetUniformRandom( ) * 10);
        temp_line.beam_tilt_group                    = myroundint(global_random_number_generator.GetUniformRandom( ) * 10);
        temp_line.beam_tilt_x                        = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.beam_tilt_y                        = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.best_2d_class                      = myroundint(global_random_number_generator.GetUniformRandom( ) * 100);
        temp_line.defocus_1                          = global_random_number_generator.GetUniformRandom( ) * 30000;
        temp_line.defocus_2                          = global_random_number_generator.GetUniformRandom( ) * 30000;
        temp_line.defocus_angle                      = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.image_is_active                    = myroundint(global_random_number_generator.GetUniformRandom( ) * 1);
        temp_line.image_shift_x                      = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.image_shift_y                      = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.logp                               = global_random_number_generator.GetUniformRandom( ) * 10000;
        temp_line.microscope_spherical_aberration_mm = global_random_number_generator.GetUniformRandom( ) * 2.7;
        temp_line.microscope_voltage_kv              = global_random_number_generator.GetUniformRandom( ) * 300;
        temp_line.occupancy                          = global_random_number_generator.GetUniformRandom( ) * 100;
        temp_line.original_image_filename            = wxString::Format("This_is_an_original_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
        temp_line.particle_group                     = myroundint(global_random_number_generator.GetUniformRandom( ) * 10);
        temp_line.phase_shift                        = global_random_number_generator.GetUniformRandom( ) * 3.14;
        temp_line.phi                                = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.pixel_size                         = global_random_number_generator.GetUniformRandom( ) * 2;
        temp_line.position_in_stack                  = counter + 1;
        temp_line.pre_exposure                       = global_random_number_generator.GetUniformRandom( ) * 10;
        temp_line.psi                                = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.reference_3d_filename              = wxString::Format("This_is_a_reference_3d_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
        temp_line.score                              = global_random_number_generator.GetUniformRandom( ) * 100;
        temp_line.sigma                              = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.stack_filename                     = wxString::Format("This_is_a_stack_filename_string_with_a_random_number_:_%f", global_random_number_generator.GetUniformRandom( ) * 100000);
        temp_line.theta                              = global_random_number_generator.GetUniformRandom( ) * 180;
        temp_line.total_exposure                     = global_random_number_generator.GetUniformRandom( ) * 100;
        temp_line.x_shift                            = global_random_number_generator.GetUniformRandom( ) * 50;
        temp_line.y_shift                            = global_random_number_generator.GetUniformRandom( ) * 50;

        test_parameters.all_parameters.Add(temp_line);
    }

    test_parameters.parameters_to_write.SetAllToTrue( );

    temp_directory = wxFileName::GetTempDir( );

    // write star and binary file..

    wxString original_star_filename    = temp_directory + "/star_file.star";
    wxString original_binary_filename  = temp_directory + "/binary_file.cistem";
    wxString star_from_binary_filename = temp_directory + "/star_file_converted_from_binary.star";
    wxString binary_from_star_filename = temp_directory + "/binary_file_converted_from_star.cistem";

    test_parameters.WriteTocisTEMStarFile(original_star_filename.ToStdString( ).c_str( ));
    test_parameters.WriteTocisTEMBinaryFile(original_binary_filename.ToStdString( ).c_str( ));

    // read in binary file and write out star file..

    test_parameters.ClearAll( );
    test_parameters.ReadFromcisTEMBinaryFile(original_binary_filename.ToStdString( ).c_str( ));
    test_parameters.WriteTocisTEMStarFile(star_from_binary_filename.ToStdString( ).c_str( ));

    // read in star file and write to binary..

    test_parameters.ClearAll( );
    test_parameters.ReadFromcisTEMStarFile(original_star_filename.ToStdString( ).c_str( ));
    test_parameters.WriteTocisTEMBinaryFile(binary_from_star_filename.ToStdString( ).c_str( ));

    // Check the sizes are the same - this isn't a very thorough test, but better than nothing.
    // It is at least an easy way to get files with all the parameters written out to quickly check by eye

    long original_star_size_in_bytes    = ReturnFileSizeInBytes(original_star_filename.ToStdString( ).c_str( ));
    long original_binary_size_in_bytes  = ReturnFileSizeInBytes(original_binary_filename.ToStdString( ).c_str( ));
    long star_from_binary_size_in_bytes = ReturnFileSizeInBytes(star_from_binary_filename.ToStdString( ).c_str( ));
    long binary_from_star_size_in_bytes = ReturnFileSizeInBytes(binary_from_star_filename.ToStdString( ).c_str( ));

    if ( original_star_size_in_bytes != star_from_binary_size_in_bytes )
        FailTest;
    if ( original_binary_size_in_bytes != binary_from_star_size_in_bytes )
        FailTest;

    // Check the star files are byte identical (apart from bytes in the text file which represent the write time)..
    // the binary files are not expected to be byte identical as there is a change in precision after being written out to star file.

    char original_star_file[original_star_size_in_bytes];
    char star_file_from_binary_file[star_from_binary_size_in_bytes];

    FILE* current_file;

    current_file = fopen(original_star_filename.ToStdString( ).c_str( ), "rb");
    fread(original_star_file, 1, original_star_size_in_bytes, current_file);
    fclose(current_file);

    current_file = fopen(star_from_binary_filename.ToStdString( ).c_str( ), "rb");
    fread(star_file_from_binary_file, 1, star_from_binary_size_in_bytes, current_file);
    fclose(current_file);

    // To avoid the first line which contains the cistem version and a timestamp, we look for the first line return.
    bool reading_first_line = true;
    for ( long byte_counter = 0; byte_counter < original_star_size_in_bytes; byte_counter++ ) {
        if ( reading_first_line ) {
            if ( original_star_file[byte_counter] == '\n' ) {
                reading_first_line = false;
            }
        }
        else {
            if ( original_star_file[byte_counter] != star_file_from_binary_file[byte_counter] ) {
                std::cerr << "failed on byte" << byte_counter << std::endl;
                FailTest;
            }
        }
    }

    EndTest( );
}

void MyTestApp::TestElectronExposureFilter( ) {
    // TODO: Depends on ZeroFloatArray, but this has no test.
    BeginTest("Test Electron Exposure Filter");

    /*
  Test exposure filter for a simple square image size, over all voltages and three pixel sizes.
  The "ground truth" values are taken from a print out using the code for electron_dose.* from commit 
  cdd0c04e984412661c983ab7176954093b502ad5 Dec 9, 2021
  using this line in the inner loop (once for odd once for even)
    for (auto & indx : indx_even)
    {
        wxPrintf("%3.9f, ", dose_filter_odd[indx]);
    }
  */
    const int size_small = 1024;

    std::vector<float> accelerating_voltage_vector = {300.f, 200.f, 100.f}; // only three supported values.
    std::vector<float> pixel_size_vector           = {0.72, 1.0, 2.1}; // values not chosen for any  good reason.
    std::vector<int>   indx_even                   = {0, 13, size_small / 3, size_small - 1};
    std::vector<int>   indx_odd                    = {0, 13, (size_small + 1) / 3, size_small};

    std::vector<float> ground_truth_even    = {1.000000000, 0.929921567, 0.017324856, 0.010133515, 1.000000000, 0.958591580, 0.031609546, 0.015432503, 1.000000000, 0.987710178, 0.155882418, 0.065504745, 1.000000000, 0.913183212, 0.006285460, 0.003215143, 1.000000000, 0.948510230, 0.013328240, 0.005439333, 1.000000000, 0.984661400, 0.097948194, 0.033139121, 1.000000000, 0.872345626, 0.000488910, 0.000178414, 1.000000000, 0.923584640, 0.001513943, 0.000393374, 1.000000000, 0.977023780, 0.030387951, 0.005955920};
    std::vector<float> ground_truth_odd     = {1.000000000, 0.930029809, 0.017352197, 0.010144006, 1.000000000, 0.958656907, 0.031672075, 0.015455280, 1.000000000, 0.987729967, 0.156189293, 0.065646693, 1.000000000, 0.913316071, 0.006297863, 0.003219303, 1.000000000, 0.948590994, 0.013361209, 0.005449366, 1.000000000, 0.984686077, 0.098189279, 0.033228900, 1.000000000, 0.872536480, 0.000490361, 0.000178761, 1.000000000, 0.923702955, 0.001519577, 0.000394466, 1.000000000, 0.977060616, 0.030500494, 0.005980202};
    int                ground_truth_counter = 0;
    ElectronDose*      my_electron_dose;

    Image test_image_small_even, test_image_small_odd;
    test_image_small_even.Allocate(size_small, size_small, true);
    test_image_small_odd.Allocate(size_small + 1, size_small + 1, true);

    float* dose_filter_even = new float[test_image_small_even.real_memory_allocated / 2];
    float* dose_filter_odd  = new float[test_image_small_odd.real_memory_allocated / 2];

    for ( auto& acceleration_voltage : accelerating_voltage_vector ) {
        for ( auto& pixel_size : pixel_size_vector ) {
            my_electron_dose = new ElectronDose(acceleration_voltage, pixel_size);

            ZeroFloatArray(dose_filter_even, test_image_small_even.real_memory_allocated / 2);
            ZeroFloatArray(dose_filter_odd, test_image_small_odd.real_memory_allocated / 2);

            my_electron_dose->CalculateDoseFilterAs1DArray(&test_image_small_even, dose_filter_even, 15.f, 30.f);
            my_electron_dose->CalculateDoseFilterAs1DArray(&test_image_small_odd, dose_filter_odd, 15.f, 30.f);

            for ( auto& indx : indx_even ) {
                if ( ! FloatsAreAlmostTheSame(dose_filter_even[indx], ground_truth_even[ground_truth_counter]) ) {
                    wxPrintf("Failed for kv,pix,ev: %3.f %3.3f, values %f %f\n", acceleration_voltage, pixel_size, dose_filter_even[indx], ground_truth_even[ground_truth_counter]);
                    FailTest;
                }
                if ( ! FloatsAreAlmostTheSame(dose_filter_odd[indx], ground_truth_odd[ground_truth_counter]) ) {
                    wxPrintf("Failed for kv,pix,od: %3.f %3.3f, values %f %f\n", acceleration_voltage, pixel_size, dose_filter_odd[indx], ground_truth_odd[ground_truth_counter]);
                    FailTest;
                }
                ground_truth_counter++;
            }

            delete my_electron_dose;
        }
    }

    delete[] dose_filter_even;
    delete[] dose_filter_odd;

    EndTest( );
}

#ifdef ENABLEGPU
void MyTestApp::TestElectronExposureFilterGPU( ) {
    // TODO: Depends on ZeroFloatArray, but this has no test.
    BeginTest("Test Electron Exposure Filter GPU");

    /*
  Test exposure filter for a simple square image size, over all voltages and three pixel sizes.
  The "ground truth" values are taken from a print out using the code for electron_dose.* from commit 
  cdd0c04e984412661c983ab7176954093b502ad5 Dec 9, 2021
  using this line in the inner loop (once for odd once for even)
    for (auto & indx : indx_even)
    {
        wxPrintf("%3.9f, ", dose_filter_odd[indx]);
    }
  */
    const int size_small = 1024;

    std::vector<float> accelerating_voltage_vector = {300.f, 200.f, 100.f}; // only three supported values.
    std::vector<float> pixel_size_vector           = {0.72, 1.0, 2.1}; // values not chosen for any  good reason.
    std::vector<int>   indx_even                   = {0, 13, size_small / 3, size_small - 1};
    // Note that moving forward, only even sized images are supported in the gpu code path (and any new cpu code I enter.)
    // I leave the "even" in the name now to make the comparison to the current CPU test more clear.
    std::vector<float> ground_truth_even = {1.000000000, 0.929921567, 0.017324856, 0.010133515, 1.000000000, 0.958591580, 0.031609546, 0.015432503, 1.000000000, 0.987710178, 0.155882418, 0.065504745, 1.000000000, 0.913183212, 0.006285460, 0.003215143, 1.000000000, 0.948510230, 0.013328240, 0.005439333, 1.000000000, 0.984661400, 0.097948194, 0.033139121, 1.000000000, 0.872345626, 0.000488910, 0.000178414, 1.000000000, 0.923584640, 0.001513943, 0.000393374, 1.000000000, 0.977023780, 0.030387951, 0.005955920};
    int                ground_truth_counter;
    ElectronDose*      my_electron_dose;

    Image test_image_small_even, filtered_output;
    test_image_small_even.Allocate(size_small, size_small, true);
    filtered_output.Allocate(size_small, size_small, true);

    // The dose filter is applied to the image as the cost of calculating it is lower than just
    // re-using a pre-calculated filter (in the event it is even re-used!)
    // Right now, using it in unblur, we apply it to a stack of images. The last image in the stack will
    // be for the results (i.e. sum_image in unblur)
    constexpr int n_images = 1;
    GpuImage*     d_test_image_small_even;
    cudaErr(cudaMallocManaged(&d_test_image_small_even, n_images * sizeof(GpuImage)));
    GpuImage d_filtered_output;

    for ( int i = 0; i < n_images; i++ ) {
        bool pin_host_memory = (i == 0) ? true : false;
        d_test_image_small_even[i].Init(test_image_small_even, pin_host_memory);
    }

    d_filtered_output.Init(filtered_output, true, true);

    // So no filters are actually calculated as in the cpu test

    for ( auto& acceleration_voltage : accelerating_voltage_vector ) {
        for ( auto& pixel_size : pixel_size_vector ) {
            my_electron_dose = new ElectronDose(acceleration_voltage, pixel_size);

            // We can get the filter values if we supply an input image that has a value of 1.0
            for ( int i = 0; i < n_images; i++ ) {
                d_test_image_small_even[i].SetToConstant(1.0f);
            }
            d_filtered_output.SetToConstant(3.0f);
            // Note: two differences to the cpu call a) we need to specifically NOT restore the noise power, and we pass the total exposure
            // as pre_exposure, and 0 as the exposure per frame, this way all three images should have the same dose filter applied.
            constexpr float pre_exposure       = 30.f;
            constexpr float exposure_per_frame = 0.f;

            my_electron_dose->CalculateDoseFilterAs1DArray<GpuImage, float2>(d_test_image_small_even, d_filtered_output.complex_values_gpu, pre_exposure, exposure_per_frame, n_images, false);

            for ( int i = 0; i < n_images; i++ ) {
                ground_truth_counter = 0;
                std::cerr << "Testing image " << i << std::endl;
                // Copy back to the host for comparison
                static constexpr bool free_gpu_mem   = false;
                static constexpr bool unpin_host_mem = false;
                d_filtered_output.CopyDeviceToHostAndSynchronize(free_gpu_mem, unpin_host_mem);

                for ( auto& indx : indx_even ) {
                    if ( ! FloatsAreAlmostTheSame(filtered_output.real_values[indx * 2], ground_truth_even[ground_truth_counter]) ) {
                        // wxPrintf("Failed for image %i/%i kv,pix,ev: %3.f %3.3f, values %f %f\n", i, n_images,
                        //          acceleration_voltage, pixel_size, filtered_output.real_values[indx * 2], ground_truth_even[ground_truth_counter]);

                        FailTest;
                    }
                    wxPrintf("Failed for image %i/%i kv,pix,ev: %3.f %3.3f, values %f %f\n", i, n_images,
                             acceleration_voltage, pixel_size, filtered_output.real_values[indx * 2], ground_truth_even[ground_truth_counter]);
                    ground_truth_counter++;
                }
            }

            delete my_electron_dose;
        }
    }
    cudaErr(cudaFree(d_test_image_small_even));
    EndTest( );
}

#endif // #ifdef ENABLEGPU

void MyTestApp::TestEmpiricalDistribution( ) {
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice"});

    BeginTest("Empirical Distribution");

    Image test_image;
    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);

    EmpiricalDistribution my_dist = test_image.ReturnDistributionOfRealValues( );

    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, 1.0f) )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), 1.0f) )
        FailTest;
    if ( my_dist.GetNumberOfSamples( ) != 6400 )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetMinimum( ), -3.1520f) )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetMaximum( ), 7.0222f) )
        FailTest;
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleSumOfSquares( ), 6400.f) )
        FailTest;

    EndTest( );
}

void MyTestApp::TestSumOfSquaresFourierAndFFTNormalization( ) {
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});

    BeginTest("Sum Of Squares Fourier");

    Image test_image;
    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    test_image.ForwardFFT(false);

    // By Parsevals theorem, the sum of squares of the Fourier transform of an image is equal to the sum of squares of the real image.
    // On the foward FFT the variance is scaled by N and on the inverse by N again. (This makes it as though the original values were scaled by N round-trip.)
    // So without normalization, the sum of squares should be N * realspace sumof squares.
    float sum_of_squares = test_image.ReturnSumOfSquares( );
    if ( ! RelativeErrorIsLessThanEpsilon(sum_of_squares, 6400.f * 6400.f) )
        FailTest;

    // We normalize for the full round trip on the forward FFT, so in this case the sum of squares should be
    // realspace sumof squares / N.
    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    test_image.ForwardFFT(true);
    sum_of_squares = test_image.ReturnSumOfSquares( );
    if ( ! RelativeErrorIsLessThanEpsilon(sum_of_squares, 1.f) )
        FailTest;

    EndTest( );
}

void MyTestApp::TestRandomVariableFunctions( ) {
    BeginTest("Random Variable Functions");

    float acceptable_error = 0.025f; // this is a random sample so it won't be so exact.

    // We want a reasonably large image to ensure that we sample the distribution well
    Image test_image;
    test_image.Allocate(1024, 1024, 1, true, false);
    EmpiricalDistribution my_dist = test_image.ReturnDistributionOfRealValues( );

    // Test multiple configurations of Gaussian noise, mean/sd 0,1 -2,1 3.1,4
    const std::vector<float> test_normal_vals{0.f, 1.f, -2.f, 1.f, 3.1f, 4.f};
    for ( int i = 0; i < test_normal_vals.size( ); i += 2 ) {
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoiseFromNormalDistribution(test_normal_vals[i], test_normal_vals[i + 1]);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        // Avoid zero division by adding 1
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, test_normal_vals[i] + 1.f, true, acceptable_error) )
            FailTest;
        if ( ! RelativeErrorIsLessThanEpsilon(sqrtf(my_dist.GetSampleVariance( )), test_normal_vals[i + 1], true, acceptable_error) )
            FailTest;
    }

    // Test multiple configurations of Poisson noise, mean/sd 0.1,1.2,4.0
    // mean and variance should be ~ equal
    const std::vector<float> test_poisson_vals{0.1, 1.2, 4.0};
    for ( auto& val : test_poisson_vals ) {
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoiseFromPoissonDistribution(val);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ), val, true, acceptable_error) )
            FailTest;
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), val, true, acceptable_error) )
            FailTest;
    }

    // Test uniform distribution, defined by its max, min and mean should be close to zero
    const float uniform_min = -1.f;
    const float uniform_max = 1.f;
    test_image.SetToConstant(0.f);
    my_dist.Reset( );
    test_image.AddNoiseFromUniformDistribution(uniform_min, uniform_max);
    test_image.UpdateDistributionOfRealValues(&my_dist);
    // Avoid zero division by adding 1
    if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, 1.f, true, acceptable_error) )
        FailTest;
    if ( my_dist.GetMinimum( ) < uniform_min )
        FailTest;
    if ( my_dist.GetMaximum( ) > uniform_max )
        FailTest;

    // Test the exponential distribution, mean = 1/lambda, sd = 1/lambda,
    // Re-use the poisson test values
    for ( auto& val : test_poisson_vals ) {
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoiseFromExponentialDistribution(val);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ), 1.f / val, true, acceptable_error) )
            FailTest;
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), 1.f / (val * val), true, acceptable_error) )
            FailTest;
    }

    // Test the gamma distribution, mean = alpha*theta, variance = alpha*theta^2
    // Re-use the poisson test values
    float beta;
    for ( auto& alpha : test_poisson_vals ) {
        beta = alpha + 1.f;
        test_image.SetToConstant(0.f);
        my_dist.Reset( );
        test_image.AddNoiseFromGammaDistribution(alpha, beta);
        test_image.UpdateDistributionOfRealValues(&my_dist);
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ), alpha * beta, true, 2.0f * acceptable_error) ) {
            wxPrintf("m,a/b %f %f\n", my_dist.GetSampleMean( ), alpha * beta);
            FailTest;
        }
        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleVariance( ), alpha * beta * beta, true, 2.0f * acceptable_error) )
            FailTest;
    }

    EndTest( );
}

void MyTestApp::TestIntegerShifts( ) {
    // Dependencies?
    BeginTest("Integer Shifts and Rotations");
    CheckDependencies({"Memory Assignment Ops and Funcs"});

    // TODO Image::RotateQuadrants should be tested, but I'm not clear on what it is supposed to do.

    // Goal is to verify image transforms that work without interpolation.
    const int eve_size = 16;
    const int odd_size = 17;

    Image odd_image, eve_image;
    odd_image.Allocate(odd_size, odd_size, 1, true, false);
    eve_image.Allocate(eve_size, eve_size, 1, true, false);

    // Set values around the center, which is also the origin for rotations.
    const int ocx = odd_image.physical_address_of_box_center_x;
    const int ocy = odd_image.physical_address_of_box_center_y;
    const int ecx = eve_image.physical_address_of_box_center_x;
    const int ecy = eve_image.physical_address_of_box_center_y;

    // Locations and test values.
    std::vector<float> tv = {1.f, 2.f, 3.f, 4.f};
    std::vector<int>   sx = {2, 0, -2, 0}; // x shift
    std::vector<int>   sy = {0, 2, 0, -2}; // y shift

    std::vector<int> addr_odd = {0, 0, 0, 0};
    std::vector<int> addr_eve = {0, 0, 0, 0};

    const int origin_addr_odd = odd_image.ReturnReal1DAddressFromPhysicalCoord(ocx, ocy, 0);
    const int origin_addr_eve = eve_image.ReturnReal1DAddressFromPhysicalCoord(ecx, ecy, 0);
    float     tmp             = 0.f;
    for ( int i = 0; i < addr_odd.size( ); i++ ) {
        addr_odd[i] = odd_image.ReturnReal1DAddressFromPhysicalCoord(ocx + sx[i], ocy + sy[i], 0);
        addr_eve[i] = eve_image.ReturnReal1DAddressFromPhysicalCoord(ecx + sx[i], ecy + sy[i], 0);
        tmp += tv[i];
    }

    // make sure we don't change the value for the origin
    const float origin_val = tmp;

    /*
      y ->
   x 
   |  . . .0 0 3 0 0
   V  . . .0 0 0 0 0
      . . .4 0 6 0 2
      . . .0 0 0 0 0
      . . .0 0 1 0 0

      Note that for an even sized image, the origin is effectively shifted by -1 in x on a +90 rotation
      This is an "artifact" of defining the origin based on the pixel grid and not the underlying image data.
      0 0     0 6
      0 6  -> 0 0
  */

    odd_image.SetToConstant(0.f);
    eve_image.SetToConstant(0.f);
    odd_image.real_values[origin_addr_odd] = origin_val;
    eve_image.real_values[origin_addr_eve] = origin_val;
    int from_addr;
    for ( int i = 0; i < tv.size( ); i++ ) {
        odd_image.real_values[addr_odd[i]] = tv[i];
        eve_image.real_values[addr_eve[i]] = tv[i];
    }

    Image odd_test_image, eve_test_image;
    odd_test_image.CopyFrom(&odd_image);
    eve_test_image.CopyFrom(&eve_image);

    bool rotate_by_positive_90_degrees = true;
    bool preserve_origin               = false;
    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve - 1] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = (i + tv.size( ) + 1) % tv.size( );
        if ( odd_test_image.real_values[addr_odd[from_addr]] != tv[i] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[from_addr] - 1] != tv[i] )
            FailTest;
    }

    // Test rotating back by -90 degress, everything should be identical to the starting conditions.
    rotate_by_positive_90_degrees = false;
    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = i;
        if ( odd_test_image.real_values[addr_odd[from_addr]] != tv[i] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[from_addr]] != tv[i] )
            FailTest;
    }

    // Test forward rotation with a integer base shift (shouldn't need offsets in Y)
    rotate_by_positive_90_degrees = true;
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees);
    eve_test_image.RealSpaceIntegerShift(-1, 0, 0);

    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = (i + tv.size( ) + 1) % tv.size( );
        if ( eve_test_image.real_values[addr_eve[from_addr]] != tv[i] )
            FailTest;
    }

    // Test combined shift and rotate, from a clean copy
    preserve_origin = true;
    odd_test_image.CopyFrom(&odd_image);
    eve_test_image.CopyFrom(&eve_image);

    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);

    // origin should be the same, pixel values should be permuted one value counter-clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;

    for ( int i = 0; i < tv.size( ); i++ ) {
        from_addr = (i + tv.size( ) + 1) % tv.size( );
        if ( odd_test_image.real_values[addr_odd[from_addr]] != tv[i] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[from_addr]] != tv[i] )
            FailTest;
    }

    // Test combined shift and inverse rotate
    rotate_by_positive_90_degrees = false;
    odd_test_image.CopyFrom(&odd_image);
    eve_test_image.CopyFrom(&eve_image);
    odd_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);
    eve_test_image.RotateInPlaceAboutZBy90Degrees(rotate_by_positive_90_degrees, preserve_origin);

    // origin should be the same, pixel values should be permuted one value clockwise.
    if ( odd_test_image.real_values[origin_addr_odd] != origin_val )
        FailTest;
    if ( eve_test_image.real_values[origin_addr_eve] != origin_val )
        FailTest;
    for ( int i = 0; i < tv.size( ); i++ ) {
        // Note the difference in address here.
        from_addr = (i + 1) % tv.size( );
        if ( odd_test_image.real_values[addr_odd[i]] != tv[from_addr] )
            FailTest;
        if ( eve_test_image.real_values[addr_eve[i]] != tv[from_addr] )
            FailTest;
    }

    EndTest( );
}

void MyTestApp::TestImageLoopingAndAddressing( ) {
    BeginTest("Image::LoopingAndAddressing");

    Image test_image;

    //
    // Even
    //
    test_image.Allocate(4, 4, 4, true);

    if ( test_image.physical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_y != 3 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_z != 3 )
        FailTest;

    if ( test_image.physical_address_of_box_center_x != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_y != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_z != 2 )
        FailTest;

    if ( test_image.physical_index_of_first_negative_frequency_y != 2 )
        FailTest;
    if ( test_image.physical_index_of_first_negative_frequency_z != 2 )
        FailTest;

    if ( test_image.logical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_y != 1 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_z != 1 )
        FailTest;

    if ( test_image.logical_lower_bound_complex_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_z != -2 )
        FailTest;

    if ( test_image.logical_upper_bound_real_x != 1 )
        FailTest;
    if ( test_image.logical_upper_bound_real_y != 1 )
        FailTest;
    if ( test_image.logical_upper_bound_real_z != 1 )
        FailTest;

    if ( test_image.logical_lower_bound_real_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_z != -2 )
        FailTest;

    if ( test_image.padding_jump_value != 2 )
        FailTest;

    //
    // Odd
    //
    test_image.Allocate(5, 5, 5, true);

    if ( test_image.physical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_y != 4 )
        FailTest;
    if ( test_image.physical_upper_bound_complex_z != 4 )
        FailTest;

    if ( test_image.physical_address_of_box_center_x != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_y != 2 )
        FailTest;
    if ( test_image.physical_address_of_box_center_z != 2 )
        FailTest;

    if ( test_image.physical_index_of_first_negative_frequency_y != 3 )
        FailTest;
    if ( test_image.physical_index_of_first_negative_frequency_z != 3 )
        FailTest;

    if ( test_image.logical_upper_bound_complex_x != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_y != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_complex_z != 2 )
        FailTest;

    if ( test_image.logical_lower_bound_complex_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_complex_z != -2 )
        FailTest;

    if ( test_image.logical_upper_bound_real_x != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_real_y != 2 )
        FailTest;
    if ( test_image.logical_upper_bound_real_z != 2 )
        FailTest;

    if ( test_image.logical_lower_bound_real_x != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_y != -2 )
        FailTest;
    if ( test_image.logical_lower_bound_real_z != -2 )
        FailTest;

    if ( test_image.padding_jump_value != 1 )
        FailTest;

    EndTest( );
}

void MyTestApp::TestSpectrumBoxConvolution( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});

    BeginTest("Image::SpectrumBoxConvolution");

    Image test_image;
    Image output_image;

    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);

    output_image.Allocate(test_image.logical_x_dimension, test_image.logical_y_dimension, test_image.logical_z_dimension);
    test_image.SpectrumBoxConvolution(&output_image, 7, 3);

    if ( FloatsAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.049189) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 1.634473) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(output_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), -0.049189) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestImageArithmeticFunctions( ) {
    CheckDependencies({"MRCFile::ReadSlice", "MRCFile::OpenFile"});
    // AddImage
    BeginTest("Image::AddImage");

    Image test_image;
    Image ref_image;
    Peak  my_peak;

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString( ), 1);
    ref_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString( ), 2);
    test_image.AddImage(&ref_image);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -1.313164) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 3.457573) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.318875) == false )
        FailTest;

    EndTest( );

    BeginTest("Image::SetToConstant");
    // SetToConstant, also sets FFTW padding addresses as well
    test_image.SetToConstant(3.14f);
    for ( long pixel_counter = 0; pixel_counter < test_image.real_memory_allocated; pixel_counter++ ) {
        if ( test_image.real_values[pixel_counter] != 3.14f )
            FailTest;
    }
    EndTest( );
}

void MyTestApp::TestNumericTextFiles( ) {
    // AddImage
    BeginTest("NumericTextFile::Init");

    NumericTextFile test_file(numeric_text_filename, OPEN_TO_READ);

    if ( test_file.number_of_lines != 4 )
        FailTest;
    if ( test_file.records_per_line != 5 )
        FailTest;

    EndTest( );

    BeginTest("NumericTextFile::ReadLine");
    float temp_float[5];

    test_file.ReadLine(temp_float);

    if ( int(temp_float[0]) != 1 )
        FailTest;
    if ( int(temp_float[1]) != 2 )
        FailTest;
    if ( int(temp_float[2]) != 3 )
        FailTest;
    if ( int(temp_float[3]) != 4 )
        FailTest;
    if ( int(temp_float[4]) != 5 )
        FailTest;

    test_file.ReadLine(temp_float);

    if ( FloatsAreAlmostTheSame(temp_float[0], 6.0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[1], 7.1) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[2], 8.3) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[3], 9.4) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[4], 10.5) == false )
        FailTest;

    test_file.ReadLine(temp_float);

    if ( FloatsAreAlmostTheSame(temp_float[0], 11.2) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[1], 12.7) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[2], 13.2) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[3], 14.1) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[4], 15.8) == false )
        FailTest;

    test_file.ReadLine(temp_float);

    if ( FloatsAreAlmostTheSame(temp_float[0], 16.1245) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[1], 17.81003) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[2], 18.5467) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[3], 19.7621) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[4], 20.11111) == false )
        FailTest;

    EndTest( );

    BeginTest("NumericTextFile::WriteLine");

    wxString        output_filename = temp_directory + "/number_out.num";
    NumericTextFile output_test_file(output_filename, OPEN_TO_WRITE, 5);

    temp_float[0] = 0.1;
    temp_float[1] = 0.2;
    temp_float[2] = 0.3;
    temp_float[3] = 0.4;
    temp_float[4] = 0.5;

    output_test_file.WriteCommentLine("This is a comment line %i", 5);
    output_test_file.WriteLine(temp_float);
    output_test_file.WriteCommentLine("Another comment = %s", "booooo!");
    temp_float[0] = 0.67;
    temp_float[1] = 0.78;
    temp_float[2] = 0.89;
    temp_float[3] = 0.91;
    temp_float[4] = 1.02;

    output_test_file.WriteLine(temp_float);
    output_test_file.Flush( );

    test_file.Close( );
    test_file.Open(output_filename, OPEN_TO_READ);

    if ( test_file.number_of_lines != 2 )
        FailTest;
    if ( test_file.records_per_line != 5 )
        FailTest;

    test_file.ReadLine(temp_float);

    if ( FloatsAreAlmostTheSame(temp_float[0], 0.1) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[1], 0.2) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[2], 0.3) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[3], 0.4) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[4], 0.5) == false )
        FailTest;

    test_file.ReadLine(temp_float);

    if ( FloatsAreAlmostTheSame(temp_float[0], 0.67) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[1], 0.78) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[2], 0.89) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[3], 0.91) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(temp_float[4], 1.02) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestAlignmentFunctions( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});
    // TODO: CalculateCrossCorrelationImageWith depends on SwapRealSpaceQuadrants, which depends on PhaseShift

    // Phaseshift
    BeginTest("Image::PhaseShift");

    Image test_image;
    Image ref_image;
    Peak  my_peak;

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString( ), 1);
    test_image.PhaseShift(20, 20, 0);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -1.010296) == false ) {
        FailTest;
    }
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), -2.280109) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.239702) == false )
        FailTest;

    EndTest( );

    // TODO: Add SwapRealSpaceQuadrants test here. (See note above)

    // CalculateCrossCorrelationImageWith
    BeginTest("Image::CalculateCrossCorrelationImageWith");

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString( ), 1);
    ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    test_image.CalculateCrossCorrelationImageWith(&ref_image);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 0.004323) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 0.543692) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.006927) == false )
        FailTest;

    EndTest( );

    //FindPeakWithIntegerCoordinates

    BeginTest("Image::FindPeakWithIntegerCoordinates");

    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    test_image.PhaseShift(7, 10, 0);

    test_image.CalculateCrossCorrelationImageWith(&ref_image);
    my_peak = test_image.FindPeakWithIntegerCoordinates( );

    if ( FloatsAreAlmostTheSame(my_peak.x, 7.0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.y, 10.0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.z, 0) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.value, 1) == false )
        FailTest;

    EndTest( );

    //FindPeakWithParabolaFit

    BeginTest("Image::FindPeakWithParabolaFit");

    test_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    ref_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);
    test_image.PhaseShift(7.3, 10.7, 0);

    test_image.CalculateCrossCorrelationImageWith(&ref_image);
    my_peak = test_image.FindPeakWithParabolaFit( );

    if ( my_peak.x > 7.3 || my_peak.x < 7.29 )
        FailTest;
    if ( my_peak.y > 10.70484 || my_peak.y < 10.70481 )
        FailTest;
    if ( FloatsAreAlmostTheSame(my_peak.z, 0) == false )
        FailTest;
    if ( my_peak.value > 0.99343 || my_peak.value < 0.99342 )
        FailTest;

    EndTest( );
}

void MyTestApp::TestFilterFunctions( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Image::BackwardFFT"});

    // BFACTOR
    BeginTest("Image::ApplyBFactor");

    Image test_image;

    test_image.QuickAndDirtyReadSlice(hiv_images_80x80x10_filename.ToStdString( ), 1);
    test_image.ForwardFFT( );
    test_image.ApplyBFactor(1500);
    test_image.BackwardFFT( );

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 0.027244) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), 1.320998) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(79, 79, 0), 0.012282) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestMaskCentralCross( ) {
    CheckDependencies({"Image::SetToConstant"});

    BeginTest("Image::MaskCentralCross");

    Image my_image;

    my_image.Allocate(128, 128, 1);
    my_image.SetToConstant(1.0);
    my_image.MaskCentralCross(3, 3);

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(127, 127, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 127, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(127, 0, 0), 1.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67, 67, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67, 61, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61, 61, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61, 67, 0), 1.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 61, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 67, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(61, 0, 0), 1.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(67, 0, 0), 1.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66, 66, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66, 62, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62, 62, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62, 66, 0), 0.0) )
        FailTest;

    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 62, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(0, 66, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(62, 0, 0), 0.0) )
        FailTest;
    if ( ! FloatsAreAlmostTheSame(my_image.ReturnRealPixelFromPhysicalCoord(66, 0, 0), 0.0) )
        FailTest;

    EndTest( );
}

void MyTestApp::TestScalingAndSizingFunctions( ) {

    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::ForwardFFT", "Memory Assignment Ops and Funcs"});

    BeginTest("Image::ClipInto");

    MRCFile             input_file(hiv_images_80x80x10_filename.ToStdString( ), false);
    Image               test_image;
    Image               clipped_image;
    std::complex<float> test_pixel;

    // test real space clipping bigger..

    clipped_image.Allocate(160, 160, 1);

    test_image.ReadSlice(&input_file, 1);
    test_image.ClipInto(&clipped_image, 0);

    //wxPrintf("value = %f\n", clipped_image.ReturnRealPixelValue(119,119));
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), -0.340068) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(80, 80, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(119, 119, 0), 0.637069) == false )
        FailTest;

    // test real space clipping smaller..

    clipped_image.Allocate(50, 50, 1);
    test_image.ClipInto(&clipped_image, 0);

    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -2.287762) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(49, 49, 0), -1.773780) == false )
        FailTest;

    // test Fourier space clipping bigger

    clipped_image.Allocate(160, 160, 1);
    test_image.ForwardFFT( );
    test_image.ClipInto(&clipped_image, 0);

    // check some values

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -100.0) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    // test Fourier space clipping smaller

    clipped_image.Allocate(50, 50, 1);
    test_image.ClipInto(&clipped_image, 0);

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    //wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    // test real space clipping smaller to odd..

    test_image.ReadSlice(&input_file, 1);
    clipped_image.Allocate(49, 49, 1);
    test_image.ClipInto(&clipped_image, 0);

    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -0.391899) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.689942) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(clipped_image.ReturnRealPixelFromPhysicalCoord(48, 48, 0), -1.773780) == false )
        FailTest;

    // test fourier space flipping smaller to odd..

    test_image.ForwardFFT( );
    test_image.ClipInto(&clipped_image, 0);

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = clipped_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    //wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    EndTest( );

    // Check Resize..

    BeginTest("Image::Resize");

    test_image.ReadSlice(&input_file, 1);
    test_image.Resize(160, 160, 1);

    //Real space big

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(40, 40, 0), -0.340068) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(80, 80, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(119, 119, 0), 0.637069) == false )
        FailTest;

    // Real space small

    test_image.ReadSlice(&input_file, 1);
    test_image.Resize(50, 50, 1);

    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(0, 0, 0), -2.287762) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(25, 25, 0), 1.819805) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.ReturnRealPixelFromPhysicalCoord(49, 49, 0), -1.773780) == false )
        FailTest;

    // Fourier space big

    test_image.ReadSlice(&input_file, 1);
    test_image.ForwardFFT( );
    test_image.Resize(160, 160, 1);

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(-90, -90, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -100.0) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    // Fourier space small

    test_image.ReadSlice(&input_file, 1);
    test_image.ForwardFFT( );
    test_image.Resize(50, 50, 1);

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(0, 0, 0, -100.0f + I * 0.0f);
    if ( FloatsAreAlmostTheSame(real(test_pixel), -0.010919) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.0) == false )
        FailTest;

    test_pixel = test_image.ReturnComplexPixelFromLogicalCoord(5, 5, 0, -100.0f + I * 0.0f);
    //wxPrintf("real = %f, image = %f\n", creal(test_pixel),cimag(test_pixel));
    if ( FloatsAreAlmostTheSame(real(test_pixel), 0.075896) == false || FloatsAreAlmostTheSame(imag(test_pixel), 0.045677) == false )
        FailTest;

    EndTest( );

    //wxPrintf ("real = %f, imag = %f", creal(test_pixel), cimag(test_pixel), 0.0);
}

void MyTestApp::TestMRCFunctions( ) {
    BeginTest("MRCFile::OpenFile");

    MRCFile input_file(hiv_images_80x80x10_filename.ToStdString( ), false);

    // check dimensions..

    if ( input_file.ReturnNumberOfSlices( ) != 10 )
        FailTest;
    if ( input_file.ReturnXSize( ) != 80 )
        FailTest;
    if ( input_file.ReturnYSize( ) != 80 )
        FailTest;
    if ( input_file.ReturnZSize( ) != 10 )
        FailTest;

    EndTest( );

    BeginTest("MRCFile::ReadSlice");

    Image test_image;
    test_image.ReadSlice(&input_file, 1);

    // check dimensions and type

    if ( test_image.is_in_real_space == false )
        FailTest;
    if ( test_image.logical_x_dimension != 80 )
        FailTest;
    if ( test_image.logical_y_dimension != 80 )
        FailTest;
    if ( test_image.logical_z_dimension != 1 )
        FailTest;

    // check first and last pixel...

    if ( FloatsAreAlmostTheSame(test_image.real_values[0], -0.340068) == false )
        FailTest;
    if ( FloatsAreAlmostTheSame(test_image.real_values[test_image.real_memory_allocated - 3], 0.637069) == false )
        FailTest;

    EndTest( );
}

void MyTestApp::TestFFTFunctions( ) {

    long counter;

    BeginTest("Image::ForwardFFT");
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Image::SetToConstant"});

    MRCFile input_file(sine_wave_128x128x1_filename.ToStdString( ), false);

    // make an image that is all 1..

    Image test_image;
    test_image.Allocate(64, 64, 1);
    test_image.SetToConstant(1);

    // ForwardFFT

    test_image.ForwardFFT( );

    // first pixel should be 1,0

    if ( FloatsAreAlmostTheSame(real(test_image.complex_values[0]), 1) == false || FloatsAreAlmostTheSame(imag(test_image.complex_values[0]), 0) == false )
        FailTest;

    // if we set this to 0,0 - all remaining pixels should now be 0

    test_image.complex_values[0] = 0.0f + 0.0f * I;

    for ( counter = 0; counter < test_image.real_memory_allocated / 2; counter++ ) {
        if ( FloatsAreAlmostTheSame(real(test_image.complex_values[counter]), 0) == false || FloatsAreAlmostTheSame(imag(test_image.complex_values[counter]), 0) == false )
            FailTest;
    }

    // sine wave

    test_image.ReadSlice(&input_file, 1);
    test_image.ForwardFFT( );

    // now one pixel should be set, and the rest should be 0..

    if ( FloatsAreAlmostTheSame(real(test_image.complex_values[20]), 0) == false || FloatsAreAlmostTheSame(imag(test_image.complex_values[20]), -5) == false )
        FailTest;
    // set it to 0, then everything should be zero..

    test_image.complex_values[20] = 0.0f + 0.0f * I;

    for ( counter = 0; counter < test_image.real_memory_allocated / 2; counter++ ) {
        if ( real(test_image.complex_values[counter]) > 0.000001 || imag(test_image.complex_values[counter]) > 0.000001 )
            FailTest;
    }

    EndTest( );

    // Backward FFT

    BeginTest("Image::BackwardFFT");

    test_image.Allocate(64, 64, 1, false);
    test_image.SetToConstant(0.0);
    test_image.complex_values[0] = 1.0f + 0.0f * I;
    test_image.BackwardFFT( );
    test_image.RemoveFFTWPadding( );

    for ( counter = 0; counter < test_image.logical_x_dimension * test_image.logical_y_dimension; counter++ ) {
        if ( FloatsAreAlmostTheSame(test_image.real_values[counter], 1.0) == false )
            FailTest;
    }

    EndTest( );
}

#ifdef _CISTEM_MODULES
void MyTestApp::TestModules( ) {

    BeginTest("Cpp20 Module");
    if ( hello::test_return_value( ) != 40 ) {
        FailTest;
    }
    if ( hello2::test_return_value( ) != 42 ) {
        FailTest;
    }

    EndTest( );
}
#endif

#ifdef ENABLE_FastFFT
void MyTestApp::TestFastFFT( ) {

    BeginTest("Fast FFT");
    CheckDependencies({"MRCFile::OpenFile", "MRCFile::ReadSlice", "Empirical Distribution"});

    Image input_image;
    input_image.QuickAndDirtyReadSlice(hiv_image_80x80x1_filename.ToStdString( ), 1);

    input_image.Resize(64, 64, 1);

    if ( input_image.logical_x_dimension == 64 && input_image.logical_y_dimension == 64 ) {
        // Make a copy of the image to transform forward and back on the cpu , which should give a scaled image by N
        Image copy_of_input, cpu_result;
        copy_of_input.CopyFrom(&input_image);
        copy_of_input.ForwardFFT(false); // no scaling
        copy_of_input.BackwardFFT( );
        cpu_result.CopyFrom(&copy_of_input);
        cpu_result.ZeroFloatAndNormalize( );
        // Make a copy of the image to transform forward and back on the gpu, which should give a scaled image by N
        copy_of_input.CopyFrom(&input_image);

        // We just make one instance of the FourierTransformer class, with calc type float.
        FastFFT::FourierTransformer<float, float, float, 2> FT;

        // This is similar to creating an FFT/CUFFT plan, so set these up before doing anything on the GPU
        // for this test, we know the size is square and 4096 with input and output size equal.
        FT.SetForwardFFTPlan(copy_of_input.logical_x_dimension, copy_of_input.logical_y_dimension, 1, copy_of_input.logical_x_dimension, copy_of_input.logical_y_dimension, 1, true, false);
        FT.SetInverseFFTPlan(copy_of_input.logical_x_dimension, copy_of_input.logical_y_dimension, 1, copy_of_input.logical_x_dimension, copy_of_input.logical_y_dimension, 1, true);

        // The padding (dims.w) is calculated based on the setup
        short4 dims_in  = FT.ReturnFwdInputDimensions( );
        short4 dims_out = FT.ReturnFwdOutputDimensions( );

        FT.SetInputPointer(copy_of_input.real_values, false);
        FT.CopyHostToDevice( );
        FT.FwdFFT( );
        FT.InvFFT( );
        FT.CopyDeviceToHost(true, true);
        FT.Wait( );

        copy_of_input.ZeroFloatAndNormalize( );
        copy_of_input.SubtractImage(&cpu_result);
        EmpiricalDistribution my_dist = copy_of_input.ReturnDistributionOfRealValues( );

        if ( ! RelativeErrorIsLessThanEpsilon(my_dist.GetSampleMean( ) + 1.f, 1.0f, true, float(1e-6)) ) {
            FailTest;
        }
    }
    else {
        wxPrintf("Not testing FastFFT for a %i x %i image\n", input_image.logical_x_dimension, true, input_image.logical_y_dimension);
    }

    EndTest( );
}
#endif

void MyTestApp::BeginTest(const char* test_name) {
    // For access by other tests when running CheckDependencies
    current_test_name               = test_name;
    test_results[current_test_name] = false;

    int length      = strlen(test_name);
    int blank_space = 45 - length;
    wxPrintf("Testing %s ", test_name);
    test_has_passed = true;

    for ( int counter = 0; counter < blank_space; counter++ ) {
        wxPrintf(" ");
    }

    wxPrintf(": ");
}

void MyTestApp::EndTest( ) {
    if ( test_has_passed == true ) {
        // For access by other tests when running CheckDependencies
        test_results[current_test_name] = true;
        PrintResult(true);
    }
    else {
        // Sets the final return value, used in auto build &
        all_tests_have_passed = false;
    }
}

bool MyTestApp::CheckDependencies(std::initializer_list<std::string> list) {
    // Nothing has been added, so must be false.'
    bool ret_val;
    if ( test_results.empty( ) ) {
        wxPrintf("\nCheckDependencies: No tests have been run yet.\n");
        ret_val = false;
    }
    else {
        for ( auto dep : list ) {
            auto search = test_results.find(dep);
            if ( search == test_results.end( ) ) {
                wxPrintf("\nCheckDependencies: %s has not been run.\n", dep);
                ret_val = false;
                break;
            }
            else {
                if ( search->second == false ) {
                    wxPrintf("\nCheckDependencies: %s has previously failed.\n", dep);
                    ret_val = false;
                    break;
                }
                else {
                    ret_val = true;
                    break;
                }
            }
        }
    }
    return ret_val;
}

void MyTestApp::PrintResultWorker(bool passed, int line) {

    if ( passed == true ) {
        if ( OutputIsAtTerminal( ) == true )
            wxPrintf(ANSI_COLOR_GREEN "PASSED!" ANSI_COLOR_RESET);
        else
            wxPrintf("PASSED!");
    }
    else {
        if ( OutputIsAtTerminal( ) == true )
            wxPrintf(ANSI_COLOR_RED "\n\t\tFAILED! (Line : %i)" ANSI_COLOR_RESET, line);
        else
            wxPrintf("\n\t\tFAILED\n! (Line : %i)", line);
        // Removing the exit behavior because I want all tests to run as this is more informative for CI, i.e.
        // multiple fixes can be made in a single go rather than a one at a time approach.
        // exit(1);
    }

    wxPrintf("\n");
}

void MyTestApp::PrintTitle(const char* title) {
    wxPrintf("\n");
    if ( OutputIsAtTerminal( ) == true )
        wxPrintf(ANSI_UNDERLINE "%s" ANSI_UNDERLINE_OFF, title);
    else
        wxPrintf("%s", title);
    wxPrintf("\n\n");
}

void MyTestApp::WriteEmbeddedFiles( ) {
    temp_directory = wxFileName::GetTempDir( );
    wxPrintf("\nWriting out embedded test files to '%s'...", temp_directory);
    fflush(stdout);

    hiv_image_80x80x1_filename   = temp_directory + "/hiv_image_80x80x1.mrc";
    hiv_images_80x80x10_filename = temp_directory + "/hiv_images_shift_noise_80x80x10.mrc";
    sine_wave_128x128x1_filename = temp_directory + "/sine_wave_128x128x1.mrc";

    WriteEmbeddedArray(hiv_image_80x80x1_filename, hiv_image_80x80x1_array, sizeof(hiv_image_80x80x1_array));
    WriteEmbeddedArray(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
    WriteEmbeddedArray(hiv_images_80x80x10_filename, hiv_images_shift_noise_80x80x10_array, sizeof(hiv_images_shift_noise_80x80x10_array));
    WriteEmbeddedArray(sine_wave_128x128x1_filename, sine_128x128x1_array, sizeof(sine_128x128x1_array));

    numeric_text_filename = temp_directory + "/numbers.num";
    WriteNumericTextFile(numeric_text_filename);
    WriteDatabase(temp_directory + "/1_0_test", temp_directory + "/1_0_test/1_0_test.db");
    wxPrintf("done!\n");
}

void MyTestApp::WriteEmbeddedArray(const char* filename, const unsigned char* array, long length) {

    FILE* output_file = NULL;
    output_file       = fopen(filename, "wb+");

    if ( output_file == NULL ) {
        wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
        wxPrintf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n", filename);
        DEBUG_ABORT;
    }

    fwrite(array, sizeof(unsigned char), length, output_file);

    fclose(output_file);
}

void MyTestApp::WriteDatabase(const char* dir, const char* filename) {
    Database database;
    wxFileName::Mkdir(dir, 0777, wxPATH_MKDIR_FULL);
    wxFileName::Mkdir(std::string(dir) + "/Assets/", 0777, wxPATH_MKDIR_FULL);
    wxFileName db_filename = wxFileName(filename);
    if ( db_filename.Exists( ) )
        wxRemoveFile(filename);
    database.CreateNewDatabase(db_filename);
    database.ExecuteSQL(R"sql(
	PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE MASTER_SETTINGS(NUMBER INTEGER PRIMARY KEY, PROJECT_DIRECTORY TEXT, PROJECT_NAME TEXT, CURRENT_VERSION INTEGER, TOTAL_CPU_HOURS REAL, TOTAL_JOBS_RUN INTEGER );
INSERT INTO MASTER_SETTINGS VALUES(1,'/tmp/1_0_test','1_0_test',1,0.0,0);
CREATE TABLE RUNNING_JOBS(JOB_NUMBER INTEGER PRIMARY KEY, JOB_CODE TEXT, MANAGER_IP_ADDRESS INTEGER );
CREATE TABLE RUN_PROFILES(RUN_PROFILE_ID INTEGER PRIMARY KEY, PROFILE_NAME TEXT, MANAGER_RUN_COMMAND TEXT, GUI_ADDRESS TEXT, CONTROLLER_ADDRESS TEXT, COMMANDS_ID INTEGER );
INSERT INTO RUN_PROFILES VALUES(1,'Default Local','/groups/cryoadmin/software/CISTEM/cistem-1.0.0-beta/$command','','',1);
CREATE TABLE MOVIE_ASSETS(MOVIE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, POSITION_IN_STACK INTEGER, X_SIZE INTEGER, Y_SIZE INTEGER, NUMBER_OF_FRAMES INTEGER, VOLTAGE REAL, PIXEL_SIZE REAL, DOSE_PER_FRAME REAL, SPHERICAL_ABERRATION REAL, GAIN_FILENAME TEXT, OUTPUT_BINNING_FACTOR REAL, CORRECT_MAG_DISTORTION INTEGER, MAG_DISTORTION_ANGLE REAL, MAG_DISTORTION_MAJOR_SCALE REAL, MAG_DISTORTION_MINOR_SCALE REAL, PROTEIN_IS_WHITE INTEGER );
CREATE TABLE IMAGE_ASSETS(IMAGE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, POSITION_IN_STACK INTEGER, PARENT_MOVIE_ID INTEGER, ALIGNMENT_ID INTEGER, CTF_ESTIMATION_ID INTEGER, X_SIZE INTEGER, Y_SIZE INTEGER, PIXEL_SIZE REAL, VOLTAGE REAL, SPHERICAL_ABERRATION REAL, PROTEIN_IS_WHITE INTEGER );
CREATE TABLE MOVIE_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE MOVIE_ALIGNMENT_LIST(ALIGNMENT_ID INTEGER PRIMARY KEY, DATETIME_OF_RUN INTEGER, ALIGNMENT_JOB_ID INTEGER, MOVIE_ASSET_ID INTEGER, OUTPUT_FILE TEXT, VOLTAGE REAL, PIXEL_SIZE REAL, EXPOSURE_PER_FRAME REAL, PRE_EXPOSURE_AMOUNT REAL, MIN_SHIFT REAL, MAX_SHIFT REAL, SHOULD_DOSE_FILTER INTEGER, SHOULD_RESTORE_POWER INTEGER, TERMINATION_THRESHOLD REAL, MAX_ITERATIONS INTEGER, BFACTOR INTEGER, SHOULD_MASK_CENTRAL_CROSS INTEGER, HORIZONTAL_MASK INTEGER, VERTICAL_MASK INTEGER, SHOULD_INCLUDE_ALL_FRAMES_IN_SUM INTEGER, FIRST_FRAME_TO_SUM INTEGER, LAST_FRAME_TO_SUM INTEGER );
CREATE TABLE IMAGE_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE PARTICLE_PICKING_LIST(PICKING_ID INTEGER PRIMARY KEY, DATETIME_OF_RUN INTEGER, PICKING_JOB_ID INTEGER, PARENT_IMAGE_ASSET_ID INTEGER, PICKING_ALGORITHM INTEGER, CHARACTERISTIC_RADIUS REAL, MAXIMUM_RADIUS REAL, THRESHOLD_PEAK_HEIGHT REAL, HIGHEST_RESOLUTION_USED_IN_PICKING REAL, MIN_DIST_FROM_EDGES INTEGER, AVOID_HIGH_VARIANCE INTEGER, AVOID_HIGH_LOW_MEAN INTEGER, NUM_BACKGROUND_BOXES INTEGER, MANUAL_EDIT INTEGER );
CREATE TABLE PARTICLE_POSITION_ASSETS(PARTICLE_POSITION_ASSET_ID INTEGER PRIMARY KEY, PARENT_IMAGE_ASSET_ID INTEGER, PICKING_ID INTEGER, PICK_JOB_ID INTEGER, X_POSITION REAL, Y_POSITION REAL, PEAK_HEIGHT REAL, TEMPLATE_ASSET_ID INTEGER, TEMPLATE_PSI REAL, TEMPLATE_THETA REAL, TEMPLATE_PHI REAL );
CREATE TABLE PARTICLE_POSITION_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE VOLUME_ASSETS(VOLUME_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, FILENAME TEXT, RECONSTRUCTION_JOB_ID INTEGER, PIXEL_SIZE REAL, X_SIZE INTEGER, Y_SIZE INTEGER, Z_SIZE INTEGER );
CREATE TABLE VOLUME_GROUP_LIST(GROUP_ID INTEGER PRIMARY KEY, GROUP_NAME TEXT, LIST_ID INTEGER );
CREATE TABLE REFINEMENT_PACKAGE_ASSETS(REFINEMENT_PACKAGE_ASSET_ID INTEGER PRIMARY KEY, NAME TEXT, STACK_FILENAME TEXT, STACK_BOX_SIZE INTEGER, SYMMETRY TEXT, MOLECULAR_WEIGHT REAL, PARTICLE_SIZE REAL, NUMBER_OF_CLASSES INTEGER, NUMBER_OF_REFINEMENTS INTEGER, LAST_REFINEMENT_ID INTEGER, STACK_HAS_WHITE_PROTEIN INTEGER );
CREATE TABLE REFINEMENT_LIST(REFINEMENT_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT, RESOLUTION_STATISTICS_ARE_GENERATED INTEGER, DATETIME_OF_RUN INTEGER, STARTING_REFINEMENT_ID INTEGER, NUMBER_OF_PARTICLES INTEGER, NUMBER_OF_CLASSES INTEGER, RESOLUTION_STATISTICS_BOX_SIZE INTEGER, RESOLUTION_STATISTICS_PIXEL_SIZE REAL, PERCENT_USED REAL );
CREATE TABLE CLASSIFICATION_LIST(CLASSIFICATION_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT, CLASS_AVERAGE_FILE TEXT, REFINEMENT_WAS_IMPORTED_OR_GENERATED INTEGER, DATETIME_OF_RUN INTEGER, STARTING_CLASSIFICATION_ID INTEGER, NUMBER_OF_PARTICLES INTEGER, NUMBER_OF_CLASSES INTEGER, LOW_RESOLUTION_LIMIT REAL, HIGH_RESOLUTION_LIMIT REAL, MASK_RADIUS REAL, ANGULAR_SEARCH_STEP REAL, SEARCH_RANGE_X REAL, SEARCH_RANGE_Y REAL, SMOOTHING_FACTOR REAL, EXCLUDE_BLANK_EDGES INTEGER, AUTO_PERCENT_USED INTEGER, PERCENT_USED REAL );
CREATE TABLE CLASSIFICATION_SELECTION_LIST(SELECTION_ID INTEGER PRIMARY KEY, SELECTION_NAME TEXT, CREATION_DATE INTEGER, REFINEMENT_PACKAGE_ID INTEGER, CLASSIFICATION_ID INTEGER, NUMBER_OF_CLASSES INTEGER, NUMBER_OF_SELECTIONS INTEGER );
CREATE TABLE STARTUP_LIST(STARTUP_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ASSET_ID INTEGER, NAME TEXT, NUMBER_OF_STARTS INTEGER, NUMBER_OF_CYCLES INTEGER, INITIAL_RES_LIMIT REAL, FINAL_RES_LIMIT REAL, AUTO_MASK INTEGER, AUTO_PERCENT_USED INTEGER, INITIAL_PERCENT_USED REAL, FINAL_PERCENT_USED REAL, MASK_RADIUS REAL, APPLY_LIKELIHOOD_BLURRING INTEGER, SMOOTHING_FACTOR REAL );
CREATE TABLE RECONSTRUCTION_LIST(RECONSTRUCTION_ID INTEGER PRIMARY KEY, REFINEMENT_PACKAGE_ID INTEGER, REFINEMENT_ID INTEGER, NAME TEXT, INNER_MASK_RADIUS REAL, OUTER_MASK_RADIUS REAL, RESOLUTION_LIMIT REAL, SCORE_WEIGHT_CONVERSION REAL, SHOULD_ADJUST_SCORES INTEGER, SHOULD_CROP_IMAGES INTEGER, SHOULD_SAVE_HALF_MAPS INTEGER, SHOULD_LIKELIHOOD_BLUR INTEGER, SMOOTHING_FACTOR REAL, CLASS_NUMBER INTEGER, VOLUME_ASSET_ID INTEGER );
CREATE TABLE ESTIMATED_CTF_PARAMETERS(CTF_ESTIMATION_ID INTEGER PRIMARY KEY, CTF_ESTIMATION_JOB_ID INTEGER, DATETIME_OF_RUN INTEGER, IMAGE_ASSET_ID INTEGER, ESTIMATED_ON_MOVIE_FRAMES INTEGER, VOLTAGE REAL, SPHERICAL_ABERRATION REAL, PIXEL_SIZE REAL, AMPLITUDE_CONTRAST REAL, BOX_SIZE INTEGER, MIN_RESOLUTION REAL, MAX_RESOLUTION REAL, MIN_DEFOCUS REAL, MAX_DEFOCUS REAL, DEFOCUS_STEP REAL, RESTRAIN_ASTIGMATISM INTEGER, TOLERATED_ASTIGMATISM REAL, FIND_ADDITIONAL_PHASE_SHIFT INTEGER, MIN_PHASE_SHIFT REAL, MAX_PHASE_SHIFT REAL, PHASE_SHIFT_STEP REAL, DEFOCUS1 REAL, DEFOCUS2 REAL, DEFOCUS_ANGLE REAL, ADDITIONAL_PHASE_SHIFT REAL, SCORE REAL, DETECTED_RING_RESOLUTION REAL, DETECTED_ALIAS_RESOLUTION REAL, OUTPUT_DIAGNOSTIC_FILE TEXT, NUMBER_OF_FRAMES_AVERAGED INTEGER, LARGE_ASTIGMATISM_EXPECTED INTEGER );
CREATE TABLE RUN_PROFILE_COMMANDS_1(COMMANDS_NUMBER INTEGER PRIMARY KEY, COMMAND_STRING TEXT, NUMBER_OF_COPIES INTEGER, DELAY_TIME_IN_MS INTEGER );
INSERT INTO RUN_PROFILE_COMMANDS_1 VALUES(0,'/groups/cryoadmin/software/CISTEM/cistem-1.0.0-beta/$command',89,10);
CREATE TABLE PROCESS_LOCK(NUMBER INTEGER PRIMARY KEY, ACTIVE_PROCESS INTEGER, ACTIVE_HOST TEXT );
COMMIT;
)sql");
    database.Close( );
}

void MyTestApp::WriteNumericTextFile(const char* filename) {

    FILE* output_file = NULL;
    output_file       = fopen(filename, "wb+");

    if ( output_file == NULL ) {
        wxPrintf(ANSI_COLOR_RED "\n\nError: Can't open output file %s.\n", filename);
        wxPrintf(ANSI_COLOR_RESET "\n\nError: Can't open output file %s.\n", filename);
        DEBUG_ABORT;
    }

    fprintf(output_file, "# This is comment, starting with #\n");
    fprintf(output_file, "C This is comment, starting with C\n");
    fprintf(output_file, "%f %f %f %f %f\n%f %f %f %f %f\n", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.1, 8.3, 9.4, 10.5);
    fprintf(output_file, "# The next line will be blank, but contain 5 spaces\n     \n");
    fprintf(output_file, "%f %f %f %f %f\n", 11.2, 12.7, 13.2, 14.1, 15.8);
    fprintf(output_file, "   # This comment line starts with #, but not at the first character\n");
    fprintf(output_file, "   C This comment line starts with C, but not at the first character\n");
    fprintf(output_file, "C The next line will have varying spaces between the datapoints\n");
    fprintf(output_file, "   %f %f   %f       %f          %f\n", 16.1245, 17.81003, 18.5467, 19.7621, 20.11111);

    fclose(output_file);
}
