// ADDING A NEW COLUMN
// ----------------------
// See top of cistem_parameters.cpp for documentation describing how to add a new column

#ifndef SRC_CORE_CISTEM_STAR_FILE_READER_H
#define SRC_CORE_CISTEM_STAR_FILE_READER_H

class cisTEMStarFileReader {

  private:
    using param_t = cistem::parameter_names::Enum;

    int current_position_in_stack;
    int current_column;

    int position_in_stack_column;
    int image_is_active_column;
    int psi_column;
    int theta_column;
    int phi_column;
    int x_shift_column;
    int y_shift_column;
    int defocus_1_column;
    int defocus_2_column;
    int defocus_angle_column;
    int phase_shift_column;
    int occupancy_column;
    int logp_column;
    int sigma_column;
    int score_column;
    int score_change_column;
    int pixel_size_column;
    int microscope_voltage_kv_column;
    int microscope_spherical_aberration_mm_column;
    int amplitude_contrast_column;
    int beam_tilt_x_column;
    int beam_tilt_y_column;
    int image_shift_x_column;
    int image_shift_y_column;
    int stack_filename_column;
    int original_image_filename_column;
    int reference_3d_filename_column;
    int best_2d_class_column;
    int beam_tilt_group_column;
    int particle_group_column;
    int assigned_subset_column;
    int pre_exposure_column;
    int total_exposure_column;

    long binary_buffer_position;

    // The following "Safely" functions are to read data from the buffer with error checking to make sure there is no segfault

    inline bool SafelyReadFromBinaryBufferIntoInteger(int& integer_to_read_into) {
        if ( binary_buffer_position + sizeof(int) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short");
            return false;
        }

        int* temp_int_pointer = reinterpret_cast<int*>(&binary_file_read_buffer[binary_buffer_position]);
        integer_to_read_into  = *temp_int_pointer;
        binary_buffer_position += sizeof(int);
        return true;
    }

    inline bool SafelyReadFromBinaryBufferIntoUnsignedInteger(unsigned int& unsigned_integer_to_read_into) {
        if ( binary_buffer_position + sizeof(unsigned int) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short");
            return false;
        }

        unsigned int* temp_int_pointer = reinterpret_cast<unsigned int*>(&binary_file_read_buffer[binary_buffer_position]);
        unsigned_integer_to_read_into  = *temp_int_pointer;
        binary_buffer_position += sizeof(unsigned int);
        return true;
    }

    inline bool SafelyReadFromBinaryBufferIntoFloat(float& float_to_read_into) {
        if ( binary_buffer_position + sizeof(float) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short\n");
            return false;
        }

        float* temp_float_pointer = reinterpret_cast<float*>(&binary_file_read_buffer[binary_buffer_position]);
        float_to_read_into        = *temp_float_pointer;
        binary_buffer_position += sizeof(float);
        return true;
    }

    inline bool SafelyReadFromBinaryBufferIntoLong(long& long_to_read_into) {
        if ( binary_buffer_position + sizeof(long) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short\n");
            return false;
        }

        long* temp_long_pointer = reinterpret_cast<long*>(&binary_file_read_buffer[binary_buffer_position]);
        long_to_read_into       = *temp_long_pointer;
        binary_buffer_position += sizeof(long);
        return true;
    }

    inline bool SafelyReadFromBinaryBufferIntoChar(char& char_to_read_into) {
        if ( binary_buffer_position + sizeof(char) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short\n");
            return false;
        }

        char* temp_long_pointer = &binary_file_read_buffer[binary_buffer_position];
        char_to_read_into       = *temp_long_pointer;
        binary_buffer_position += sizeof(char);
        return true;
    }

    inline bool SafelyReadFromBinaryBufferIntoDouble(double& double_to_read_into) {
        if ( binary_buffer_position + sizeof(double) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short\n");
            return false;
        }

        double* temp_double_pointer = reinterpret_cast<double*>(&binary_file_read_buffer[binary_buffer_position]);
        double_to_read_into         = *temp_double_pointer;
        binary_buffer_position += sizeof(double);
        return true;
    }

    inline bool SafelyReadFromBinaryBufferIntowxString(wxString& wxstring_to_read_into) {
        int length_of_string;
        if ( SafelyReadFromBinaryBufferIntoInteger(length_of_string) == false )
            return false;

        if ( length_of_string < 0 ) {
            MyPrintWithDetails("Error Reading string, length is %i", length_of_string);
        }

        if ( binary_buffer_position + length_of_string * sizeof(char) - 1 >= binary_file_size ) {
            MyPrintWithDetails("Error: Binary file is too short\n");
            return false;
        }

        char string_buffer[length_of_string + 1];

        for ( int array_counter = 0; array_counter < length_of_string; array_counter++ ) {
            string_buffer[array_counter] = binary_file_read_buffer[binary_buffer_position + array_counter];
        }

        string_buffer[length_of_string] = 0;
        wxstring_to_read_into           = string_buffer;

        binary_buffer_position += sizeof(char) * length_of_string;
        return true;
    }

  public:
    wxString    filename;
    wxTextFile* input_text_file;

    char* binary_file_read_buffer;
    long  binary_file_size;

    bool using_external_array;

    ArrayOfcisTEMParameterLines* cached_parameters;
    cisTEMParameterMask          parameters_that_were_read;

    cisTEMStarFileReader( );
    ~cisTEMStarFileReader( );

    cisTEMStarFileReader(wxString wanted_filename, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);

    void Open(wxString wanted_filename, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer = NULL, bool read_as_binary = false);
    void Close( );
    bool ReadTextFile(wxString wanted_filename, wxString* error_string = NULL, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);
    bool ReadBinaryFile(wxString wanted_filename, ArrayOfcisTEMParameterLines* alternate_cached_parameters_pointer = NULL, bool exclude_negative_film_numbers = false);

    bool ExtractParametersFromLine(wxString& wanted_line, wxString* error_string = NULL, bool exclude_negative_film_numbers = false);
    void Reset( );
    void ResetColumnPositions( );

    inline int ReturnPositionInStack(int line_number) { return std::get<param_t::position_in_stack>(cached_parameters->Item(line_number).values); }

    inline int ReturnImageIsActive(int line_number) { return std::get<param_t::image_is_active>(cached_parameters->Item(line_number).values); }

    inline float ReturnPhi(int line_number) { return std::get<param_t::phi>(cached_parameters->Item(line_number).values); }

    inline float ReturnTheta(int line_number) { return std::get<param_t::theta>(cached_parameters->Item(line_number).values); }

    inline float ReturnPsi(int line_number) { return std::get<param_t::psi>(cached_parameters->Item(line_number).values); }

    inline float ReturnXShift(int line_number) { return std::get<param_t::x_shift>(cached_parameters->Item(line_number).values); }

    inline float ReturnYShift(int line_number) { return std::get<param_t::y_shift>(cached_parameters->Item(line_number).values); }

    inline float ReturnDefocus1(int line_number) { return std::get<param_t::defocus_1>(cached_parameters->Item(line_number).values); }

    inline float ReturnDefocus2(int line_number) { return std::get<param_t::defocus_2>(cached_parameters->Item(line_number).values); }

    inline float ReturnDefocusAngle(int line_number) { return std::get<param_t::defocus_angle>(cached_parameters->Item(line_number).values); }

    inline float ReturnPhaseShift(int line_number) { return std::get<param_t::phase_shift>(cached_parameters->Item(line_number).values); }

    inline int ReturnLogP(int line_number) { return std::get<param_t::logp>(cached_parameters->Item(line_number).values); }

    inline float ReturnSigma(int line_number) { return std::get<param_t::sigma>(cached_parameters->Item(line_number).values); }

    inline float ReturnScore(int line_number) { return std::get<param_t::score>(cached_parameters->Item(line_number).values); }

    inline float ReturnScoreChange(int line_number) { return std::get<param_t::score_change>(cached_parameters->Item(line_number).values); }

    inline float ReturnPixelSize(int line_number) { return std::get<param_t::pixel_size>(cached_parameters->Item(line_number).values); }

    inline float ReturnMicroscopekV(int line_number) { return std::get<param_t::microscope_voltage_kv>(cached_parameters->Item(line_number).values); }

    inline float ReturnMicroscopeCs(int line_number) { return std::get<param_t::microscope_spherical_aberration_mm>(cached_parameters->Item(line_number).values); }

    inline float ReturnAmplitudeContrast(int line_number) { return std::get<param_t::amplitude_contrast>(cached_parameters->Item(line_number).values); }

    inline float ReturnBeamTiltX(int line_number) { return std::get<param_t::beam_tilt_x>(cached_parameters->Item(line_number).values); }

    inline float ReturnBeamTiltY(int line_number) { return std::get<param_t::beam_tilt_y>(cached_parameters->Item(line_number).values); }

    inline float ReturnImageShiftX(int line_number) { return std::get<param_t::image_shift_x>(cached_parameters->Item(line_number).values); }

    inline float ReturnImageShiftY(int line_number) { return std::get<param_t::image_shift_y>(cached_parameters->Item(line_number).values); }

    inline wxString ReturnStackFilename(int line_number) { return std::get<param_t::stack_filename>(cached_parameters->Item(line_number).values); }

    inline wxString ReturnOriginalImageFilename(int line_number) { return std::get<param_t::original_image_filename>(cached_parameters->Item(line_number).values); }

    inline wxString ReturnReference3DFilename(int line_number) { return std::get<param_t::reference_3d_filename>(cached_parameters->Item(line_number).values); }

    inline int ReturnBest2DClass(int line_number) { return std::get<param_t::best_2d_class>(cached_parameters->Item(line_number).values); }

    inline int ReturnBeamTiltGroup(int line_number) { return std::get<param_t::beam_tilt_group>(cached_parameters->Item(line_number).values); }

    inline int ReturnParticleGroup(int line_number) { return std::get<param_t::particle_group>(cached_parameters->Item(line_number).values); }

    inline int ReturnAssignedSubset(int line_number) { return std::get<param_t::assigned_subset>(cached_parameters->Item(line_number).values); }

    inline int ReturnPreExposure(int line_number) { return std::get<param_t::pre_exposure>(cached_parameters->Item(line_number).values); }

    inline int ReturnTotalExpsosure(int line_number) { return std::get<param_t::total_exposure>(cached_parameters->Item(line_number).values); }
};

#endif