// ADDING A NEW COLUMN
// ----------------------
// See top of cistem_parameters.cpp for documentation describing how to add a new column

#ifndef SRC_CORE_CISTEM_STAR_FILE_READER_H
#define SRC_CORE_CISTEM_STAR_FILE_READER_H

class cisTEMStarFileReader {

  private:
#ifdef EXPERIMENTAL_CISTEMPARAMS
    using cp_t = cistem::parameter_names::Enum;
    std::array<int, cistem::parameter_names::count> column_positions;
#else

    int        position_in_stack_column;
    int        image_is_active_column;
    int        psi_column;
    int        theta_column;
    int        phi_column;
    int        x_shift_column;
    int        y_shift_column;
    int        defocus_1_column;
    int        defocus_2_column;
    int        defocus_angle_column;
    int        phase_shift_column;
    int        occupancy_column;
    int        logp_column;
    int        sigma_column;
    int        score_column;
    int        score_change_column;
    int        pixel_size_column;
    int        microscope_voltage_kv_column;
    int        microscope_spherical_aberration_mm_column;
    int        amplitude_contrast_column;
    int        beam_tilt_x_column;
    int        beam_tilt_y_column;
    int        image_shift_x_column;
    int        image_shift_y_column;
    int        stack_filename_column;
    int        original_image_filename_column;
    int        reference_3d_filename_column;
    int        best_2d_class_column;
    int        beam_tilt_group_column;
    int        particle_group_column;
    int        assigned_subset_column;
    int        pre_exposure_column;
    int        total_exposure_column;
#endif
    int  current_position_in_stack;
    int  current_column;
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

#ifdef EXPERIMENTAL_CISTEMPARAMS

    template <cp_t param_type>
    inline auto get(int line_number) const { return cached_parameters->Item(line_number).get<param_type>( ); }

#else
    inline int ReturnPositionInStack(int line_number) { return cached_parameters->Item(line_number).position_in_stack; }

    inline int ReturnImageIsActive(int line_number) { return cached_parameters->Item(line_number).image_is_active; }

    inline float ReturnPhi(int line_number) { return cached_parameters->Item(line_number).phi; }

    inline float ReturnTheta(int line_number) { return cached_parameters->Item(line_number).theta; }

    inline float ReturnPsi(int line_number) { return cached_parameters->Item(line_number).psi; }

    inline float ReturnXShift(int line_number) { return cached_parameters->Item(line_number).x_shift; }

    inline float ReturnYShift(int line_number) { return cached_parameters->Item(line_number).y_shift; }

    inline float ReturnDefocus1(int line_number) { return cached_parameters->Item(line_number).defocus_1; }

    inline float ReturnDefocus2(int line_number) { return cached_parameters->Item(line_number).defocus_2; }

    inline float ReturnDefocusAngle(int line_number) { return cached_parameters->Item(line_number).defocus_angle; }

    inline float ReturnPhaseShift(int line_number) { return cached_parameters->Item(line_number).phase_shift; }

    inline int ReturnLogP(int line_number) { return cached_parameters->Item(line_number).logp; }

    inline float ReturnSigma(int line_number) { return cached_parameters->Item(line_number).sigma; }

    inline float ReturnScore(int line_number) { return cached_parameters->Item(line_number).score; }

    inline float ReturnScoreChange(int line_number) { return cached_parameters->Item(line_number).score_change; }

    inline float ReturnPixelSize(int line_number) { return cached_parameters->Item(line_number).pixel_size; }

    inline float ReturnMicroscopekV(int line_number) { return cached_parameters->Item(line_number).microscope_voltage_kv; }

    inline float ReturnMicroscopeCs(int line_number) { return cached_parameters->Item(line_number).microscope_spherical_aberration_mm; }

    inline float ReturnAmplitudeContrast(int line_number) { return cached_parameters->Item(line_number).amplitude_contrast; }

    inline float ReturnBeamTiltX(int line_number) { return cached_parameters->Item(line_number).beam_tilt_x; }

    inline float ReturnBeamTiltY(int line_number) { return cached_parameters->Item(line_number).beam_tilt_y; }

    inline float ReturnImageShiftX(int line_number) { return cached_parameters->Item(line_number).image_shift_x; }

    inline float ReturnImageShiftY(int line_number) { return cached_parameters->Item(line_number).image_shift_y; }

    inline wxString ReturnStackFilename(int line_number) { return cached_parameters->Item(line_number).stack_filename; }

    inline wxString ReturnOriginalImageFilename(int line_number) { return cached_parameters->Item(line_number).original_image_filename; }

    inline wxString ReturnReference3DFilename(int line_number) { return cached_parameters->Item(line_number).reference_3d_filename; }

    inline int ReturnBest2DClass(int line_number) { return cached_parameters->Item(line_number).best_2d_class; }

    inline int ReturnBeamTiltGroup(int line_number) { return cached_parameters->Item(line_number).beam_tilt_group; }

    inline int ReturnParticleGroup(int line_number) { return cached_parameters->Item(line_number).particle_group; }

    inline int ReturnAssignedSubset(int line_number) { return cached_parameters->Item(line_number).assigned_subset; }

    inline int ReturnPreExposure(int line_number) { return cached_parameters->Item(line_number).pre_exposure; }

    inline int ReturnTotalExpsosure(int line_number) { return cached_parameters->Item(line_number).total_exposure; }
#endif

#ifdef EXPERIMENTAL_CISTEMPARAMS
    template <typename T>
    void _ParseTupleOp(T& val, const int parameter_type, const int column_position, const wxArrayString& all_tokens, wxString* error_string, bool& ret_val) {

        if ( column_position != -1 ) {
            // We have already set defaults for all -1 columns prior to getting here., Now what we depends on the type of the tuple element
            if constexpr ( std::is_integral_v<T> ) {
                long temp_long;
                if ( all_tokens[column_position].ToLong(&temp_long) == false ) {
                    MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[column_position]);
                    if ( error_string != NULL )
                        *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[column_position]);
                    ret_val = false;
                    return;
                }
                val = static_cast<T>(temp_long);
            }
            else if constexpr ( std::is_floating_point_v<T> ) {
                double temp_double;
                if ( all_tokens[column_position].ToDouble(&temp_double) == false ) {
                    MyPrintWithDetails("Error: Converting to a number (%s)\n", all_tokens[wanted_column]);
                    if ( error_string != NULL )
                        *error_string = wxString::Format("Error: Converting to a number (%s)\n", all_tokens[wanted_column]);
                    ret_val = false;
                    return;
                }
                val = static_cast<T>(temp_double);
            }
            else if constepxr (std::is_same_v<T, wxString>) {
                wxString temp_string = all_tokens[column_position].Trim(true).Trim(false);
                if ( StripEnclosingSingleQuotesFromString(temp_string) == false ) {
                    switch ( parameter_type ) {
                        case (cistem::parameter_names::stack_filename):
                            MyPrintfRed("Error: stack file name read as %s is not enclosed in single quotes ('), replacing with blank string\n", temp_string);
                            temp_parameters.stack_filename(wxEmptyString);
                            break;
                        case (cistem::parameter_names::reference_3d):
                            MyPrintfRed("Error: reference 3D file name read as %s is not enclosed in single quotes ('), replacing with blank string\n", temp_string);
                            temp_parameters.reference_3d(wxEmptyString);
                            break;
                        case (cistem::parameter_names::original_image):
                            MyPrintfRed("Error: original image file name read as %s is not enclosed in single quotes ('), replacing with blank string\n", temp_string);
                            temp_parameters.original_image(wxEmptyString);
                            break;
                        default:
                            MyPrintfRed("Error: unknown parameter type %i\n", parameter_type);
                            break;
                    }
                    val = wxEmptyString;
                }
                // Unlike the numeric types, the original code does not set any value as in the next line. I'm not sure if that is the desired behavior.
                // val = temp_string;
            }

            else {
                MyDebugAssertTrue(false, "Unknown type for cistem parameter element")
            }
        }
    }

    template <typename TupleT, std::size_t... Is>
    void _For_Each_Tuple_Parse_Column_impl(TupleT& tp, const std::array<int, cistem::parameter_names::count>& column_positions, const wxArrayString& all_tokens, wxString* error_string, bool& ret_val, std::index_sequence<Is...>) {
        // Use a fold expression to call the tuple Op on each element of the tuple pair
        // Note: Is is passed to use as the parameter name to disambiguate string type
        (_ParseTupleOp(std::get<Is>(tp), Is, column_positions.at(Is), all_tokens, error_string, ret_val), ...);
    }

    template <typename TupleT, std::size_t TupSize = std::tuple_size_v<TupleT>>
    bool For_Each_Tuple_Parse_Column(TupleT& tp, const std::array<int, cistem::parameter_names::count>& column_positions, const wxArrayString& all_tokens, wxString* error_string) {
        // Assuming this is called from cistem_staf_file_reader::ExtractParametersFromLine, maybe it should be amethod in that header?
        // Assuming the default values have been set for all parameters where column positions = -1
        bool ret_val = true;
        _For_Each_Tuple_Parse_Column_impl(tp, column_positions, all_tokens, error_string, ret_val, std::make_index_sequence<TupSize>{ });
        return ret_val;
    }

    //////////////////////////////////////
    ////////////////////////////////////////

#endif
};

#endif