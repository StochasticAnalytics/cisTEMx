/*  \brief  class for cisTEM parameters */

#ifndef _src_core_cistem_parameters_h_
#define _src_core_cistem_parameters_h_

#ifdef EXPERIMENTAL_CISTEMPARAMS
#include "../constants/constants.h"
#else
#define POSITION_IN_STACK 1
#define IMAGE_IS_ACTIVE 2
#define PSI 4
#define X_SHIFT 8
#define Y_SHIFT 16
#define DEFOCUS_1 32
#define DEFOCUS_2 64
#define DEFOCUS_ANGLE 128
#define PHASE_SHIFT 256
#define OCCUPANCY 512
#define LOGP 1024
#define SIGMA 2048
#define SCORE 4096
#define SCORE_CHANGE 8192
#define PIXEL_SIZE 16384
#define MICROSCOPE_VOLTAGE 32768
#define MICROSCOPE_CS 65536
#define AMPLITUDE_CONTRAST 131072
#define BEAM_TILT_X 262144
#define BEAM_TILT_Y 524288
#define IMAGE_SHIFT_X 1048576
#define IMAGE_SHIFT_Y 2097152
#define THETA 4194304
#define PHI 8388608
#define STACK_FILENAME 16777216
#define ORIGINAL_IMAGE_FILENAME 33554432
#define REFERENCE_3D_FILENAME 67108864
#define BEST_2D_CLASS 134217728
#define BEAM_TILT_GROUP 268435456
#define PARTICLE_GROUP 536870912
#define PRE_EXPOSURE 1073741824
#define TOTAL_EXPOSURE 2147483648
#define ASSIGNED_SUBSET 4294967296

#endif

// ADDING A NEW COLUMN
// ----------------------
// See top of cistem_parameters.cpp for documentation describing how to add a new column

class cisTEMParameterLine {

#ifdef EXPERIMENTAL_CISTEMPARAMS
    // Rather than member variables, we are going to make a cisTEM parameter line an array of tuples corresponding to
    // cistem::paramter_names::Enum, value, is_active (bool), which combines the functionality of the old cisTEMParameterLine and cisTEMParameterMask

    // The existing constructor setz all values to zero, and sets all is_active to false
    // The existing destructor does nothing which also means that no copy or move constructors are created.
    // Instead, we'll initialize the tuples on default construction and let the compiler generate the copy and move constructors

  private:
    using cp_t = cistem::parameter_names::Enum;

  public:
    std::tuple<unsigned int, // position_in_stack
               int, // image_is_active
               float, // psi
               float, // theta
               float, // phi
               float, // x_shift
               float, // y_shift
               float, // defocus_1
               float, // defocus_2
               float, // defocus_angle
               float, // phase_shift
               float, // occupancy
               float, // logp
               float, // sigma
               float, // score
               float, // score_change
               float, // pixel_size
               float, // microscope_voltage_kv
               float, // microscope_spherical_aberration_mm
               float, // amplitude_contrast
               float, // beam_tilt_x
               float, // beam_tilt_y
               float, // image_shift_x
               float, // image_shift_y
               wxString, // stack_filename
               wxString, // original_image_filename
               wxString, // reference_3d_filename
               int, // best_2d_class
               int, // beam_tilt_group
               int, // particle_group
               int, // assigned_subset
               float, // pre_exposure
               float // total_exposure
               >
            values;

  public:
    cisTEMParameterLine( )  = default;
    ~cisTEMParameterLine( ) = default;

    // template <cistem::parameter_names::Enum E>
    // inline auto get( ) const { return std::get<E>(values); }

    template <cistem::parameter_names::Enum E>
    inline auto& get( ) { return std::get<E>(values); }

    template <cistem::parameter_names::Enum E, typename T>
    inline void set(T wanted_value) { std::get<E>(values) = wanted_value; }

    // Check that the number of elements defined in constants.h matches the number here. This at least helps make it a little less brittle.
    static_assert(cistem::parameter_names::count == std::tuple_size_v<decltype(values)>, "cisTEMParameterLine::values tuple size does not match cistem::parameter_names::count");

#else

  public:
    unsigned int position_in_stack;
    int          image_is_active;
    float        psi;
    float        theta;
    float        phi;
    float        x_shift;
    float        y_shift;
    float        defocus_1;
    float        defocus_2;
    float        defocus_angle;
    float        phase_shift;
    float        occupancy;
    float        logp;
    float        sigma;
    float        score;
    float        score_change;
    float        pixel_size;
    float        microscope_voltage_kv;
    float        microscope_spherical_aberration_mm;
    float        amplitude_contrast;
    float        beam_tilt_x;
    float        beam_tilt_y;
    float        image_shift_x;
    float        image_shift_y;
    wxString     stack_filename;
    wxString     original_image_filename;
    wxString     reference_3d_filename;
    int          best_2d_class;
    int          beam_tilt_group; // identify particles expected to have the same beam tilt parameters
    int          particle_group; // identify images of the same particle (all images of a given particle should have the same PARTICLE_GROUP number. E.g. across a tilt-series or movie, i.e. a frame-series))
    int          assigned_subset; // used for example to assign particles to half-datasets, half-maps for the purposes of FSCs
    float        pre_exposure;
    float        total_exposure;

    cisTEMParameterLine( );
    ~cisTEMParameterLine( );
#endif
    //void SwapPsiAndPhi(); shouldn't need
    void Add(cisTEMParameterLine& line_to_add);
    void Subtract(cisTEMParameterLine& line_to_subtract);
    void AddSquare(cisTEMParameterLine& line_to_add);

    void SetAllToZero( );
    void SetAllToDefault( );
    void SetAllToDefault(std::array<int, cistem::parameter_names::count>& column_positions);

    void ReplaceNanAndInfWithOther(cisTEMParameterLine& other_params);
};

class cisTEMParameterMask {

#ifdef EXPERIMENTAL_CISTEMPARAMS

  private:
    using cp_t = cistem::parameter_names::Enum;

  public:
    std::array<bool, cistem::parameter_names::count> is_active;

    template <cp_t E>
    inline bool get( ) const { return is_active.at[E]; }

    template <cp_t E>
    inline void set(bool wanted_bool) { is_active.at(E) = wanted_bool; }

    void SetAllToTrue( );
    void SetAllToFalse( );
    void SetActiveParameters(std::vector<cp_t>& wanted_active_parameters); // uses takes the defines above bitwise

    int ReturnNumberOfActiveParameters( ) { return std::count(is_active.begin( ), is_active.end( ), true); }

    cisTEMParameterMask( );

#else

  public:
    bool position_in_stack;
    bool image_is_active;
    bool psi;
    bool theta;
    bool phi;
    bool x_shift;
    bool y_shift;
    bool defocus_1;
    bool defocus_2;
    bool defocus_angle;
    bool phase_shift;
    bool occupancy;
    bool logp;
    bool sigma;
    bool score;
    bool score_change;
    bool pixel_size;
    bool microscope_voltage_kv;
    bool microscope_spherical_aberration_mm;
    bool amplitude_contrast;
    bool beam_tilt_x;
    bool beam_tilt_y;
    bool image_shift_x;
    bool image_shift_y;
    bool stack_filename;
    bool original_image_filename;
    bool reference_3d_filename;
    bool best_2d_class;
    bool beam_tilt_group;
    bool particle_group;
    bool assigned_subset;
    bool pre_exposure;
    bool total_exposure;

    void SetAllToTrue( );
    void SetAllToFalse( );
    void SetActiveParameters(long parameters_to_set); // uses takes the defines above bitwise

    cisTEMParameterMask( );
#endif
};

WX_DECLARE_OBJARRAY(cisTEMParameterLine, ArrayOfcisTEMParameterLines);

class cisTEMParameters {

  private:
    using cp_t = cistem::parameter_names::Enum;

  public:
    wxArrayString               header_comments;
    ArrayOfcisTEMParameterLines all_parameters;

    cisTEMParameterMask parameters_to_write;
    cisTEMParameterMask parameters_that_were_read;

    // for defocus dependance

    float average_defocus;
    float defocus_coeff_a;
    float defocus_coeff_b;

    cisTEMParameters( );
    ~cisTEMParameters( );

    void ReadFromcisTEMStarFile(wxString wanted_filename, bool exclude_negative_film_numbers = false);
    void ReadFromcisTEMBinaryFile(wxString wanted_filename, bool exclude_negative_film_numbers = false);

    void ReadFromFrealignParFile(wxString wanted_filename,
                                 float    wanted_pixel_size         = 0.0f,
                                 float    wanted_microscope_voltage = 0.0f,
                                 float    wanted_microscope_cs      = 0.0f,
                                 float    wanted_amplitude_contrast = 0.0f,
                                 float    wanted_beam_tilt_x        = 0.0f,
                                 float    wanted_beam_tilt_y        = 0.0f,
                                 float    wanted_image_shift_x      = 0.0f,
                                 float    wanted_image_shift_y      = 0.0f,
                                 int      wanted_particle_group     = 1,
                                 float    wanted_pre_exposure       = 0.0f,
                                 float    wanted_total_exposure     = 0.1f);

    int  ReturnNumberOfParametersToWrite( );
    int  ReturnNumberOfLinesToWrite(int first_image_to_write, int last_image_to_write);
    void WriteTocisTEMBinaryFile(wxString wanted_filename, int first_image_to_write = -1, int last_image_to_write = -1);

    void AddCommentToHeader(wxString comment_to_add);
    void WriteTocisTEMStarFile(wxString wanted_filename, int first_line_to_write = -1, int last_line_to_write = -1, int first_image_to_write = -1, int last_image_to_write = -1);

    void ClearAll( );

    void PreallocateMemoryAndBlank(int number_to_allocate);

    inline long ReturnNumberofLines( ) { return all_parameters.GetCount( ); }

    inline cisTEMParameterLine ReturnLine(int line_number) { return all_parameters.Item(line_number); }

#ifdef EXPERIMENTAL_CISTEMPARAMS

    template <cp_t param_type>
    inline auto get(int line_number) const { return all_parameters.Item(line_number).get<param_type>( ); }

    template <cp_t param_type, typename T>
    inline void set(int line_number, T wanted_value) { all_parameters.Item(line_number).get<param_type>( ) = wanted_value; }
#else

    inline int ReturnPositionInStack(int line_number) { return all_parameters.Item(line_number).position_in_stack; }

    inline int ReturnImageIsActive(int line_number) { return all_parameters.Item(line_number).image_is_active; }

    inline float ReturnPhi(int line_number) { return all_parameters.Item(line_number).phi; }

    inline float ReturnTheta(int line_number) { return all_parameters.Item(line_number).theta; }

    inline float ReturnPsi(int line_number) { return all_parameters.Item(line_number).psi; }

    inline float ReturnXShift(int line_number) { return all_parameters.Item(line_number).x_shift; }

    inline float ReturnYShift(int line_number) { return all_parameters.Item(line_number).y_shift; }

    inline float ReturnDefocus1(int line_number) { return all_parameters.Item(line_number).defocus_1; }

    inline float ReturnDefocus2(int line_number) { return all_parameters.Item(line_number).defocus_2; }

    inline float ReturnDefocusAngle(int line_number) { return all_parameters.Item(line_number).defocus_angle; }

    inline float ReturnPhaseShift(int line_number) { return all_parameters.Item(line_number).phase_shift; }

    inline float ReturnOccupancy(int line_number) { return all_parameters.Item(line_number).occupancy; }

    inline float ReturnLogP(int line_number) { return all_parameters.Item(line_number).logp; }

    inline float ReturnSigma(int line_number) { return all_parameters.Item(line_number).sigma; }

    inline float ReturnScore(int line_number) { return all_parameters.Item(line_number).score; }

    inline float ReturnScoreChange(int line_number) { return all_parameters.Item(line_number).score_change; }

    inline float ReturnPixelSize(int line_number) { return all_parameters.Item(line_number).pixel_size; }

    inline float ReturnMicroscopekV(int line_number) { return all_parameters.Item(line_number).microscope_voltage_kv; }

    inline float ReturnMicroscopeCs(int line_number) { return all_parameters.Item(line_number).microscope_spherical_aberration_mm; }

    inline float ReturnAmplitudeContrast(int line_number) { return all_parameters.Item(line_number).amplitude_contrast; }

    inline float ReturnBeamTiltX(int line_number) { return all_parameters.Item(line_number).beam_tilt_x; }

    inline float ReturnBeamTiltY(int line_number) { return all_parameters.Item(line_number).beam_tilt_y; }

    inline float ReturnImageShiftX(int line_number) { return all_parameters.Item(line_number).image_shift_x; }

    inline float ReturnImageShiftY(int line_number) { return all_parameters.Item(line_number).image_shift_y; }

    inline wxString ReturnStackFilename(int line_number) { return all_parameters.Item(line_number).stack_filename; }

    inline wxString ReturnOriginalImageFilename(int line_number) { return all_parameters.Item(line_number).original_image_filename; }

    inline wxString ReturnReference3DFilename(int line_number) { return all_parameters.Item(line_number).reference_3d_filename; }

    inline int ReturnBest2DClass(int line_number) { return all_parameters.Item(line_number).best_2d_class; }

    inline int ReturnBeamTiltGroup(int line_number) { return all_parameters.Item(line_number).beam_tilt_group; }

    inline int ReturnParticleGroup(int line_number) { return all_parameters.Item(line_number).particle_group; }

    inline int ReturnAssignedSubset(int line_number) { return all_parameters.Item(line_number).assigned_subset; }

    inline float ReturnPreExposure(int line_number) { return all_parameters.Item(line_number).pre_exposure; }

    inline float ReturnTotalExposure(int line_number) { return all_parameters.Item(line_number).total_exposure; }
#endif
    float ReturnAverageSigma(bool exclude_negative_film_numbers = false);
    float ReturnAverageOccupancy(bool exclude_negative_film_numbers = false);
    float ReturnAverageScore(bool exclude_negative_film_numbers = false);

    bool ContainsMultipleParticleGroups( );

    void RemoveSigmaOutliers(float wanted_standard_deviation, bool exclude_negative_film_numbers = false, bool reciprocal_square = false);
    void RemoveScoreOutliers(float wanted_standard_deviation, bool exclude_negative_film_numbers = false, bool reciprocal_square = false);

    void  CalculateDefocusDependence(bool exclude_negative_film_numbers = false);
    void  AdjustScores(bool exclude_negative_film_numbers = false);
    float ReturnScoreAdjustment(float defocus);
    float ReturnScoreThreshold(float wanted_percentage, bool exclude_negative_film_numbers = false);

    float ReturnMinScore(bool exclude_negative_film_numbers = false);
    float ReturnMaxScore(bool exclude_negative_film_numbers = false);
    int   ReturnMinPositionInStack(bool exclude_negative_film_numbers = false);
    int   ReturnMaxPositionInStack(bool exclude_negative_film_numbers = false);

    void SetAllReference3DFilename(wxString wanted_filename);
    void SortByReference3DFilename( );

    cisTEMParameterLine ReturnParameterAverages(bool only_average_active = true);
    cisTEMParameterLine ReturnParameterVariances(bool only_average_active = true);

#ifdef EXPERIMENTAL_CISTEMPARAMS
    template <typename T>
    void _WriteParamToHeader(T& val, long cp_t, bool is_active, FILE* cistem_binary_file) {
        if ( is_active ) {
            char data_type;
            // convert to legacy cisTEM type define macros
            if constexpr ( std::is_same_v<unsigned int, T> ) {
                data_type = INTEGER_UNSIGNED
            }
            else if constexpr ( std::is_same_v<int, T> ) {
                data_type = INTEGER;
            }
            else if constexpr ( std::is_same_v<float, T> ) {
                data_type = FLOAT;
            }
            else if constexpr ( std::is_same_v<wxString, T> ) {
                data_type = VARIABLE_LENGTH;
            }
            else {
                throw std::runtime_error("Unknown type");
            }
            fwrite(&cp_t, sizeof(long), 1, cisTEM_binary_file);
            fwrite(&data_type, sizeof(char), 1, cisTEM_binary_file);
        }
    }

    // These are specialized in the .cpp file to capture the correct type
    template <typename TupleT, std::size_t... Is>
    void _For_Each_Tuple_Write_Param_Header_impl(TupleT& tp, std::array<int, cistem::parameter_names::count>& is_active, FILE* cisTEM_binary_star_file, std::index_sequence<Is...>) {
        // Use a fold expression to call the tuple Op on each element of the tuple pair
        // Note: Is is passed to use as the parameter name to disambiguate string type
        (_WriteParamToHeader(std::get<Is>(tp), Is, is_active.at(Is), cisTEM_binary_star_file), ...);
        //<std::tuple_element<Is, TupleT>
    }

    template <typename TupleT, std::size_t TupSize = std::tuple_size_v<TupleT>>
    void For_Each_Tuple_Write_Param_Header(TupleT& tp, std::array<bool, cistem::parameter_names::count>& is_active, FILE* cisTEM_binary_star_file) {
        _For_Each_Tuple_Write_Param_Header_impl(tp, is_active, cisTEM_binary_star_file, std::make_index_sequence<TupSize>{ });
    }

    /////////////////////////////////
    template <typename T>
    void _WriteParamToBinary(T& val, bool is_active, FILE* cisTEM_binary_star_file) {
        // The only specialization we need is for strings.
        if ( is_active ) {
            if constexpr ( std::is_same_v<T, wxString> ) {
                size_t length_of_string = val.Length( );
                fwrite(&length_of_string, sizeof(size_t), 1, cisTEM_binary_star_file);
                fwrite(val.ToStdString( ).c_str( ), length_of_string * sizeof(char), 1, cisTEM_binary_star_file);
            }
            else {
                fwrite(&val, sizeof(T), 1, cisTEM_binary_star_file);
            }
        }
    }

    // These are specialized in the .cpp file to capture the correct type
    template <typename TupleT, std::size_t... Is>
    void _For_Each_Tuple_Write_Param_To_Binary_impl(TupleT& tp, std::array<int, cistem::parameter_names::count>& is_active, FILE* cisTEM_binary_star_file, std::index_sequence<Is...>) {
        // Use a fold expression to call the tuple Op on each element of the tuple pair
        // Note: Is is passed to use as the parameter name to disambiguate string type
        (_WriteParamToBinary(std::get<Is>(tp), is_active.at(Is), cisTEM_binary_star_file), ...);
        //<std::tuple_element<Is, TupleT>
    }

    template <typename TupleT, std::size_t TupSize = std::tuple_size_v<TupleT>>
    void For_Each_Tuple_Write_Param_To_Binary(TupleT& tp, std::array<bool, cistem::parameter_names::count>& is_active, FILE* cisTEM_binary_star_file) {
        _For_Each_Tuple_Write_Param_To_Binary_impl(tp, is_active, cisTEM_binary_star_file, std::make_index_sequence<TupSize>{ });
    }
#endif
};

#endif