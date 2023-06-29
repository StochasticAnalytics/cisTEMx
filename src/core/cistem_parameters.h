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

  public:
#ifdef EXPERIMENTAL_CISTEMPARAMS
    // Rather than member variables, we are going to make a cisTEM parameter line an array of tuples corresponding to
    // cistem::paramter_names::Enum, value, is_active (bool), which combines the functionality of the old cisTEMParameterLine and cisTEMParameterMask

    // The existing constructor setz all values to zero, and sets all is_active to false
    // The existing destructor does nothing which also means that no copy or move constructors are created.
    // Instead, we'll initialize the tuples on default construction and let the compiler generate the copy and move constructors

  public:
    cisTEMParameterLine( )  = default;
    ~cisTEMParameterLine( ) = default;

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

    // Check that the number of elements defined in constants.h matches the number here. This at least helps make it a little less brittle.
    static_assert(cistem::parameter_names::count == std::tuple_size_v<decltype(values)>, "cisTEMParameterLine::values tuple size does not match cistem::parameter_names::count");

    // Define getters for each parameter
    inline unsigned int position_in_stack( ) const { return std::get<cistem::parameter_names::position_in_stack>(values); }

    inline int image_is_active( ) const { return std::get<cistem::parameter_names::image_is_active>(values); }

    inline float psi( ) const { return std::get<cistem::parameter_names::psi>(values); }

    inline float theta( ) const { return std::get<cistem::parameter_names::theta>(values); }

    inline float phi( ) const { return std::get<cistem::parameter_names::phi>(values); }

    inline float x_shift( ) const { return std::get<cistem::parameter_names::x_shift>(values); }

    inline float y_shift( ) const { return std::get<cistem::parameter_names::y_shift>(values); }

    inline float defocus_1( ) const { return std::get<cistem::parameter_names::defocus_1>(values); }

    inline float defocus_2( ) const { return std::get<cistem::parameter_names::defocus_2>(values); }

    inline float defocus_angle( ) const { return std::get<cistem::parameter_names::defocus_angle>(values); }

    inline float phase_shift( ) const { return std::get<cistem::parameter_names::phase_shift>(values); }

    inline float occupancy( ) const { return std::get<cistem::parameter_names::occupancy>(values); }

    inline float logp( ) const { return std::get<cistem::parameter_names::logp>(values); }

    inline float sigma( ) const { return std::get<cistem::parameter_names::sigma>(values); }

    inline float score( ) const { return std::get<cistem::parameter_names::score>(values); }

    inline float score_change( ) const { return std::get<cistem::parameter_names::score_change>(values); }

    inline float pixel_size( ) const { return std::get<cistem::parameter_names::pixel_size>(values); }

    inline float microscope_voltage_kv( ) const { return std::get<cistem::parameter_names::microscope_voltage_kv>(values); }

    inline float microscope_spherical_aberration_mm( ) const { return std::get<cistem::parameter_names::microscope_spherical_aberration_mm>(values); }

    inline float amplitude_contrast( ) const { return std::get<cistem::parameter_names::amplitude_contrast>(values); }

    inline float beam_tilt_x( ) const { return std::get<cistem::parameter_names::beam_tilt_x>(values); }

    inline float beam_tilt_y( ) const { return std::get<cistem::parameter_names::beam_tilt_y>(values); }

    inline float image_shift_x( ) const { return std::get<cistem::parameter_names::image_shift_x>(values); }

    inline float image_shift_y( ) const { return std::get<cistem::parameter_names::image_shift_y>(values); }

    inline wxString stack_filename( ) const { return std::get<cistem::parameter_names::stack_filename>(values); }

    inline wxString original_image_filename( ) const { return std::get<cistem::parameter_names::original_image_filename>(values); }

    inline wxString reference_3d_filename( ) const { return std::get<cistem::parameter_names::reference_3d_filename>(values); }

    inline int best_2d_class( ) const { return std::get<cistem::parameter_names::best_2d_class>(values); }

    inline int beam_tilt_group( ) const { return std::get<cistem::parameter_names::beam_tilt_group>(values); }

    inline int particle_group( ) const { return std::get<cistem::parameter_names::particle_group>(values); }

    inline int assigned_subset( ) const { return std::get<cistem::parameter_names::assigned_subset>(values); }

    inline float pre_exposure( ) const { return std::get<cistem::parameter_names::pre_exposure>(values); }

    inline float total_exposure( ) const { return std::get<cistem::parameter_names::total_exposure>(values); }

    // Define setters for each parameter
    inline void position_in_stack(unsigned int wanted_val) { std::get<cistem::parameter_names::position_in_stack>(values) = wanted_val; }

    inline void image_is_active(int wanted_val) { std::get<cistem::parameter_names::image_is_active>(values) = wanted_val; }

    inline void psi(float wanted_val) { std::get<cistem::parameter_names::psi>(values) = wanted_val; }

    inline void theta(float wanted_val) { std::get<cistem::parameter_names::theta>(values) = wanted_val; }

    inline void phi(float wanted_val) { std::get<cistem::parameter_names::phi>(values) = wanted_val; }

    inline void x_shift(float wanted_val) { std::get<cistem::parameter_names::x_shift>(values) = wanted_val; }

    inline void y_shift(float wanted_val) { std::get<cistem::parameter_names::y_shift>(values) = wanted_val; }

    inline void defocus_1(float wanted_val) { std::get<cistem::parameter_names::defocus_1>(values) = wanted_val; }

    inline void defocus_2(float wanted_val) { std::get<cistem::parameter_names::defocus_2>(values) = wanted_val; }

    inline void defocus_angle(float wanted_val) { std::get<cistem::parameter_names::defocus_angle>(values) = wanted_val; }

    inline void phase_shift(float wanted_val) { std::get<cistem::parameter_names::phase_shift>(values) = wanted_val; }

    inline void occupancy(float wanted_val) { std::get<cistem::parameter_names::occupancy>(values) = wanted_val; }

    inline void logp(float wanted_val) { std::get<cistem::parameter_names::logp>(values) = wanted_val; }

    inline void sigma(float wanted_val) { std::get<cistem::parameter_names::sigma>(values) = wanted_val; }

    inline void score(float wanted_val) { std::get<cistem::parameter_names::score>(values) = wanted_val; }

    inline void score_change(float wanted_val) { std::get<cistem::parameter_names::score_change>(values) = wanted_val; }

    inline void pixel_size(float wanted_val) { std::get<cistem::parameter_names::pixel_size>(values) = wanted_val; }

    inline void microscope_voltage_kv(float wanted_val) { std::get<cistem::parameter_names::microscope_voltage_kv>(values) = wanted_val; }

    inline void microscope_spherical_aberration_mm(float wanted_val) { std::get<cistem::parameter_names::microscope_spherical_aberration_mm>(values) = wanted_val; }

    inline void amplitude_contrast(float wanted_val) { std::get<cistem::parameter_names::amplitude_contrast>(values) = wanted_val; }

    inline void beam_tilt_x(float wanted_val) { std::get<cistem::parameter_names::beam_tilt_x>(values) = wanted_val; }

    inline void beam_tilt_y(float wanted_val) { std::get<cistem::parameter_names::beam_tilt_y>(values) = wanted_val; }

    inline void image_shift_x(float wanted_val) { std::get<cistem::parameter_names::image_shift_x>(values) = wanted_val; }

    inline void image_shift_y(float wanted_val) { std::get<cistem::parameter_names::image_shift_y>(values) = wanted_val; }

    inline void stack_filename(wxString wanted_val) { std::get<cistem::parameter_names::stack_filename>(values) = wanted_val; }

    inline void original_image_filename(wxString wanted_val) { std::get<cistem::parameter_names::original_image_filename>(values) = wanted_val; }

    inline void reference_3d_filename(wxString wanted_val) { std::get<cistem::parameter_names::reference_3d_filename>(values) = wanted_val; }

    inline void best_2d_class(int wanted_val) { std::get<cistem::parameter_names::best_2d_class>(values) = wanted_val; }

    inline void beam_tilt_group(int wanted_val) { std::get<cistem::parameter_names::beam_tilt_group>(values) = wanted_val; }

    inline void particle_group(int wanted_val) { std::get<cistem::parameter_names::particle_group>(values) = wanted_val; }

    inline void assigned_subset(int wanted_val) { std::get<cistem::parameter_names::assigned_subset>(values) = wanted_val; }

    inline void pre_exposure(float wanted_val) { std::get<cistem::parameter_names::pre_exposure>(values) = wanted_val; }

    inline void total_exposure(float wanted_val) { std::get<cistem::parameter_names::total_exposure>(values) = wanted_val; }

#else
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
    void SetAllToDefault( ); // does not seem to be used
    void ReplaceNanAndInfWithOther(cisTEMParameterLine& other_params);
};

class cisTEMParameterMask {

#ifdef EXPERIMENTAL_CISTEMPARAMS

  public:
    std::array<bool, cistem::parameter_names::count> is_active;

    void SetAllToTrue( );
    void SetAllToFalse( );
    void SetActiveParameters(std::vector<cistem::parameter_names::Enum>& wanted_active_parameters); // uses takes the defines above bitwise

    cisTEMParameterMask( );

    // Define getters each parameter
    inline bool position_in_stack( ) { return is_active[cistem::parameter_names::position_in_stack]; }

    inline bool image_is_active( ) { return is_active[cistem::parameter_names::image_is_active]; }

    inline bool psi( ) { return is_active[cistem::parameter_names::psi]; }

    inline bool theta( ) { return is_active[cistem::parameter_names::theta]; }

    inline bool phi( ) { return is_active[cistem::parameter_names::phi]; }

    inline bool x_shift( ) { return is_active[cistem::parameter_names::x_shift]; }

    inline bool y_shift( ) { return is_active[cistem::parameter_names::y_shift]; }

    inline bool defocus_1( ) { return is_active[cistem::parameter_names::defocus_1]; }

    inline bool defocus_2( ) { return is_active[cistem::parameter_names::defocus_2]; }

    inline bool defocus_angle( ) { return is_active[cistem::parameter_names::defocus_angle]; }

    inline bool phase_shift( ) { return is_active[cistem::parameter_names::phase_shift]; }

    inline bool occupancy( ) { return is_active[cistem::parameter_names::occupancy]; }

    inline bool logp( ) { return is_active[cistem::parameter_names::logp]; }

    inline bool sigma( ) { return is_active[cistem::parameter_names::sigma]; }

    inline bool score( ) { return is_active[cistem::parameter_names::score]; }

    inline bool score_change( ) { return is_active[cistem::parameter_names::score_change]; }

    inline bool pixel_size( ) { return is_active[cistem::parameter_names::pixel_size]; }

    inline bool microscope_voltage_kv( ) { return is_active[cistem::parameter_names::microscope_voltage_kv]; }

    inline bool microscope_spherical_aberration_mm( ) { return is_active[cistem::parameter_names::microscope_spherical_aberration_mm]; }

    inline bool amplitude_contrast( ) { return is_active[cistem::parameter_names::amplitude_contrast]; }

    inline bool beam_tilt_x( ) { return is_active[cistem::parameter_names::beam_tilt_x]; }

    inline bool beam_tilt_y( ) { return is_active[cistem::parameter_names::beam_tilt_y]; }

    inline bool image_shift_x( ) { return is_active[cistem::parameter_names::image_shift_x]; }

    inline bool image_shift_y( ) { return is_active[cistem::parameter_names::image_shift_y]; }

    inline bool stack_filename( ) { return is_active[cistem::parameter_names::stack_filename]; }

    inline bool original_image_filename( ) { return is_active[cistem::parameter_names::original_image_filename]; }

    inline bool reference_3d_filename( ) { return is_active[cistem::parameter_names::reference_3d_filename]; }

    inline bool best_2d_class( ) { return is_active[cistem::parameter_names::best_2d_class]; }

    inline bool beam_tilt_group( ) { return is_active[cistem::parameter_names::beam_tilt_group]; }

    inline bool particle_group( ) { return is_active[cistem::parameter_names::particle_group]; }

    inline bool assigned_subset( ) { return is_active[cistem::parameter_names::assigned_subset]; }

    inline bool pre_exposure( ) { return is_active[cistem::parameter_names::pre_exposure]; }

    inline bool total_exposure( ) { return is_active[cistem::parameter_names::total_exposure]; }

    // Define ses for each parameter
    inline void position_in_stack(bool value) { is_active[cistem::parameter_names::position_in_stack] = value; }

    inline void image_is_active(bool value) { is_active[cistem::parameter_names::image_is_active] = value; }

    inline void psi(bool value) { is_active[cistem::parameter_names::psi] = value; }

    inline void theta(bool value) { is_active[cistem::parameter_names::theta] = value; }

    inline void phi(bool value) { is_active[cistem::parameter_names::phi] = value; }

    inline void x_shift(bool value) { is_active[cistem::parameter_names::x_shift] = value; }

    inline void y_shift(bool value) { is_active[cistem::parameter_names::y_shift] = value; }

    inline void defocus_1(bool value) { is_active[cistem::parameter_names::defocus_1] = value; }

    inline void defocus_2(bool value) { is_active[cistem::parameter_names::defocus_2] = value; }

    inline void defocus_angle(bool value) { is_active[cistem::parameter_names::defocus_angle] = value; }

    inline void phase_shift(bool value) { is_active[cistem::parameter_names::phase_shift] = value; }

    inline void occupancy(bool value) { is_active[cistem::parameter_names::occupancy] = value; }

    inline void logp(bool value) { is_active[cistem::parameter_names::logp] = value; }

    inline void sigma(bool value) { is_active[cistem::parameter_names::sigma] = value; }

    inline void score(bool value) { is_active[cistem::parameter_names::score] = value; }

    inline void score_change(bool value) { is_active[cistem::parameter_names::score_change] = value; }

    inline void pixel_size(bool value) { is_active[cistem::parameter_names::pixel_size] = value; }

    inline void microscope_voltage_kv(bool value) { is_active[cistem::parameter_names::microscope_voltage_kv] = value; }

    inline void microscope_spherical_aberration_mm(bool value) { is_active[cistem::parameter_names::microscope_spherical_aberration_mm] = value; }

    inline void amplitude_contrast(bool value) { is_active[cistem::parameter_names::amplitude_contrast] = value; }

    inline void beam_tilt_x(bool value) { is_active[cistem::parameter_names::beam_tilt_x] = value; }

    inline void beam_tilt_y(bool value) { is_active[cistem::parameter_names::beam_tilt_y] = value; }

    inline void image_shift_x(bool value) { is_active[cistem::parameter_names::image_shift_x] = value; }

    inline void image_shift_y(bool value) { is_active[cistem::parameter_names::image_shift_y] = value; }

    inline void stack_filename(bool value) { is_active[cistem::parameter_names::stack_filename] = value; }

    inline void original_image_filename(bool value) { is_active[cistem::parameter_names::original_image_filename] = value; }

    inline void reference_3d_filename(bool value) { is_active[cistem::parameter_names::reference_3d_filename] = value; }

    inline void best_2d_class(bool value) { is_active[cistem::parameter_names::best_2d_class] = value; }

    inline void beam_tilt_group(bool value) { is_active[cistem::parameter_names::beam_tilt_group] = value; }

    inline void particle_group(bool value) { is_active[cistem::parameter_names::particle_group] = value; }

    inline void assigned_subset(bool value) { is_active[cistem::parameter_names::assigned_subset] = value; }

    inline void pre_exposure(bool value) { is_active[cistem::parameter_names::pre_exposure] = value; }

    inline void total_exposure(bool value) { is_active[cistem::parameter_names::total_exposure] = value; }

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
    using param_t = cistem::parameter_names::Enum;

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
    // TODO: It would be too laborious until approved to change all of these to simple getters, but ideally it would just be
    // inline int position_in_stack(int line_number)  { return all_parameters.Item(line_number).position_in_stack(); }
    inline int ReturnPositionInStack(int line_number) { return all_parameters.Item(line_number).position_in_stack( ); }

    inline int ReturnImageIsActive(int line_number) { return all_parameters.Item(line_number).image_is_active( ); }

    inline int ReturnPsi(int line_number) { return all_parameters.Item(line_number).psi( ); }

    inline int ReturnTheta(int line_number) { return all_parameters.Item(line_number).theta( ); }

    inline int ReturnPhi(int line_number) { return all_parameters.Item(line_number).phi( ); }

    inline int ReturnXShift(int line_number) { return all_parameters.Item(line_number).x_shift( ); }

    inline int ReturnYShift(int line_number) { return all_parameters.Item(line_number).y_shift( ); }

    inline int ReturnDefocus1(int line_number) { return all_parameters.Item(line_number).defocus_1( ); }

    inline int ReturnDefocus2(int line_number) { return all_parameters.Item(line_number).defocus_2( ); }

    inline int ReturnDefocusAngle(int line_number) { return all_parameters.Item(line_number).defocus_angle( ); }

    inline int ReturnPhaseShift(int line_number) { return all_parameters.Item(line_number).phase_shift( ); }

    inline int ReturnOccupancy(int line_number) { return all_parameters.Item(line_number).occupancy( ); }

    inline int ReturnLogP(int line_number) { return all_parameters.Item(line_number).logp( ); }

    inline int ReturnSigma(int line_number) { return all_parameters.Item(line_number).sigma( ); }

    inline int ReturnScore(int line_number) { return all_parameters.Item(line_number).score( ); }

    inline int ReturnMicroscopeVoltage(int line_number) { return all_parameters.Item(line_number).microscope_voltage_kv( ); }

    inline int ReturnMicroscopeCs(int line_number) { return all_parameters.Item(line_number).microscope_spherical_aberration_mm( ); }

    inline int ReturnAmplitudeContrast(int line_number) { return all_parameters.Item(line_number).amplitude_contrast( ); }

    inline int ReturnBeamTiltX(int line_number) { return all_parameters.Item(line_number).beam_tilt_x( ); }

    inline int ReturnBeamTiltY(int line_number) { return all_parameters.Item(line_number).beam_tilt_y( ); }

    inline float ReturnImageShiftX(int line_number) { return all_parameters.Item(line_number).image_shift_x( ); }

    inline float ReturnImageShiftY(int line_number) { return all_parameters.Item(line_number).image_shift_y( ); }

    inline wxString ReturnStackFilename(int line_number) { return all_parameters.Item(line_number).stack_filename( ); }

    inline wxString ReturnOriginalImageFilename(int line_number) { return all_parameters.Item(line_number).original_image_filename( ); }

    inline wxString ReturnReference3DFilename(int line_number) { return all_parameters.Item(line_number).reference_3d_filename( ); }

    inline int ReturnBest2DClass(int line_number) { return all_parameters.Item(line_number).best_2d_class( ); }

    inline int ReturnParticleGroup(int line_number) { return all_parameters.Item(line_number).particle_group( ); }

    inline int ReturnBeamTiltGroup(int line_number) { return all_parameters.Item(line_number).beam_tilt_group( ); }

    inline int ReturnAssignedSubset(int line_number) { return all_parameters.Item(line_number).assigned_subset( ); }

    inline float ReturnPreExposure(int line_number) { return all_parameters.Item(line_number).pre_exposure( ); }

    inline float ReturnTotalExposure(int line_number) { return all_parameters.Item(line_number).total_exposure( ); }

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
};

#endif