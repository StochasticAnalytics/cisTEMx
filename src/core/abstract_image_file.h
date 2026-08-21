// Respace tightly packed rows into a row-padded layout in place, back to front, one memmove
// per row. ReadSlicesFromDisk implementations use this to honour output_padding_jump without
// requiring a separate whole-image respace pass (Image::AddFFTWPadding) after the read.
inline void RespaceTightRowsToPadded(float* data, long row_length, long number_of_rows, long padding_jump) {
    if ( padding_jump == 0 )
        return;
    for ( long row = number_of_rows - 1; row > 0; row-- ) {
        memmove(&data[row * (row_length + padding_jump)], &data[row * row_length], sizeof(float) * row_length);
    }
}

class AbstractImageFile {

  public:
    wxFileName filename;

    AbstractImageFile( );
    AbstractImageFile(std::string filename, bool overwrite = false);
    ~AbstractImageFile( );

    virtual int   ReturnXSize( )          = 0;
    virtual int   ReturnYSize( )          = 0;
    virtual int   ReturnZSize( )          = 0;
    virtual int   ReturnNumberOfSlices( ) = 0;
    virtual float ReturnPixelSize( )      = 0;

    virtual bool IsOpen( ) = 0;

    virtual bool OpenFile(std::string filename, bool overwrite = false, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false, int eer_super_res_factor = 1, int eer_frames_per_image = 0, bool create_if_missing = true) = 0; // Return true if everything about the file looks OK
    virtual void CloseFile( )                                                                                                                                                                                                                    = 0;

    virtual void ReadSliceFromDisk(int slice_number, float* output_array) = 0;
    // output_padding_jump > 0 asks for the data to be written in row-padded (FFTW) layout,
    // with that many unwritten floats after each row; 0 keeps the historical tight layout.
    virtual void ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array, int output_padding_jump = 0) = 0;

    virtual void WriteSliceToDisk(int slice_number, float* input_array)                = 0;
    virtual void WriteSlicesToDisk(int start_slice, int end_slice, float* input_array) = 0;

    virtual void PrintInfo( ) = 0;
};
