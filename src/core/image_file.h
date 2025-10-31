/*
 * Original Copyright (c) 2017, Howard Hughes Medical Institute
 * Licensed under Janelia Research Campus Software License 1.2
 * See license_details/LICENSE-JANELIA.txt
 *
 * Modifications Copyright (c) 2025, Stochastic Analytics, LLC
 * Modifications licensed under MPL 2.0 for academic use; 
 * commercial license required for commercial use.
 * See LICENSE.md for details.
 */

/*
 * Provide a common interface to interact with all the different
 * files we support (MRC, DM, TIF, etc.)
 */

enum supported_file_types {
    MRC_FILE,
    TIFF_FILE,
    DM_FILE,
    EER_FILE,
    UNSUPPORTED_FILE_TYPE
};

class ImageFile : public AbstractImageFile {
  private:
    // These are the actual file objects doing the work
    MRCFile  mrc_file;
    TiffFile tiff_file;
    DMFile   dm_file;
    EerFile  eer_file;

    int      file_type;
    wxString file_type_string;
    void     SetFileTypeFromExtension( );

  public:
    ImageFile( );
    ImageFile(std::string wanted_filename, bool overwrite = false);
    ~ImageFile( );

    int   ReturnXSize( );
    int   ReturnYSize( );
    int   ReturnZSize( );
    int   ReturnNumberOfSlices( );
    float ReturnPixelSize( );

    bool IsOpen( );

    bool OpenFile(std::string wanted_filename, bool overwrite, bool wait_for_file_to_exist = false, bool check_only_the_first_image = false, int eer_super_res_factor = 1, int eer_frames_per_image = 0);
    void CloseFile( );

    void ReadSliceFromDisk(int slice_number, float* output_array);
    void ReadSlicesFromDisk(int start_slice, int end_slice, float* output_array);

    void WriteSliceToDisk(int slice_number, float* input_array);
    void WriteSlicesToDisk(int start_slice, int end_slice, float* input_array);

    void PrintInfo( );
};
