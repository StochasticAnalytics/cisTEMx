/**
 * MRC File Reading Examples (cisTEM C++)
 *
 * Demonstrates common patterns for reading MRC files in cisTEM.
 * All data is converted to float32 in memory regardless of file MODE.
 */

#include "core/core_headers.h"

// ============================================================================
// Example 1: Basic File Reading
// ============================================================================

void read_basic(const char* filename) {
    // Open MRC file (false = read mode)
    MRCFile input_file(filename, false);

    // Get dimensions
    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );
    int nz = input_file.ReturnZSize( );

    wxPrintf("Dimensions: %d × %d × %d\n", nx, ny, nz);

    // Get pixel size
    float pixel_size = input_file.ReturnPixelSize( ); // Ångströms
    wxPrintf("Pixel size: %.3f Å\n", pixel_size);

    // Allocate buffer for all data
    float* data = new float[nx * ny * nz];

    // Read all slices (0-indexed, inclusive end)
    input_file.ReadSlicesFromDisk(0, nz - 1, data);

    wxPrintf("Read %d voxels\n", nx * ny * nz);

    // Access specific voxel (x=50, y=100, z=10)
    int index = (10 * ny * nx) + (100 * nx) + 50; // z*ny*nx + y*nx + x
    wxPrintf("Voxel at (50, 100, 10): %.3f\n", data[index]);

    // Clean up
    delete[] data;
    input_file.CloseFile( );
}

// ============================================================================
// Example 2: Reading Specific Slices
// ============================================================================

void read_slices(const char* filename, int start_slice, int end_slice) {
    MRCFile input_file(filename, false);

    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );
    int nz = input_file.ReturnZSize( );

    // Validate slice range
    MyDebugAssertTrue(start_slice >= 0 && start_slice < nz, "Invalid start slice");
    MyDebugAssertTrue(end_slice >= start_slice && end_slice < nz, "Invalid end slice");

    int n_slices = end_slice - start_slice + 1;

    // Allocate buffer for slice range only
    float* slice_data = new float[nx * ny * n_slices];

    // Read slice range (inclusive)
    input_file.ReadSlicesFromDisk(start_slice, end_slice, slice_data);

    wxPrintf("Read slices %d-%d (%d slices)\n", start_slice, end_slice, n_slices);
    wxPrintf("  Slice size: %d × %d\n", nx, ny);

    delete[] slice_data;
    input_file.CloseFile( );
}

// ============================================================================
// Example 3: Reading Single Slice
// ============================================================================

void read_single_slice(const char* filename, int slice_number) {
    MRCFile input_file(filename, false);

    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );

    // Allocate buffer for one slice
    float* slice = new float[nx * ny];

    // Read single slice
    input_file.ReadSliceFromDisk(slice_number, slice);

    wxPrintf("Read slice %d\n", slice_number);

    // Calculate statistics
    float sum     = 0.0f;
    float min_val = slice[0];
    float max_val = slice[0];

    for ( int i = 0; i < nx * ny; i++ ) {
        sum += slice[i];
        if ( slice[i] < min_val )
            min_val = slice[i];
        if ( slice[i] > max_val )
            max_val = slice[i];
    }

    float mean = sum / (nx * ny);

    wxPrintf("  Min: %.3f, Max: %.3f, Mean: %.3f\n", min_val, max_val, mean);

    delete[] slice;
    input_file.CloseFile( );
}

// ============================================================================
// Example 4: Reading Header Information
// ============================================================================

void read_header_info(const char* filename) {
    MRCFile input_file(filename, false);

    // Dimensions
    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );
    int nz = input_file.ReturnZSize( );
    wxPrintf("Dimensions: %d × %d × %d\n", nx, ny, nz);

    // Pixel size (assumes isotropic)
    float pixel_size = input_file.ReturnPixelSize( );
    wxPrintf("Pixel size: %.3f Å\n", pixel_size);

    // Physical size
    float phys_x = nx * pixel_size;
    float phys_y = ny * pixel_size;
    float phys_z = nz * pixel_size;
    wxPrintf("Physical size: %.1f × %.1f × %.1f Å\n", phys_x, phys_y, phys_z);

    // Print all header info
    input_file.PrintInfo( );

    input_file.CloseFile( );
}

// ============================================================================
// Example 5: Reading with Image Class Integration
// ============================================================================

void read_into_image(const char* filename) {
    // Image class provides higher-level operations
    Image my_image;

    // ReadSlices handles file opening/closing
    my_image.ReadSlices(filename, 1, 100); // Read slices 1-100

    wxPrintf("Image dimensions: %d × %d × %d\n",
             my_image.logical_x_dimension,
             my_image.logical_y_dimension,
             my_image.logical_z_dimension);

    wxPrintf("Pixel size: %.3f Å\n", my_image.pixel_size);

    // Image class provides analysis methods
    float mean     = my_image.ReturnAverageOfRealValues( );
    float variance = my_image.ReturnVarianceOfRealValues( );
    float sigma    = sqrtf(variance);

    wxPrintf("Statistics: Mean=%.3f, Sigma=%.3f\n", mean, sigma);
}

// ============================================================================
// Example 6: Handling Different Data Types
// ============================================================================

void demonstrate_data_type_handling(const char* filename) {
    MRCFile input_file(filename, false);

    // cisTEM converts all modes to float32 automatically
    // Supported reading modes: 0, 1, 2, 5, 6, 12, 101

    wxPrintf("MODE handling:\n");
    wxPrintf("  Mode 0 (int8)    → converted to float32\n");
    wxPrintf("  Mode 1 (int16)   → converted to float32\n");
    wxPrintf("  Mode 2 (float32) → read directly\n");
    wxPrintf("  Mode 6 (uint16)  → converted to float32\n");
    wxPrintf("  Mode 12 (float16) → converted to float32\n");
    wxPrintf("  Mode 101 (4-bit)  → unpacked to float32\n");

    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );

    float* data = new float[nx * ny];
    input_file.ReadSliceFromDisk(0, data);

    wxPrintf("Data read as float32 regardless of file MODE\n");

    delete[] data;
    input_file.CloseFile( );
}

// ============================================================================
// Example 7: Checking File Compatibility
// ============================================================================

void check_file_compatibility(const char* filename) {
    MRCFile input_file(filename, false);

    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );
    int nz = input_file.ReturnZSize( );

    // Validate dimensions
    bool valid = true;

    if ( nx <= 0 || ny <= 0 || nz < 0 ) {
        wxPrintf("Error: Invalid dimensions\n");
        valid = false;
    }

    // Check if open
    if ( ! input_file.IsOpen( ) ) {
        wxPrintf("Error: File not open\n");
        valid = false;
    }

    if ( valid ) {
        wxPrintf("✓ File is compatible with cisTEM\n");
        input_file.PrintInfo( );
    }

    input_file.CloseFile( );
}

// ============================================================================
// Example 8: Memory-Efficient Slice Processing
// ============================================================================

void process_slices_sequentially(const char* filename) {
    MRCFile input_file(filename, false);

    int nx = input_file.ReturnXSize( );
    int ny = input_file.ReturnYSize( );
    int nz = input_file.ReturnZSize( );

    // Allocate buffer for ONE slice only (memory efficient)
    float* slice = new float[nx * ny];

    // Process each slice sequentially
    for ( int z = 0; z < nz; z++ ) {
        // Read slice
        input_file.ReadSliceFromDisk(z, slice);

        // Process slice (e.g., apply filter, calculate statistics)
        // ... your processing here ...

        if ( (z + 1) % 100 == 0 ) {
            wxPrintf("Processed %d/%d slices\n", z + 1, nz);
        }
    }

    delete[] slice;
    input_file.CloseFile( );

    wxPrintf("Completed sequential processing\n");
}

// ============================================================================
// Example 9: Comparing Two Files
// ============================================================================

void compare_files(const char* file1, const char* file2) {
    MRCFile mrc1(file1, false);
    MRCFile mrc2(file2, false);

    // Check dimensions match
    bool same_dimensions = mrc1.HasSameDimensionsAs(mrc2);

    if ( same_dimensions ) {
        wxPrintf("✓ Files have same dimensions\n");
    }
    else {
        wxPrintf("✗ Files have different dimensions\n");
        wxPrintf("  File 1: %d × %d × %d\n",
                 mrc1.ReturnXSize( ), mrc1.ReturnYSize( ), mrc1.ReturnZSize( ));
        wxPrintf("  File 2: %d × %d × %d\n",
                 mrc2.ReturnXSize( ), mrc2.ReturnYSize( ), mrc2.ReturnZSize( ));
    }

    // Compare pixel sizes
    float ps1 = mrc1.ReturnPixelSize( );
    float ps2 = mrc2.ReturnPixelSize( );

    if ( fabsf(ps1 - ps2) < 0.001f ) {
        wxPrintf("✓ Files have same pixel size: %.3f Å\n", ps1);
    }
    else {
        wxPrintf("✗ Files have different pixel sizes: %.3f vs %.3f Å\n", ps1, ps2);
    }

    mrc1.CloseFile( );
    mrc2.CloseFile( );
}

// ============================================================================
// Example 10: Reading with Wait (for Files Being Written)
// ============================================================================

void read_with_wait(const char* filename) {
    // OpenFile with wait_for_file_to_exist = true
    // Useful when reading files that are still being written

    MRCFile input_file;

    bool success = input_file.OpenFile(
            filename,
            false, // overwrite = false (read mode)
            true, // wait_for_file_to_exist = true
            false, // check_only_the_first_image = false
            1, // eer_super_res_factor (EER files only)
            0 // eer_frames_per_image (EER files only)
    );

    if ( success ) {
        wxPrintf("File opened successfully\n");
        input_file.PrintInfo( );
        input_file.CloseFile( );
    }
    else {
        wxPrintf("Failed to open file\n");
    }
}

// ============================================================================
// Main: Example Usage
// ============================================================================

int main(int argc, char* argv[]) {
    if ( argc < 2 ) {
        wxPrintf("Usage: %s <mrc_file>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];

    wxPrintf("=======================================================\n");
    wxPrintf("Example 1: Basic File Reading\n");
    wxPrintf("=======================================================\n");
    read_basic(filename);

    wxPrintf("\n=======================================================\n");
    wxPrintf("Example 4: Reading Header Information\n");
    wxPrintf("=======================================================\n");
    read_header_info(filename);

    wxPrintf("\n=======================================================\n");
    wxPrintf("Example 7: Checking File Compatibility\n");
    wxPrintf("=======================================================\n");
    check_file_compatibility(filename);

    return 0;
}
