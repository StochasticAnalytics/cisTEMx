/**
 * MRC File Writing Examples (cisTEM C++)
 *
 * Demonstrates common patterns for writing MRC files in cisTEM.
 * cisTEM writes MODE 2 (float32) by default, or MODE 12 (float16) if requested.
 */

#include "core/core_headers.h"

// ============================================================================
// Example 1: Basic File Writing
// ============================================================================

void write_basic(const char* filename, int nx, int ny, int nz, float pixel_size) {
    // Open MRC file for writing (true = write mode)
    MRCFile output_file(filename, true);

    // Set pixel size
    output_file.SetPixelSize(pixel_size);

    // Generate test data
    float* data = new float[nx * ny * nz];
    for ( int i = 0; i < nx * ny * nz; i++ ) {
        data[i] = (float)i / (nx * ny * nz); // Normalized values
    }

    // Write all slices (0-indexed, inclusive end)
    output_file.WriteSlicesToDisk(0, nz - 1, data);

    wxPrintf("Wrote %s: %d × %d × %d, pixel_size=%.3f Å\n",
             filename, nx, ny, nz, pixel_size);

    // Clean up
    delete[] data;
    output_file.CloseFile( );
}

// ============================================================================
// Example 2: Writing Single Slice
// ============================================================================

void write_single_slice(const char* filename, int nx, int ny, float pixel_size) {
    // Create file with nz=1
    MRCFile output_file(filename, true);
    output_file.SetPixelSize(pixel_size);

    // Generate 2D slice
    float* slice = new float[nx * ny];
    for ( int i = 0; i < nx * ny; i++ ) {
        slice[i] = sinf((float)i * 0.01f); // Example pattern
    }

    // Write single slice (slice index 0)
    output_file.WriteSliceToDisk(0, slice);

    wxPrintf("Wrote single slice %s: %d × %d\n", filename, nx, ny);

    delete[] slice;
    output_file.CloseFile( );
}

// ============================================================================
// Example 3: Writing Slices Incrementally
// ============================================================================

void write_slices_incremental(const char* filename, int nx, int ny, int nz, float pixel_size) {
    MRCFile output_file(filename, true);
    output_file.SetPixelSize(pixel_size);

    // Allocate buffer for ONE slice (memory efficient)
    float* slice = new float[nx * ny];

    // Write slices one by one
    for ( int z = 0; z < nz; z++ ) {
        // Generate/process slice data
        for ( int i = 0; i < nx * ny; i++ ) {
            slice[i] = z * 0.1f + i * 0.001f; // Example: varies by Z
        }

        // Write this slice
        output_file.WriteSliceToDisk(z, slice);

        if ( (z + 1) % 100 == 0 ) {
            wxPrintf("Written %d/%d slices\n", z + 1, nz);
        }
    }

    wxPrintf("Wrote %d slices incrementally to %s\n", nz, filename);

    delete[] slice;
    output_file.CloseFile( );
}

// ============================================================================
// Example 4: Writing with Specific Data Statistics
// ============================================================================

void write_with_statistics(const char* filename, float* data, int nx, int ny, int nz, float pixel_size) {
    MRCFile output_file(filename, true);
    output_file.SetPixelSize(pixel_size);

    // Calculate statistics
    int    n_voxels = nx * ny * nz;
    float  min_val  = data[0];
    float  max_val  = data[0];
    double sum      = 0.0;

    for ( int i = 0; i < n_voxels; i++ ) {
        if ( data[i] < min_val )
            min_val = data[i];
        if ( data[i] > max_val )
            max_val = data[i];
        sum += data[i];
    }

    float mean_val = (float)(sum / n_voxels);

    // Calculate RMS
    double sum_sq = 0.0;
    for ( int i = 0; i < n_voxels; i++ ) {
        float diff = data[i] - mean_val;
        sum_sq += diff * diff;
    }
    float rms_val = sqrtf((float)(sum_sq / n_voxels));

    // Set header statistics
    output_file.SetDensityStatistics(min_val, max_val, mean_val, rms_val);

    // Write data
    output_file.WriteSlicesToDisk(0, nz - 1, data);

    wxPrintf("Wrote %s with statistics:\n", filename);
    wxPrintf("  Min: %.3f, Max: %.3f\n", min_val, max_val);
    wxPrintf("  Mean: %.3f, RMS: %.3f\n", mean_val, rms_val);

    output_file.CloseFile( );
}

// ============================================================================
// Example 5: Writing Compressed (FP16 for 50% Size Reduction)
// ============================================================================

void write_compressed(const char* filename, float* data, int nx, int ny, int nz, float pixel_size) {
    MRCFile output_file(filename, true);
    output_file.SetPixelSize(pixel_size);

    // Enable FP16 output (MODE 12)
    output_file.SetOutputToFP16( );

    // Write data (automatically converted to float16)
    output_file.WriteSlicesToDisk(0, nz - 1, data);

    int   n_voxels = nx * ny * nz;
    float size_f32 = n_voxels * 4.0f / 1e6; // MB
    float size_f16 = n_voxels * 2.0f / 1e6; // MB

    wxPrintf("Wrote compressed %s (MODE 12)\n", filename);
    wxPrintf("  Size reduction: %.1f MB → %.1f MB (50%%)\n", size_f32, size_f16);
    wxPrintf("  Precision: ~3 significant digits\n");

    output_file.CloseFile( );
}

// ============================================================================
// Example 6: Writing from Image Class
// ============================================================================

void write_from_image(const char* filename, Image& my_image) {
    // Image class provides QuickAndDirtyWriteSlices
    my_image.QuickAndDirtyWriteSlices(filename, 1, my_image.logical_z_dimension);

    wxPrintf("Wrote Image to %s\n", filename);
    wxPrintf("  Dimensions: %d × %d × %d\n",
             my_image.logical_x_dimension,
             my_image.logical_y_dimension,
             my_image.logical_z_dimension);
}

// ============================================================================
// Example 7: Setting Pixel Size and Writing Header
// ============================================================================

void write_and_update_header(const char* filename, float* data, int nx, int ny, int nz, float pixel_size) {
    MRCFile output_file(filename, true);

    // Set pixel size and immediately write header
    output_file.SetPixelSizeAndWriteHeader(pixel_size);

    // Write data
    output_file.WriteSlicesToDisk(0, nz - 1, data);

    wxPrintf("Wrote %s with pixel size %.3f Å\n", filename, pixel_size);

    output_file.CloseFile( );
}

// ============================================================================
// Example 8: Copying and Modifying File
// ============================================================================

void copy_and_modify(const char* input_filename, const char* output_filename) {
    // Read input
    MRCFile input_file(input_filename, false);

    int   nx         = input_file.ReturnXSize( );
    int   ny         = input_file.ReturnYSize( );
    int   nz         = input_file.ReturnZSize( );
    float pixel_size = input_file.ReturnPixelSize( );

    float* data = new float[nx * ny * nz];
    input_file.ReadSlicesFromDisk(0, nz - 1, data);
    input_file.CloseFile( );

    // Modify data (e.g., scale by 2)
    for ( int i = 0; i < nx * ny * nz; i++ ) {
        data[i] *= 2.0f;
    }

    // Write output
    MRCFile output_file(output_filename, true);
    output_file.SetPixelSize(pixel_size);
    output_file.WriteSlicesToDisk(0, nz - 1, data);
    output_file.CloseFile( );

    wxPrintf("Copied %s → %s and scaled by 2\n", input_filename, output_filename);

    delete[] data;
}

// ============================================================================
// Example 9: Writing Subset of Slices
// ============================================================================

void write_subset(const char* input_filename, const char* output_filename, int start_slice, int end_slice) {
    // Read subset from input
    MRCFile input_file(input_filename, false);

    int   nx         = input_file.ReturnXSize( );
    int   ny         = input_file.ReturnYSize( );
    int   nz_subset  = end_slice - start_slice + 1;
    float pixel_size = input_file.ReturnPixelSize( );

    float* subset_data = new float[nx * ny * nz_subset];
    input_file.ReadSlicesFromDisk(start_slice, end_slice, subset_data);
    input_file.CloseFile( );

    // Write subset to new file
    MRCFile output_file(output_filename, true);
    output_file.SetPixelSize(pixel_size);
    output_file.WriteSlicesToDisk(0, nz_subset - 1, subset_data);
    output_file.CloseFile( );

    wxPrintf("Extracted slices %d-%d to %s\n", start_slice, end_slice, output_filename);

    delete[] subset_data;
}

// ============================================================================
// Example 10: Flushing Data During Long Writes
// ============================================================================

void write_with_flushing(const char* filename, int nx, int ny, int nz, float pixel_size) {
    MRCFile output_file(filename, true);
    output_file.SetPixelSize(pixel_size);

    float* slice = new float[nx * ny];

    for ( int z = 0; z < nz; z++ ) {
        // Generate slice
        for ( int i = 0; i < nx * ny; i++ ) {
            slice[i] = z * 0.1f;
        }

        // Write slice
        output_file.WriteSliceToDisk(z, slice);

        // Flush every 100 slices to ensure data is written to disk
        if ( (z + 1) % 100 == 0 ) {
            output_file.FlushFile( );
            wxPrintf("Flushed after %d slices\n", z + 1);
        }
    }

    delete[] slice;
    output_file.CloseFile( );

    wxPrintf("Completed writing with periodic flushing\n");
}

// ============================================================================
// Example 11: Standard vs Compressed Comparison
// ============================================================================

void write_both_formats(const char* base_filename, float* data, int nx, int ny, int nz, float pixel_size) {
    // Write standard MODE 2 (float32)
    char filename_f32[256];
    snprintf(filename_f32, sizeof(filename_f32), "%s_float32.mrc", base_filename);

    MRCFile output_f32(filename_f32, true);
    output_f32.SetPixelSize(pixel_size);
    output_f32.WriteSlicesToDisk(0, nz - 1, data);
    output_f32.CloseFile( );

    // Write compressed MODE 12 (float16)
    char filename_f16[256];
    snprintf(filename_f16, sizeof(filename_f16), "%s_float16.mrc", base_filename);

    MRCFile output_f16(filename_f16, true);
    output_f16.SetPixelSize(pixel_size);
    output_f16.SetOutputToFP16( );
    output_f16.WriteSlicesToDisk(0, nz - 1, data);
    output_f16.CloseFile( );

    int n_voxels = nx * ny * nz;
    wxPrintf("Wrote both formats:\n");
    wxPrintf("  %s: %.1f MB (MODE 2, float32)\n", filename_f32, n_voxels * 4.0f / 1e6);
    wxPrintf("  %s: %.1f MB (MODE 12, float16)\n", filename_f16, n_voxels * 2.0f / 1e6);
}

// ============================================================================
// Main: Example Usage
// ============================================================================

int main(int argc, char* argv[]) {
    if ( argc < 2 ) {
        wxPrintf("Usage: %s <output_prefix>\n", argv[0]);
        return 1;
    }

    const char* output_prefix = argv[1];

    // Example dimensions
    int   nx         = 256;
    int   ny         = 256;
    int   nz         = 10;
    float pixel_size = 1.5f; // Ångströms

    // Generate test data
    float* data = new float[nx * ny * nz];
    for ( int i = 0; i < nx * ny * nz; i++ ) {
        data[i] = sinf((float)i * 0.001f);
    }

    wxPrintf("=======================================================\n");
    wxPrintf("Example 1: Basic File Writing\n");
    wxPrintf("=======================================================\n");
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_basic.mrc", output_prefix);
    write_basic(filename, nx, ny, nz, pixel_size);

    wxPrintf("\n=======================================================\n");
    wxPrintf("Example 3: Writing Slices Incrementally\n");
    wxPrintf("=======================================================\n");
    snprintf(filename, sizeof(filename), "%s_incremental.mrc", output_prefix);
    write_slices_incremental(filename, nx, ny, nz, pixel_size);

    wxPrintf("\n=======================================================\n");
    wxPrintf("Example 5: Writing Compressed (FP16)\n");
    wxPrintf("=======================================================\n");
    snprintf(filename, sizeof(filename), "%s_compressed.mrc", output_prefix);
    write_compressed(filename, data, nx, ny, nz, pixel_size);

    delete[] data;

    wxPrintf("\nAll examples completed\n");

    return 0;
}
