#ifndef _SRC_PROGRAMS_REFINE3D_BATCHED_SEARCH_H_
#define _SRC_PROGRAMS_REFINE3D_BATCHED_SEARCH_H_

class GpuImage;

// class Peak;

typedef struct _IntegerPeak {
    float value;
    int   x;
    int   y;
    int   z;
    int   address_in_image;
} IntegerPeak;

class BatchedSearch {

  public:
    BatchedSearch( );
    ~BatchedSearch( );

    void Deallocate( );

    void Init(GpuImage& reference_image, int wanted_number_search_images, int wanted_batch_size, bool test_mirror, int max_pix_x = 0, int max_pix_y = 0);

    void SetMaxSearchExtension(int max_pixel_radius_x, int max_pixel_radius_y) {
        _max_pixel_radius_x = max_pixel_radius_x;
        _max_pixel_radius_y = max_pixel_radius_y;
    }

    inline int n_batches( ) const { return _n_batches; };

    inline int loop_batch_size(int iBatch) { return (iBatch < _n_batches - 1) ? _batch_size : _n_in_last_batch; };

    inline int stride( ) const { return _stride; };

    inline int max_pixel_radius_x( ) { return _max_pixel_radius_x; };

    inline int max_pixel_radius_y( ) { return _max_pixel_radius_y; };

    inline int intra_loop_inc( ) { return _intra_loop_inc; };

    inline int batch_size( ) { return _batch_size; };

    Peak* _peak_buffer;
    Peak* _d_peak_buffer;

    inline void print_member_variables( ) {
        std::cerr << "BatchedSearch::_n_batches: " << _n_batches << std::endl;
        std::cerr << "BatchedSearch::_batch_size: " << _batch_size << std::endl;
        std::cerr << "BatchedSearch::_n_in_last_batch: " << _n_in_last_batch << std::endl;
        std::cerr << "BatchedSearch::_stride: " << _stride << std::endl;
        std::cerr << "BatchedSearch::_max_pixel_radius_x: " << _max_pixel_radius_x << std::endl;
        std::cerr << "BatchedSearch::_max_pixel_radius_y: " << _max_pixel_radius_y << std::endl;
        std::cerr << "BatchedSearch::_peak_buffer: " << _peak_buffer << std::endl;
        std::cerr << "BatchedSearch::_d_peak_buffer: " << _d_peak_buffer << std::endl;
    }

    // IntegerPeak* _peak_buffer;
    // IntegerPeak* _device_peak_buffer;

  private:
    int _n_search_images;
    int _batch_size;
    int _n_batches;
    int _n_in_last_batch;
    int _intra_loop_inc;
    int _intra_loop_batch_size;

    int _max_pixel_radius_x;
    int _max_pixel_radius_y;
    int _stride;

    bool _test_mirror;

    bool _is_initialized;
};
#endif