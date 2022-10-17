#ifndef _SRC_REFINE3D_DEFINES_H_
#define _SRC_REFINE3D_DEFINES_H_

/*
    Defines used for manually altering control flow for debugging. 
    Local to refine3d.cpp and ProjectionComparision.cpp
*/

////////////////////////////////////////////////////////////
/*
Print the scores and line numbers of the current line.
*/
////////////////////////////////////////////////////////////

// #define PRINT_SCORES

////////////////////////////////////////////////////////////
/* 
Calculate the score on the CPU. 
disables copy of the particle_image to gpu (per particle loop N) and 
enables copy of the projection image to host per search position(N* O[50])
Note; this is a performance check and changes data movement.
*/
////////////////////////////////////////////////////////////

#define CALCULATE_SCORE_ON_CPU_DISABLE_GPU_PARTICLE

////////////////////////////////////////////////////////////
/*
Calculate the score on both GPU and CPU printing to stdout
Note: this is a correctness check and not a performance check.
All extra data movement and calculation incurred.
*/
////////////////////////////////////////////////////////////

// #define COMPARE_GPU_CPU_SCORE

////////////////////////////////////////////////////////////
/*
Save output images to CWD to compare CPU / GPU projection and 
particles. Will Exit after N_DEBUG_IMAGES are saved.
*/
////////////////////////////////////////////////////////////

// #define SAVE_DEBUG_IMAGES
// #define N_DEBUG_IMAGES 10

////////////////////////////////////////////////////////////
/*
To test the impact of reduced use of texture memory vs competition
for shared resources. Threads will use 
*/
////////////////////////////////////////////////////////////

#define N_SHARING_GPU_DENSITY_MAP = 0
////////////////////////////////////////////////////////////
/*
*/
////////////////////////////////////////////////////////////

// Note: this does not work yet, even though it seems to be the same as on the GPU?
// #define USE_OPTIMIZED_CPU_SCORE_CALCULATION

// Some sanity checks to ensure options are not mutually exclusive.

#if ( defined(SAVE_DEBUG_IMAGES) && ! defined(N_DEBUG_IMAGES) ) || (! defined(SAVE_DEBUG_IMAGES) && defined(N_DEBUG_IMAGES))
#error "SAVE_DEBUG_IMAGES and N_DEBUG_IMAGES must be defined together"
#endif

#if ! defined(CALCULATE_SCORE_ON_CPU_DISABLE_GPU_PARTICLE) && defined(USE_OPTIMIZED_CPU_SCORE_CALCULATION)
// It would not make sense to define the use of the optimized calculation (which allocates sum buffers in teh comparison object)
// if we did not actually use those buffers
#define CALCULATE_SCORE_ON_CPU_DISABLE_GPU_PARTICLE
#endif

#endif