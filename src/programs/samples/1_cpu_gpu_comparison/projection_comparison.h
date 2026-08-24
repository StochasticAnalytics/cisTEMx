#ifndef SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_PROJECTION_COMPARISON_H_
#define SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_PROJECTION_COMPARISON_H_

void CPUvsGPUProjectionRunner(const wxString& temp_directory);
bool DoCPUvsGPUProjectionTest(const wxString& cistem_ref_dir, const wxString& temp_directory);

#if defined(cisTEM_EXPERIMENTAL_3d_TEXTURE_ENABLE) && defined(cisTEM_USING_FastFFT) && cisTEM_EXPERIMENTAL_3d_TEXTURE_TYPE != 0
bool DoTexturePreparationParityTest(const wxString& cistem_ref_dir, const wxString& temp_directory);
#endif

#endif