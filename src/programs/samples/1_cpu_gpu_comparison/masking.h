#ifndef SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_MASKING_COMPARISON_H_
#define SRC_PROGRAMS_SAMPLES_1_CPU_GPU_COMPARISON_MASKING_COMPARISON_H_

bool CPUvsGPUMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);
bool DoCosineMaskingTest(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);
bool DoTestCuTensorCompilation(const wxString& hiv_image_80x80x1_filename, const wxString& temp_directory);
bool DoTestCuTensorReduction(const wxString& hiv_image_80x80x1_filename, const wxString& temp_directory);

#endif