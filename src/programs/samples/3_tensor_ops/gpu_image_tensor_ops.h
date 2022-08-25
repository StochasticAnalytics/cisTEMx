#ifndef _SRC_PROGRAMS_SAMPLES_3_TENSOR_OPS_GPU_IMAGE_TENSOR_OPS_H_
#define _SRC_PROGRAMS_SAMPLES_3_TENSOR_OPS_GPU_IMAGE_TENSOR_OPS_H_

void BasicTensorOpsRunner(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory);
bool TestCudaSample(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory);
bool TestTensorManagerManual(const wxString& hiv_images_80x80x10_filename, wxString& temp_directory);

#endif
