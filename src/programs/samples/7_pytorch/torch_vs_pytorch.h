#ifndef SRC_PROGRAMS_SAMPLES_7_PYTORCH_TORCH_VS_PYTORCH_H_
#define SRC_PROGRAMS_SAMPLES_7_PYTORCH_TORCH_VS_PYTORCH_H_

#include <torch/torch.h>

void TorchVsPytorchRunner(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);
bool TorchVsPytorch(const wxString& hiv_image_80x80x1_filename, wxString& temp_directory);

#endif