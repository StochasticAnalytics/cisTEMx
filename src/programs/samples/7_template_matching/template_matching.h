#ifndef SRC_PROGRAMS_SAMPLES_7_TEMPLATE_MATCHING_TEMPLATE_MATCHING_H_
#define SRC_PROGRAMS_SAMPLES_7_TEMPLATE_MATCHING_TEMPLATE_MATCHING_H_

// End-to-end template-matching pipeline tests driving the real match_template_gpu
// binary on the real reference data (PLASMONLABS_REF_IMAGES/TM_tests). See
// template_matching.cpp for the test list and the baseline read/write contract.
void TemplateMatchingPipelineRunner(const wxString& temp_directory);

bool DoApoferritinSearchRegressionTest(const wxString& cistem_ref_dir, const wxString& temp_directory);
bool DoBatchSizeInvarianceTest(const wxString& cistem_ref_dir, const wxString& temp_directory);

#endif
