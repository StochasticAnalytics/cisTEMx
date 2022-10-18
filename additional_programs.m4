

# These are programs not central to running cisTEM but that users may find helpful
AC_DEFUN([AX_ADDITIONAL_PROGRAMS],
[

build_apply_ctf="no"
AC_ARG_ENABLE(build-applyctf, AS_HELP_STRING([--enable-build-applyctf],[build applyctf  [default="no"]]),[
  if test "$enableval" = yes; then
    build_apply_ctf=yes
  	AC_MSG_NOTICE([Building applyctf])	
  fi
  ])  
AM_CONDITIONAL([ENABLE_APPLYCTF_AM], [test "x$build_apply_ctf" = "xyes"])

build_project3D="no"
AC_ARG_ENABLE(build-project3D, AS_HELP_STRING([--enable-build-project3D],[build project3D  [default="no"]]),[
  if test "$enableval" = yes; then
    build_project3D=yes
  	AC_MSG_NOTICE([Building project3D])	
  fi
  ])
AM_CONDITIONAL([ENABLE_PROJECT3D_AM], [test "x$build_project3D" = "xyes"])

build_calc_occ="no"
AC_ARG_ENABLE(build-calc-occ, AS_HELP_STRING([--enable-build-calc-occ],[build calc_occ  [default="no"]]),[
  if test "$enableval" = yes; then
    build_calc_occ=yes
  	AC_MSG_NOTICE([Building calc_occ])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CALCOCC_AM], [test "x$build_calc_occ" = "xyes"])

build_remove_outlier_pixels="no"
AC_ARG_ENABLE(build-remove-outlier-pixels, AS_HELP_STRING([--enable-build-remove-outlier-pixels],[build remove_outlier_pixels  [default="no"]]),[
  if test "$enableval" = yes; then
    build_remove_outlier_pixels=yes
  	AC_MSG_NOTICE([Building remove_outlier_pixels])	
  fi
  ])
AM_CONDITIONAL([ENABLE_REMOVEOUTLIERPIXELS_AM], [test "x$build_remove_outlier_pixels" = "xyes"])

build_resize="no"
AC_ARG_ENABLE(build-resize, AS_HELP_STRING([--enable-build-resize],[build resize  [default="no"]]),[
  if test "$enableval" = yes; then
    build_resize=yes
  	AC_MSG_NOTICE([Building resize])	
  fi
  ])
AM_CONDITIONAL([ENABLE_RESIZE_AM], [test "x$build_resize" = "xyes"])

build_resample="no" 
AC_ARG_ENABLE(build-resample, AS_HELP_STRING([--enable-build-resample],[build resample  [default="no"]]),[
  if test "$enableval" = yes; then
    build_resample=yes
  	AC_MSG_NOTICE([Building resample])	
  fi
  ])
AM_CONDITIONAL([ENABLE_RESAMPLE_AM], [test "x$build_resample" = "xyes"])

build_reset_mrc_header="no"
AC_ARG_ENABLE(build-reset-mrc-header, AS_HELP_STRING([--enable-build-reset-mrc-header],[build reset_mrc_header  [default="no"]]),[
  if test "$enableval" = yes; then
    build_reset_mrc_header=yes
  	AC_MSG_NOTICE([Building reset_mrc_header])	
  fi
  ])
AM_CONDITIONAL([ENABLE_RESETMRCHEADER_AM], [test "x$build_reset_mrc_header" = "xyes"])

build_estimate_dataset_ssnr="no"
AC_ARG_ENABLE(build-estimate-dataset-ssnr, AS_HELP_STRING([--enable-build-estimate-dataset-ssnr],[build estimate_dataset_ssnr  [default="no"]]),[
  if test "$enableval" = yes; then
    build_estimate_dataset_ssnr=yes
  	AC_MSG_NOTICE([Building estimate_dataset_ssnr])	
  fi
  ])
AM_CONDITIONAL([ENABLE_ESTIMATEDATASETSSNR_AM], [test "x$build_estimate_dataset_ssnr" = "xyes"])

build_montage="no"
AC_ARG_ENABLE(build-montage, AS_HELP_STRING([--enable-build-montage],[build montage  [default="no"]]),[
  if test "$enableval" = yes; then
    build_montage=yes
  	AC_MSG_NOTICE([Building montage])	
  fi
  ])
AM_CONDITIONAL([ENABLE_MONTAGE_AM], [test "x$build_montage" = "xyes"])

build_extract_particles="no"
AC_ARG_ENABLE(build-extract-particles, AS_HELP_STRING([--enable-build-extract-particles],[build extract_particles  [default="no"]]),[
  if test "$enableval" = yes; then
    build_extract_particles=yes
  	AC_MSG_NOTICE([Building extract_particles])	
  fi
  ])
AM_CONDITIONAL([ENABLE_EXTRACTPARTICLES_AM], [test "x$build_extract_particles" = "xyes"])

build_sum_all_mrc_files="no"
AC_ARG_ENABLE(build-sum-all-mrc-files, AS_HELP_STRING([--enable-build-sum-all-mrc-files],[build sum_all_mrc_files  [default="no"]]),[
  if test "$enableval" = yes; then
    build_sum_all_mrc_files=yes
  	AC_MSG_NOTICE([Building sum_all_mrc_files])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SUMALLMRCFILES_AM], [test "x$build_sum_all_mrc_files" = "xyes"])

build_sum_all_tif_files="no"
AC_ARG_ENABLE(build-sum-all-tif-files, AS_HELP_STRING([--enable-build-sum-all-tif-files],[build sum_all_tif_files  [default="no"]]),[
  if test "$enableval" = yes; then
    build_sum_all_tif_files=yes
  	AC_MSG_NOTICE([Building sum_all_tif_files])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SUMALLTIFFILES_AM], [test "x$build_sum_all_tif_files" = "xyes"])

build_apply_gain_ref="no"
AC_ARG_ENABLE(build-apply-gain-ref, AS_HELP_STRING([--enable-build-apply-gain-ref],[build apply_gain_ref  [default="no"]]),[
  if test "$enableval" = yes; then
    build_apply_gain_ref=yes
  	AC_MSG_NOTICE([Building apply_gain_ref])	
  fi
  ])
AM_CONDITIONAL([ENABLE_APPLYGAINREF_AM], [test "x$build_apply_gain_ref" = "xyes"])

build_scale_with_mask="no"
AC_ARG_ENABLE(build-scale-with-mask, AS_HELP_STRING([--enable-build-scale-with-mask],[build scale_with_mask  [default="no"]]),[
  if test "$enableval" = yes; then
    build_scale_with_mask=yes
  	AC_MSG_NOTICE([Building scale_with_mask])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SCALEWITHMASK_AM], [test "x$build_scale_with_mask" = "xyes"])

build_mag_distortion_correct="no" 
AC_ARG_ENABLE(build-mag-distortion-correct, AS_HELP_STRING([--enable-build-mag-distortion-correct],[build mag_distortion_correct  [default="no"]]),[
  if test "$enableval" = yes; then
    build_mag_distortion_correct=yes
  	AC_MSG_NOTICE([Building mag_distortion_correct])	
  fi
  ])
AM_CONDITIONAL([ENABLE_MAGDISTORTIONCORRECT_AM], [test "x$build_mag_distortion_correct" = "xyes"])

build_apply_mask="no" 
AC_ARG_ENABLE(build-apply-mask, AS_HELP_STRING([--enable-build-apply-mask],[build apply_mask  [default="no"]]),[
  if test "$enableval" = yes; then
    build_apply_mask=yes
  	AC_MSG_NOTICE([Building apply_mask])	
  fi
  ])
AM_CONDITIONAL([ENABLE_APPLYMASK_AM], [test "x$build_apply_mask" = "xyes"])

build_convert_tif_to_mrc="no" 
AC_ARG_ENABLE(build-convert-tif-to-mrc, AS_HELP_STRING([--enable-build-convert-tif-to-mrc],[build convert_tif_to_mrc  [default="no"]]),[
  if test "$enableval" = yes; then
    build_convert_tif_to_mrc=yes
  	AC_MSG_NOTICE([Building convert_tif_to_mrc])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CONVERTTIFTOMRC_AM], [test "x$build_convert_tif_to_mrc" = "xyes"])

build_remove_inf_and_nan=no 
AC_ARG_ENABLE(build-remove-inf-and-nan, AS_HELP_STRING([--enable-build-remove-inf-and-nan],[build remove_inf_and_nan  [default="no"]]),[
  if test "$enableval" = yes; then
    build_remove_inf_and_nan=yes
  	AC_MSG_NOTICE([Building remove_inf_and_nan])	
  fi
  ])
AM_CONDITIONAL([ENABLE_REMOVEINFANDNAN_AM], [test "x$build_remove_inf_and_nan" = "xyes"])

build_make_orth_views="no"
AC_ARG_ENABLE(build-make-orth-views, AS_HELP_STRING([--enable-build-make-orth-views],[build make_orth_views  [default="no"]]),[
  if test "$enableval" = yes; then
    build_make_orth_views=yes
  	AC_MSG_NOTICE([Building make_orth_views])	
  fi
  ])
AM_CONDITIONAL([ENABLE_MAKEORTHVIEWS_AM], [test "x$build_make_orth_views" = "xyes"])

build_sharpen_map="no" 
AC_ARG_ENABLE(build-sharpen-map, AS_HELP_STRING([--enable-build-sharpen-map],[build sharpen_map  [default="no"]]),[
  if test "$enableval" = yes; then
    build_sharpen_map=yes
  	AC_MSG_NOTICE([Building sharpen_map])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SHARPENMAP_AM], [test "x$build_sharpen_map" = "xyes"])

build_calculate_fsc="no" 
AC_ARG_ENABLE(build-calculate-fsc, AS_HELP_STRING([--enable-build-calculate-fsc],[build calculate_fsc  [default="no"]]),[
  if test "$enableval" = yes; then
    build_calculate_fsc=yes
  	AC_MSG_NOTICE([Building calculate_fsc])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CALCULATEFSC_AM], [test "x$build_calculate_fsc" = "xyes"])

build_make_size_map="no"
AC_ARG_ENABLE(build-make-size-map, AS_HELP_STRING([--enable-build-make-size-map],[build make_size_map  [default="no"]]),[
  if test "$enableval" = yes; then
    build_make_size_map=yes
  	AC_MSG_NOTICE([Building make_size_map])	
  fi
  ])
AM_CONDITIONAL([ENABLE_MAKESIZEMAP_AM], [test "x$build_make_size_map" = "xyes"])

build_convert_par_to_star="no"
AC_ARG_ENABLE(build-convert-par-to-star, AS_HELP_STRING([--enable-build-convert-par-to-star],[build convert_par_to_star  [default="no"]]),[
  if test "$enableval" = yes; then
    build_convert_par_to_star=yes
  	AC_MSG_NOTICE([Building convert_par_to_star])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CONVERTPARTOSTAR_AM], [test "x$build_convert_par_to_star" = "xyes"])

build_quick_test="no"
AC_ARG_ENABLE(build-quick-test, AS_HELP_STRING([--enable-build-quick-test],[build quick_test  [default="no"]]),[
  if test "$enableval" = yes; then
    build_quick_test=yes
  	AC_MSG_NOTICE([Building quick_test])	
  fi
  ])
AM_CONDITIONAL([ENABLE_QUICKTEST_AM], [test "x$build_quick_test" = "xyes"])

build_subtract_from_stack="no"
AC_ARG_ENABLE(build-subtract_from_stack, AS_HELP_STRING([--enable-build-subtract-from-stack],[build subtract-from-stack  [default="no"]]),[
  if test "$enableval" = yes; then
    build_subtract_from_stack=yes
  	AC_MSG_NOTICE([Building subtract_from_stack])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SUBTRACTFROMSTACK_AM], [test "x$build_subtract_from_stack" = "xyes"])

build_binarize="no"
AC_ARG_ENABLE(build-binarize, AS_HELP_STRING([--enable-build-binarize],[build binarize  [default="no"]]),[
  if test "$enableval" = yes; then
    build_binarize=yes
  	AC_MSG_NOTICE([Building binarize])	
  fi
  ])
AM_CONDITIONAL([ENABLE_BINARIZE_AM], [test "x$build_binarize" = "xyes"])

build_move_volume_xyz="no"
AC_ARG_ENABLE(build-move-volume-xyz, AS_HELP_STRING([--enable-build-move-volume-xyz],[build move_volume_xyz  [default="no"]]),[
  if test "$enableval" = yes; then
    build_move_volume_xyz=yes
  	AC_MSG_NOTICE([Building move_volume_xyz])	
  fi
  ])
AM_CONDITIONAL([ENABLE_MOVEVOLUMEXYZ_AM], [test "x$build_move_volume_xyz" = "xyes"])

build_symmetry_expand_stack_and_par="no"
AC_ARG_ENABLE(build-symmetry-expand-stack-and-par, AS_HELP_STRING([--enable-build-symmetry-expand-stack-and-par],[build symmetry_expand_stack_and_par  [default="no"]]),[
  if test "$enableval" = yes; then
    build_symmetry_expand_stack_and_par=yes
  	AC_MSG_NOTICE([Building symmetry_expand_stack_and_par])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SYMMETRYEXPANDSTACKANDPAR_AM], [test "x$build_symmetry_expand_stack_and_par" = "xyes"])

build_subtract_two_stacks="no"
AC_ARG_ENABLE(build-subtract-two-stacks, AS_HELP_STRING([--enable-build-subtract-two-stacks],[build subtract_two_stacks  [default="no"]]),[
  if test "$enableval" = yes; then
    build_subtract_two_stacks=yes
  	AC_MSG_NOTICE([Building subtract_two_stacks])	
  fi
  ])
AM_CONDITIONAL([ENABLE_SUBTRACTTWOSTACKS_AM], [test "x$build_subtract_two_stacks" = "xyes"])

build_add_two_stacks="no"
AC_ARG_ENABLE(build-add-two-stacks, AS_HELP_STRING([--enable-build-add-two-stacks],[build add_two_stacks  [default="no"]]),[
  if test "$enableval" = yes; then
    build_add_two_stacks=yes
  	AC_MSG_NOTICE([Building add_two_stacks])	
  fi
  ])
AM_CONDITIONAL([ENABLE_ADDTWOSTACKS_AM], [test "x$build_add_two_stacks" = "xyes"])

build_multiply_two_stacks="no"
AC_ARG_ENABLE(build-multiply-two-stacks, AS_HELP_STRING([--enable-build-multiply-two-stacks],[build multiply_two_stacks  [default="no"]]),[
  if test "$enableval" = yes; then
    build_multiply_two_stacks=yes
  	AC_MSG_NOTICE([Building multiply_two_stacks])	
  fi
  ])
AM_CONDITIONAL([ENABLE_MULTIPLYTWOSTACKS_AM], [test "x$build_multiply_two_stacks" = "xyes"])

build_divide_two_stacks="no"
AC_ARG_ENABLE(build-divide-two-stacks, AS_HELP_STRING([--enable-build-divide-two-stacks],[build divide_two_stacks  [default="no"]]),[
  if test "$enableval" = yes; then
    build_divide_two_stacks=yes
  	AC_MSG_NOTICE([Building divide_two_stacks])	
  fi
  ])
AM_CONDITIONAL([ENABLE_DIVIDETWOSTACKS_AM], [test "x$build_divide_two_stacks" = "xyes"])

build_invert_stack="no"
AC_ARG_ENABLE(build-invert-stack, AS_HELP_STRING([--enable-build-invert-stack],[build invert_stack  [default="no"]]),[
  if test "$enableval" = yes; then
    build_invert_stack=yes
  	AC_MSG_NOTICE([Building invert_stack])	
  fi
  ])
AM_CONDITIONAL([ENABLE_INVERTSTACK_AM], [test "x$build_invert_stack" = "xyes"])

build_align_coordinates="no"
AC_ARG_ENABLE(build-align-coordinates, AS_HELP_STRING([--enable-build-align-coordinates],[build align_coordinates  [default="no"]]),[
  if test "$enableval" = yes; then
    build_align_coordinates=yes
  	AC_MSG_NOTICE([Building align_coordinates])	
  fi
  ])
AM_CONDITIONAL([ENABLE_ALIGNCOORDINATES_AM], [test "x$build_align_coordinates" = "xyes"])

build_find_dqe="no"
AC_ARG_ENABLE(build-find-dqe, AS_HELP_STRING([--enable-build-find-dqe],[build find_dqe  [default="no"]]),[
  if test "$enableval" = yes; then
    build_find_dqe=yes
  	AC_MSG_NOTICE([Building find_dqe])	
  fi
  ])
AM_CONDITIONAL([ENABLE_FINDDQE_AM], [test "x$build_find_dqe" = "xyes"])

build_combine_via_max="no"
AC_ARG_ENABLE(build-combine-via-max, AS_HELP_STRING([--enable-build-combine-via-max],[build combine_via_max  [default="no"]]),[
  if test "$enableval" = yes; then
    build_combine_via_max=yes
  	AC_MSG_NOTICE([Building combine_via_max])	
  fi
  ])
AM_CONDITIONAL([ENABLE_COMBINEVIAMAX_AM], [test "x$build_combine_via_max" = "xyes"])

build_remove_relion_stripes="no"
AC_ARG_ENABLE(build-remove-relion-stripes, AS_HELP_STRING([--enable-build-remove-relion-stripes],[build remove_relion_stripes  [default="no"]]),[
  if test "$enableval" = yes; then
    build_remove_relion_stripes=yes
  	AC_MSG_NOTICE([Building remove_relion_stripes])	
  fi
  ])
AM_CONDITIONAL([ENABLE_REMOVERELIONSTRIPES_AM], [test "x$build_remove_relion_stripes" = "xyes"])

build_create_mask="no"
AC_ARG_ENABLE(build-create-mask, AS_HELP_STRING([--enable-build-create-mask],[build create_mask  [default="no"]]),[
  if test "$enableval" = yes; then
    build_create_mask=yes
  	AC_MSG_NOTICE([Building create_mask])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CREATEMASK_AM], [test "x$build_create_mask" = "xyes"])

build_invert_hand="no"
AC_ARG_ENABLE(build-invert-hand, AS_HELP_STRING([--enable-build-invert-hand],[build invert_hand  [default="no"]]),[
  if test "$enableval" = yes; then
    build_invert_hand=yes
  	AC_MSG_NOTICE([Building invert_hand])	
  fi
  ])
AM_CONDITIONAL([ENABLE_INVERTHAND_AM], [test "x$build_invert_hand" = "xyes"])

build_append_stacks="no"
AC_ARG_ENABLE(build-append-stacks, AS_HELP_STRING([--enable-build-append-stacks],[build append_stacks  [default="no"]]),[
  if test "$enableval" = yes; then
    build_append_stacks=yes
  	AC_MSG_NOTICE([Building append_stacks])	
  fi
  ])
AM_CONDITIONAL([ENABLE_APPENDSTACKS_AM], [test "x$build_append_stacks" = "xyes"])

build_convert_star_to_binary="no"
AC_ARG_ENABLE(build-convert-star-to-binary, AS_HELP_STRING([--enable-build-convert-star-to-binary],[build convert_star_to_binary  [default="no"]]),[
  if test "$enableval" = yes; then
    build_convert_star_to_binary=yes
  	AC_MSG_NOTICE([Building convert_star_to_binary])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CONVERTSTARTOBINARY_AM], [test "x$build_convert_star_to_binary" = "xyes"])

build_convert_binary_to_star="no"
AC_ARG_ENABLE(build-convert-binary-to-star, AS_HELP_STRING([--enable-build-convert-binary-to-star],[build convert_binary_to_star  [default="no"]]),[
  if test "$enableval" = yes; then
    build_convert_binary_to_star=yes
  	AC_MSG_NOTICE([Building convert_binary_to_star])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CONVERTBINARYTOSTAR_AM], [test "x$build_convert_binary_to_star" = "xyes"])

build_convert_eer_to_mrc="no"
AC_ARG_ENABLE(build-convert-eer-to-mrc, AS_HELP_STRING([--enable-build-convert-eer-to-mrc],[build convert_eer_to_mrc  [default="no"]]),[
  if test "$enableval" = yes; then
    build_convert_eer_to_mrc=yes
  	AC_MSG_NOTICE([Building convert_eer_to_mrc])	
  fi
  ])
AM_CONDITIONAL([ENABLE_CONVERTEERTOMRC_AM], [test "x$build_convert_eer_to_mrc" = "xyes"])

build_azimuthal_average="no"
AC_ARG_ENABLE(build-azimuthal-average, AS_HELP_STRING([--enable-build-azimuthal-average],[build azimuthal_average  [default="no"]]),[
  if test "$enableval" = yes; then
    build_azimuthal_average=yes
  	AC_MSG_NOTICE([Building azimuthal_average])	
  fi
  ])
AM_CONDITIONAL([ENABLE_AZIMUTHALAVERAGE_AM], [test "x$build_azimuthal_average" = "xyes"])

build_normalize_stack="no"
AC_ARG_ENABLE(build-normalize-stack, AS_HELP_STRING([--enable-build-normalize-stack],[build normalize_stack  [default="no"]]),[
  if test "$enableval" = yes; then
    build_normalize_stack=yes
  	AC_MSG_NOTICE([Building normalize_stack])	
  fi
  ])
AM_CONDITIONAL([ENABLE_NORMALIZESTACK_AM], [test "x$build_normalize_stack" = "xyes"])

build_print_stack_statistics="no"
AC_ARG_ENABLE(build-print-stack-statistics, AS_HELP_STRING([--enable-build-print-stack-statistics],[build print_stack_statistics  [default="no"]]),[
  if test "$enableval" = yes; then
    build_print_stack_statistics=yes
  	AC_MSG_NOTICE([Building print_stack_statistics])	
  fi
  ])
AM_CONDITIONAL([ENABLE_PRINTSTACKSTATISTICS_AM], [test "x$build_print_stack_statistics" = "xyes"])

build_combine_stacks_by_star="no"
AC_ARG_ENABLE(build-combine-stacks-by-star, AS_HELP_STRING([--enable-build-combine-stacks-by-star],[build combine_stacks_by_star  [default="no"]]),[
  if test "$enableval" = yes; then
    build_combine_stacks_by_star=yes
  	AC_MSG_NOTICE([Building combine_stacks_by_star])	
  fi
  ])
AM_CONDITIONAL([ENABLE_COMBINESTACKSBYSTAR_AM], [test "x$build_combine_stacks_by_star" = "xyes"])



])