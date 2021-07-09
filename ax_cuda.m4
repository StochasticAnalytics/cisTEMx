# ax_cuda.m4: An m4 macro to detect and configure Cuda
#
# Copyright © 2008 Frederic Chateau <frederic.chateau@cea.fr>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# As a special exception to the GNU General Public License, if you
# distribute this file as part of a program that contains a
# configuration script generated by Autoconf, you may include it under
# the same distribution terms that you use for the rest of that program.
#


#
# SYNOPSIS
#	AX_CUDA()
#
# DESCRIPTION
#	Checks the existence of Cuda binaries and libraries.
#	Options:
#	--with-cuda=(path|yes|no)
#		Indicates whether to use Cuda or not, and the path of a non-standard
#		installation location of Cuda if necessary.
#
#	This macro calls:
#		AC_SUBST(CUDA_CFLAGS)
#		AC_SUBST(CUDA_LIBS)
#		AC_SUBST(NVCC)
#		AC_SUBST(NVCCFLAGS)
#


AC_DEFUN([AX_CUDA],
[

# Default install for cuda
default_cuda_home_path="/usr/local/cuda"

# In Thrust 1.10 c++11 is deprecated. Not using those libs now, so squash warnings, but we should consider switching to a newer standard
AC_DEFINE(THRUST_IGNORE_DEPRECATED_CPP_11,true)

AC_MSG_NOTICE([Checking for gpu request and vars])
AC_ARG_WITH([cuda], AS_HELP_STRING([--with-cuda@<:@=yes|no|DIR@:>@], [prefix where cuda is installed (default=no)]),
[
	with_cuda=$withval
	if test "$withval" = "yes" ; then
		want_cuda="yes"
		cuda_home_path=$default_cuda_home_path
	else 
		if test "$withval" = "no" ; then
			want_cuda="no"
		else
			want_cuda="yes"
			cuda_home_path=$withval
		fi
	fi
	
], [ want_cuda="no"] )

if test "$want_cuda" = "yes" ; then

	# check that nvcc compiler is in the path
	if test -n "$cuda_home_path"
	then
	    nvcc_search_dirs="$PATH$PATH_SEPARATOR$cuda_home_path/bin"
	else
	    nvcc_search_dirs=$PATH
	fi

	AC_PATH_PROG([NVCC], [nvcc], [], [$nvcc_search_dirs])
	if test -n "$NVCC"
	then
		have_nvcc="yes"
	else
		have_nvcc="no"
	fi

	# test if nvcc version is >= 2.3
	NVCC_VERSION=`$NVCC --version | grep release | awk 'gsub(/,/, "") {print [$]5}'`
	AC_MSG_RESULT([nvcc version : $NVCC_VERSION $NVCC_VERSION])
  # I don't like relying on parsing strings, but this works fine for cuda 8-11.3
  is_cuda_ge_11=`echo $NVCC_VERSION | awk '{if([$]1 < 11.0) print 0 ; else print 1}'`
	
 	# we'll only use 64 bit arch
  	libdir=lib64

	# set CUDA flags for static compilation. This is required for cufft callbacks.
	if test -n "$cuda_home_path"
	then
	  CUDA_CFLAGS="-I$cuda_home_path/include  -I$cuda_home_path/samples/common/inc/"
      CUDA_LIBS="-L$cuda_home_path/$libdir -lcufft_static -lnppial_static -lnppist_static -lnppc_static -lcurand_static -lculibos -lcudart_static -lrt"
	else
	  CUDA_CFLAGS="-I/usr/local/cuda/include  -I/usr/local/cuda/samples/common/inc/"
	  CUDA_LIBS="-L/usr/local/cuda/$libdir -lcufft_static -lnppial_static -lnppist_static -lnppc_static -lcurand_static -lculibos -lcudart_static -lrt"
	fi


	saved_CPPFLAGS=$CPPFLAGS
	saved_LIBS=$LIBS
  	saved_CUDA_LIBS=$CUDA_LIBS

	# Env var CUDA_DRIVER_LIB_PATH can be used to set an alternate driver library path
	# this is usefull when building on a host where only toolkit (nvcc) is installed
	# and not driver. Driver libs must be placed in some location specified by this var.
	if test -n "$CUDA_DRIVER_LIB_PATH"
	then
	    CUDA_LIBS+=" -L$CUDA_DRIVER_LIB_PATH -lcuda"
	else
	    CUDA_LIBS+=" -lcuda"
	fi

	CPPFLAGS="$CPPFLAGS $CUDA_CFLAGS"
	LIBS="$LIBS $CUDA_LIBS"

	AC_LANG_PUSH(C)
	AC_MSG_CHECKING([for Cuda headers])
  	AC_MSG_NOTICE([cuda path is $cuda_home_path])
	AC_COMPILE_IFELSE(
	[
		AC_LANG_PROGRAM([@%:@include <cuda.h>], [])
	],
	[
		have_cuda_headers="yes"
		AC_MSG_RESULT([yes])
	],
	[
		have_cuda_headers="no"
		AC_MSG_RESULT([not found])
	])

	AC_MSG_CHECKING([for Cuda libraries])
	AC_LINK_IFELSE(
	[
		AC_LANG_PROGRAM([@%:@include <cuda.h>],
		[
			CUmodule cuModule;
			cuModuleLoad(&cuModule, "myModule.cubin");
			CUdeviceptr devPtr;
			CUfunction cuFunction;
			unsigned pitch, width = 250, height = 500;
			cuMemAllocPitch(&devPtr, &pitch,width * sizeof(float), height, 4);
			cuModuleGetFunction(&cuFunction, cuModule, "myKernel");
			cuFuncSetBlockShape(cuFunction, 512, 1, 1);
			cuParamSeti(cuFunction, 0, devPtr);
			cuParamSetSize(cuFunction, sizeof(devPtr));
			cuLaunchGrid(cuFunction, 100, 1);
		])
	],
	[
		have_cuda_libs="yes"
		AC_MSG_RESULT([yes])
	],
	[
		have_cuda_libs="no"
		AC_MSG_RESULT([not found])
	])
	AC_LANG_POP(C)

	CPPFLAGS=$saved_CPPFLAGS
	LIBS=$saved_LIBS
  	CUDA_LIBS=$saved_CUDA_LIBS
	
	if test "$have_cuda_headers" = "yes" -a "$have_cuda_libs" = "yes" -a "$have_nvcc" = "yes"
	then
		have_cuda="yes"
	else
		have_cuda="no"
		AC_MSG_ERROR([Cuda is requested but not available])
	fi
fi

# This is the code that will be generated at compile time and should be specified for the most used gpu 
target_arch=""
AC_ARG_WITH([target-gpu-arch], AS_HELP_STRING([--with-target-gpu-arch@<:@=60,61,70,75,80,86@:>@], [Primary architecture to compile for (default=86)]),
[
	if test "$withval" = "86" ; then target_arch=86 
	elif  test "$withval" = "80" ; then target_arch=80
	elif  test "$withval" = "75" ; then target_arch=75
	elif  test "$withval" = "70" ; then target_arch=70
	elif  test "$withval" = "61" ; then target_arch=61
	elif  test "$withval" = "60" ; then target_arch=60
	else
		AC_MSG_ERROR([Requested target-gpu-arch must be in 60,61,70,75,80,86, not $withval])
	fi
	
], [ target_arch="86"] )
AC_MSG_NOTICE([target gpu architecture is sm$target_arch])

# Default nvcc flags
NVCCFLAGS=" -ccbin $CXX"
NVCCFLAGS+=" --gpu-architecture=sm_$target_arch -gencode=arch=compute_$target_arch,code=compute_$target_arch"

# This is the oldest arch that will have JIT-able code g
oldest_arch=""
AC_ARG_WITH([oldest-gpu-arch], AS_HELP_STRING([--with-oldest-gpu-arch@<:@=60,61,70,75,80,86:>@], [Oldest architecture make compatible for (default=70)]),
[
	if test "$withval" = "86" ; then oldest_arch=86 
	elif  test "$withval" = "80" ; then oldest_arch=80
	elif  test "$withval" = "75" ; then oldest_arch=75
	elif  test "$withval" = "70" ; then oldest_arch=70
	elif  test "$withval" = "61" ; then oldest_arch=61
	elif  test "$withval" = "60" ; then oldest_arch=60
	else
		AC_MSG_ERROR([Requested target-oldest_arch must be in 60,61,70,75,80,86, not $withval])
	fi
	
], [ oldest_arch="70"] )
AC_MSG_NOTICE([oldest gpu architecture is sm$oldest_arch])

if test "$oldest_arch" -gt "$target_arch" ; then 
	AC_MSG_ERROR([Requested target-oldest_arch is greater than the target arch.]) 
else
	current_arch="60"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -lt "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi
	
	current_arch="61"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -lt "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi
	
	current_arch="70"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -lt "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	
	
	current_arch="75"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -lt "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi	
	
	current_arch="80"
	if test "$current_arch" -ge $oldest_arch && test "$current_arch" -lt "$target_arch" ; then
		NVCCFLAGS+=" -gencode=arch=compute_$current_arch,code=sm_$current_arch"
	fi		
		
fi

if test "x$is_cuda_ge_11" -eq "x1" ; then
  AC_MSG_NOTICE([CUDA >= 11.0, enabling --extra-device-vectorization])
  NVCCFLAGS+=" --extra-device-vectorization"
fi
  
#--extra-device-vectorization
NVCCFLAGS+=" --default-stream per-thread -m64 -O3 --use_fast_math  -Xptxas --warn-on-local-memory-usage,--warn-on-spills, --generate-line-info -std=c++17 -Xcompiler= -DGPU -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_DLFCN_H=1"

AC_ARG_ENABLE(gpu-cache-hints, AS_HELP_STRING([--disable-gpu-cache-hints],[Do not use the intrinsics for cache hints]),[
  if test "$enableval" = no; then
  	NVCCFLAGS+=" -Xcompiler= -DDISABLECACHEHINTS"
  	AC_MSG_NOTICE([Disabling cache hint intrinsics requiring CUDA 11 or newer])  	
  fi])
  
AC_SUBST(CUDA_LIBS)
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(NVCCFLAGS)
])
