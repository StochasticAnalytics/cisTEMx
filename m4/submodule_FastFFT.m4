# Setup use of the FastFFT library
#
# FastFFT is distributed as a prebuilt, closed-source library: a compiled,
# relocatable-device-code object plus its host headers, staged in the build
# container at /opt/FastFFT (lib/FastFFT.o + include/). cisTEM device-links the
# prebuilt object and includes the prebuilt headers. There is no from-source
# build path — FastFFT source is not distributed.

AC_DEFUN([submodule_FastFFT],
[

use_FastFFT="yes"

AS_IF([test "x$want_cuda" = "xyes"], [

AC_ARG_ENABLE(FastFFT, AS_HELP_STRING([--disable-FastFFT],[Do not use the FastFFT library, even if present]), [
    if test "x$enableval" = "xno"; then
        use_FastFFT="no"
        AC_MSG_NOTICE([Not using the FastFFT Library b/c --disable-FastFFT is configured.])
    else
        AC_MSG_ERROR([FastFFT is enabled by default, if present. Specifying --enable-FastFFT breaks the configuration. If you want to disable FastFFT, please configure with --disable-FastFFT])
    fi
],
[
    # Not explicitly disabled: use the prebuilt library if it is staged in the container.
    AC_CHECK_FILE("/opt/FastFFT/lib/FastFFT.o",[use_FastFFT="yes"],[use_FastFFT="no"])
    if test "x$use_FastFFT" = "xyes"; then
        AC_DEFINE(cisTEM_USING_FastFFT, [], [Use the FastFFT library for GPU FFTs where appropriate.])
        AC_MSG_NOTICE([Using the FastFFT Library.])
        AC_DEFINE([CUFFTDX_DISABLE_RUNTIME_ASSERTS], [], [Define the CUFFTDX_DISABLE_RUNTIME_ASSERTS flag])
        libFastFFT_OBJECTS="/opt/FastFFT/lib/FastFFT.o "
    else
        AC_MSG_NOTICE([Not using the FastFFT Library b/c /opt/FastFFT/lib/FastFFT.o was not found.])
    fi
])
], [
    use_FastFFT="no"
    AC_MSG_NOTICE([Not using the FastFFT Library b/c --with-cuda is not configured.])
])
])
