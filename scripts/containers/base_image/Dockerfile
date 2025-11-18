FROM ubuntu:22.04

# /groups/himesb/.config/Code\ -\ Insiders/User/globalStorage/ms-vscode-remote.remote-containers/cli-bin/code-insiders build --image-name bhimesbhimes/cistem_build_env:latest ./.devcontainer_build

# For compiling wxWidgets
ARG n_threads=16
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York
ARG GCC_VER=11



RUN apt-get update &&\
    apt-get -y install tzdata sudo &&\
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
    apt-get install -y locales &&\
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 &&\
    useradd -ms /bin/bash cisTEMdev &&\
    echo "cisTEMdev:cisTEMdev" | chpasswd && adduser cisTEMdev sudo &&\
    rm -rf /var/lib/apt/lists/*

ENV LANG en_US.utf8

# Install useful ubuntu packages (wxWidgets built from source below - NO system wx packages)
RUN apt-get --allow-releaseinfo-change update && apt-get install -y \
    apt-utils ca-certificates \
    libgtk2.0-dev libgtk-3-dev \
    libtool autoconf autotools-dev nano gedit meld cmake \
    libfftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget \
    build-essential git xauth zip unzip parallel sqlite3 python3 python3-pip curl gdb \
    strace htop valgrind lldb tree jq ripgrep fd-find tmux \
    linux-tools-generic sysstat shellcheck &&\
    rm -rf /var/lib/apt/lists/*


# ============================================================================
# INTEL MKL MINIMAL INSTALLATION FOR CONTAINERIZED DEPLOYMENT
# ============================================================================
# This section installs Intel MKL and configures it for either static or
# dynamic linking. Choose your linking strategy by passing argument to cleanup script.
#
# LINKING STRATEGY:
#   - DYNAMIC: Smaller binaries, requires .so files at runtime, faster iteration
#   - STATIC:  Larger binaries, self-contained, portable, no runtime dependencies
#
# For containerized deployment, EITHER approach works well.
# Choose based on your deployment model.
# ============================================================================

# Add Intel oneAPI repository and install MKL
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends intel-oneapi-mkl-devel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Similarl to CUDA we will clean up for the MK, we options for dynamic or static
# (Currently we leave the static libs)
COPY mkl_cleanup.sh /tmp/
RUN chmod +x /tmp/mkl_cleanup.sh && \
    /tmp/mkl_cleanup.sh static && \
    rm /tmp/mkl_cleanup.sh

    # Set MKL environment variables
# These are used by both static and dynamic linking
ENV MKLROOT=/opt/intel/oneapi/mkl/latest \
    TBBROOT=/opt/intel/oneapi/tbb/latest

    # LD_LIBRARY_PATH: Required for DYNAMIC linking at runtime (finding .so files)
# LIBRARY_PATH: Required for linking phase (both static and dynamic)
# CPATH: Required for compilation (finding header files)
ENV LD_LIBRARY_PATH=${MKLROOT}/lib/intel64:/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64:${TBBROOT}/lib/intel64/gcc4.8:${LD_LIBRARY_PATH} \
    LIBRARY_PATH=${MKLROOT}/lib/intel64:/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64:${LIBRARY_PATH} \
    CPATH=${MKLROOT}/include:${CPATH}

    # ============================================================================
# CONFIGURE.AC / MAKEFILE CHANGES FOR MKL LINKING
# ============================================================================
#
# The linking commands differ significantly between static and dynamic.
# Update your configure.ac or Makefile as follows:
#
# ----------------------------------------------------------------------------
# FOR DYNAMIC LINKING (simplest approach):
# ----------------------------------------------------------------------------
# In configure.ac, add to your LIBS or LDFLAGS:
#
#   MKL_LIBS="-L${MKLROOT}/lib/intel64 -lmkl_rt -lpthread -lm -ldl"
#   LIBS="$LIBS $MKL_LIBS"
#
# Or in Makefile:
#   LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -ldl
#
# The -lmkl_rt (runtime) library automatically loads the right kernels.
# No need to specify threading model explicitly.
#
# ----------------------------------------------------------------------------
# FOR STATIC LINKING (requires explicit library specification):
# ----------------------------------------------------------------------------
# In configure.ac or Makefile, you must specify the three-layer model:
#
# For GNU OpenMP threading (recommended for g++/gcc):
#   MKL_STATIC_LIBS="-Wl,--start-group \
#                    ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
#                    ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a \
#                    ${MKLROOT}/lib/intel64/libmkl_core.a \
#                    -Wl,--end-group -lgomp -lpthread -lm -ldl"
#
# For single-threaded (sequential):
#   MKL_STATIC_LIBS="-Wl,--start-group \
#                    ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
#                    ${MKLROOT}/lib/intel64/libmkl_sequential.a \
#                    ${MKLROOT}/lib/intel64/libmkl_core.a \
#                    -Wl,--end-group -lpthread -lm -ldl"
#
# For ILP64 interface (64-bit integers, arrays >2GB):
#   Add -DMKL_ILP64 to CFLAGS/CXXFLAGS
#   Use libmkl_intel_ilp64.a instead of libmkl_intel_lp64.a
#
# CRITICAL: The -Wl,--start-group ... -Wl,--end-group wrapper is REQUIRED
#           for static linking to resolve circular dependencies.
#
# ----------------------------------------------------------------------------
# EXAMPLE CONFIGURE.AC SNIPPET:
# ----------------------------------------------------------------------------
# AC_ARG_ENABLE([mkl-static],
#   [AS_HELP_STRING([--enable-mkl-static], [Use static MKL linking])],
#   [mkl_static=yes], [mkl_static=no])
#
# if test "x$mkl_static" = "xyes"; then
#   # Static linking
#   MKL_LIBS="-Wl,--start-group \
#             \$MKLROOT/lib/intel64/libmkl_intel_lp64.a \
#             \$MKLROOT/lib/intel64/libmkl_gnu_thread.a \
#             \$MKLROOT/lib/intel64/libmkl_core.a \
#             -Wl,--end-group -lgomp -lpthread -lm -ldl"
# else
#   # Dynamic linking
#   MKL_LIBS="-L\$MKLROOT/lib/intel64 -lmkl_rt -lpthread -lm -ldl"
# fi
# LIBS="$LIBS $MKL_LIBS"
#
# ----------------------------------------------------------------------------
# EXAMPLE MAKEFILE SNIPPET:
# ----------------------------------------------------------------------------
# # Set to 'yes' for static linking, 'no' for dynamic
# MKL_STATIC := no
#
# ifeq ($(MKL_STATIC),yes)
#   # Static linking
#   LDFLAGS += -Wl,--start-group \
#              $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a \
#              $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a \
#              $(MKLROOT)/lib/intel64/libmkl_core.a \
#              -Wl,--end-group -lgomp -lpthread -lm -ldl
# else
#   # Dynamic linking
#   LDFLAGS += -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -ldl
# endif
#
# ----------------------------------------------------------------------------
# INCLUDE FLAGS (same for both static and dynamic):
# ----------------------------------------------------------------------------
# CFLAGS += -I$(MKLROOT)/include
# CXXFLAGS += -I$(MKLROOT)/include
#
# ----------------------------------------------------------------------------
# RECOMMENDED APPROACH:
# ----------------------------------------------------------------------------
# Use DYNAMIC linking for development (simpler, faster iteration)
# Use STATIC linking for production deployment (self-contained)
#
# You can also use Intel's MKL Link Line Advisor for customized flags:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
# ============================================================================


# Install gh to work with automating interactions with the repo
RUN type -p wget >/dev/null || (apt update && apt-get install wget -y) &&\ 
    mkdir -p -m 755 /etc/apt/keyrings &&\ 
    wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null &&\ 
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg &&\ 
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null &&\ 
    apt update &&\ 
    sudo apt install -y gh &&\ 
    rm -rf /var/lib/apt/lists/*

# Install clang format 14 - available natively in Ubuntu 22.04
RUN apt-get update && apt-get install -y clang-format-14 clang-14 clang-tidy-14 &&\
    cd /usr/bin && ln -s clang-format-14 clang-format && ln -s clang-14 clang && ln -s clang-tidy-14 clang-tidy && ln -s clang++-14 clang++ &&\
    rm -rf /var/lib/apt/lists/*

# It looks like we are moving away form intel due to NVCC support, but figuring this out was a PITA so leaving for reference.
# # Pre-empt the intel function that checks intel or not at runtime so that optimal codepaths may be used on AMD procs.
# # This is only needed for dynamic builds. Note the lib is build in the next step.
# ENV LD_PRELOAD=/opt/intel/libfakeIntel.so

# wxWidgets builds moved to end of Dockerfile (after CUDA) to minimize layer invalidation during development


# Install cuda (when the web is live)
# NOTE: the MINIMUM driver verions is >=525.60.13
ARG CUDA_VER=12.3.2
ARG DRIVER_VER=545.23.08

# Install cuda (when the web is live)
RUN cd /tmp && wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/cuda_${CUDA_VER}_${DRIVER_VER}_linux.run &&\
    sh cuda_${CUDA_VER}_${DRIVER_VER}_linux.run --silent --toolkit &&\
    rm cuda_${CUDA_VER}_${DRIVER_VER}_linux.run &&\
    cd /usr/local/cuda/lib64 && rm -rf libcuspars* libcusolver* libcublasLt* libnppif* nsight-compute-* nsight-systems-* nsightee_plugins

# Note that for dynamic builds, the base container will be larger than the top layer because we defer removing the static cuda libs until the end

RUN echo 'alias lt="ls -lrth"' >> /home/cisTEMdev/.bashrc &&\
    echo 'alias dU="du -ch --max-depth=1 | sort -h"' >> /home/cisTEMdev/.bashrc &&\
    echo 'source /opt/intel/oneapi/setvars.sh' >> /home/cisTEMdev/.bashrc &&\
    echo 'bind "set bell-style none"' >> /home/cisTEMdev/.bashrc &&\
    echo 'export PATH=/usr/bin:/usr/local/cuda/bin:$PATH' >> /home/cisTEMdev/.bashrc

# =============================================================================
# wxWidgets Multi-Version, Multi-Compiler Build Matrix
# =============================================================================
# Build wxWidgets with all compiler variants for version migration testing
# Each installation is isolated to allow coexistence without conflicts
#
# Directory naming: /opt/WX/wx{version}-{compiler}-static-gtk{version}/
#
# Matrix:
#   - wxWidgets 3.0.5 (stable) with gtk2: icc, icpx, gcc, clang
#   - wxWidgets 3.2.8.1 (stable) with gtk3: icc, icpx, gcc, clang
#
# IMPORTANT: Do NOT create /usr/bin/wx-config symlinks - use explicit paths
#            in configure: --with-wx-config=/opt/WX/wx{ver}-{compiler}-static-gtk{ver}/bin/wx-config
# =============================================================================

# Download wxWidgets source tarballs
RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wx305.tar.bz2 &&\
    wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.2.8.1/wxWidgets-3.2.8.1.tar.bz2 -O /tmp/wx328.tar.bz2

# TODO: wxWebView

# Build wxWidgets 3.0.5 with all 2 compilers (gtk2)
RUN echo "=== Building wxWidgets 3.0.5 with 2 compilers (gtk2) [intel deprecated NVCC support] ===" &&\
    # GCC Compiler
    mkdir -p /opt/WX/wx305-gcc-static-gtk2 && tar -xjf /tmp/wx305.tar.bz2 -C /opt/WX/wx305-gcc-static-gtk2 &&\
    cd /opt/WX/wx305-gcc-static-gtk2/wxWidgets-3.0.5 &&\
    CXXFLAGS=-fPIC CFLAGS=-fPIC CXX=g++ CC=gcc ./configure --disable-precomp-headers \
        --prefix=/opt/WX/wx305-gcc-static-gtk2 --with-libnotify=no --disable-shared \
        --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin \
        --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma \
        --without-libjbig --with-gtk=2 --disable-sys-libs &&\
    make -j$n_threads && make install && cd /opt/WX && rm -rf wx305-gcc-static-gtk2/wxWidgets-3.0.5 &&\
    # Clang Compiler
    mkdir -p /opt/WX/wx305-clang-static-gtk2 && tar -xjf /tmp/wx305.tar.bz2 -C /opt/WX/wx305-clang-static-gtk2 &&\
    cd /opt/WX/wx305-clang-static-gtk2/wxWidgets-3.0.5 &&\
    CXXFLAGS=-fPIC CFLAGS=-fPIC CXX=clang++ CC=clang ./configure --disable-precomp-headers \
        --prefix=/opt/WX/wx305-clang-static-gtk2 --with-libnotify=no --disable-shared \
        --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin \
        --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma \
        --without-libjbig --with-gtk=2 --disable-sys-libs &&\
    make -j$n_threads && make install && cd /opt/WX && rm -rf wx305-clang-static-gtk2/wxWidgets-3.0.5 &&\
    rm -f /tmp/wx305.tar.bz2

# Build wxWidgets 3.2.8.1 with all 2 compilers (gtk3)
RUN echo "=== Building wxWidgets 3.2.8.1 with 2 compilers (gtk3) [intel deprecated NVCC support] ===" &&\
    # GCC Compiler
    mkdir -p /opt/WX/wx328-gcc-static-gtk3 && tar -xjf /tmp/wx328.tar.bz2 -C /opt/WX/wx328-gcc-static-gtk3 &&\
    cd /opt/WX/wx328-gcc-static-gtk3/wxWidgets-3.2.8.1 &&\
    CXXFLAGS=-fPIC CFLAGS=-fPIC CXX=g++ CC=gcc ./configure --disable-precomp-headers \
        --prefix=/opt/WX/wx328-gcc-static-gtk3 --with-libnotify=no --disable-shared \
        --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin \
        --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma \
        --without-libjbig --with-gtk=3 --disable-sys-libs &&\
    make -j$n_threads && make install && cd /opt/WX && rm -rf wx328-gcc-static-gtk3/wxWidgets-3.2.8.1 &&\
    # Clang Compiler
    mkdir -p /opt/WX/wx328-clang-static-gtk3 && tar -xjf /tmp/wx328.tar.bz2 -C /opt/WX/wx328-clang-static-gtk3 &&\
    cd /opt/WX/wx328-clang-static-gtk3/wxWidgets-3.2.8.1 &&\
    CXXFLAGS=-fPIC CFLAGS=-fPIC CXX=clang++ CC=clang ./configure --disable-precomp-headers \
        --prefix=/opt/WX/wx328-clang-static-gtk3 --with-libnotify=no --disable-shared \
        --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin \
        --with-zlib=builtin --with-expat=builtin --disable-compat28 --without-liblzma \
        --without-libjbig --with-gtk=3 --disable-sys-libs &&\
    make -j$n_threads && make install && cd /opt/WX && rm -rf wx328-clang-static-gtk3/wxWidgets-3.2.8.1 &&\
    rm -f /tmp/wx328.tar.bz2

# Apply longlong.h patch to all wxWidgets 3.0.5 installations
# This fixes compilation errors with newer compilers
RUN for wxdir in /opt/WX/wx305-*/; do \
        tf=$(tempfile) && \
        cp ${wxdir}include/wx-3.0/wx/longlong.h ${wxdir}include/wx-3.0/wx/longlong.h.orig && \
        awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' \
            ${wxdir}include/wx-3.0/wx/longlong.h.orig > $tf && \
        mv $tf ${wxdir}include/wx-3.0/wx/longlong.h && \
        chmod a+r ${wxdir}include/wx-3.0/wx/longlong.h; \
    done




