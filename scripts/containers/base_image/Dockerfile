FROM ubuntu:20.04

# /groups/himesb/.config/Code\ -\ Insiders/User/globalStorage/ms-vscode-remote.remote-containers/cli-bin/code-insiders build --image-name bhimesbhimes/cistem_build_env:latest ./.devcontainer_build

# For compiling wxWidgets
ARG n_threads=16

# By default this will create cisTEMx as uid==1000 which is not what we want as it is also going to be the user id
# of a host user when building the top layer with singularity. < 1000 is reserved for system users. 814 is a somewhat random choice.
RUN useradd --uid 814 -ms /bin/bash cisTEMx
RUN echo "cisTEMx:cisTEMx" | chpasswd && adduser cisTEMx sudo

# Install useful ubuntu packages libgtk2.0-dev
RUN apt-get --allow-releaseinfo-change update && apt-get install -y \
    libgtk-3-dev libwxgtk3.0-gtk3-dev libwxbase3.0-dev \
    libtool autoconf autotools-dev nano gedit meld cmake \
    libfftw3-dev libtiff-dev software-properties-common libffi-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev libjpeg-dev libtiff-dev \
    libreadline-dev liblzma-dev libssl-dev libncursesw5-dev wget \
    build-essential git xauth zip unzip parallel sqlite3 python3 python3-pip curl gdb  \
    && rm -rf /var/lib/apt/lists/*

# Pre-empt the intel function that checks intel or not at runtime so that optimal codepaths may be used on AMD procs.
RUN mkdir /opt/intel && echo 'int mkl_serv_intel_cpu_true() {return 1;}' > /opt/intel/fakeIntel.c && \
    gcc -shared -fPIC -o /opt/intel/libfakeIntel.so /opt/intel/fakeIntel.c 

ENV LD_PRELOAD=/opt/intel/libfakeIntel.so    

# Get the MKL and intel compiler: note, this is 19G by default, will try to determine minimal set needed huge waste but works well enough for now, final size is ~5gb

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && apt-get install -y intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic-2021.4.0 intel-oneapi-mkl-2021.4.0 intel-oneapi-mkl-devel-2021.4.0 \
    && cd /opt/intel/oneapi/ && rm -rf debugger/ conda_channel/ mkl/latest/lib/intel64/*_sycl* \
    && rm -rf /var/lib/apt/lists/* 


# Install wxWidgets 3.1.5

RUN wget -qO - https://repos.codelite.org/CodeLite.asc | apt-key add - && \
    apt-add-repository 'deb https://repos.codelite.org/wx3.1.5/ubuntu/ groovy universe' && \
    apt-get update && apt-get install -y libwxbase3.1-0-unofficial \
    libwxbase3.1unofficial-dev \
    libwxgtk3.1-0-unofficial \
    libwxgtk3.1unofficial-dev \
    wx3.1-headers \
    wx-common \
    libwxgtk-media3.1-0-unofficial \
    libwxgtk-media3.1unofficial-dev \
    libwxgtk-webview3.1-0-unofficial \
    libwxgtk-webview3.1unofficial-dev \
    libwxgtk-webview3.1-0-unofficial-dbg \
    libwxbase3.1-0-unofficial-dbg \
    libwxgtk3.1-0-unofficial-dbg \
    libwxgtk-media3.1-0-unofficial-dbg \
    && rm -rf /var/lib/apt/lists/*

# RUN wget -q https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O /tmp/wxwidgets.tar.bz2 && \
#     echo 'Installing wxWidgets' && \
#     tar -xf /tmp/wxwidgets.tar.bz2 -C /tmp && \
#     cd /tmp/wxWidgets-3.0.5  && \
#     mkdir /opt/WX 

# # Now build with out static intel
# RUN . /opt/intel/oneapi/setvars.sh  && cd /tmp/wxWidgets-3.0.5 && CXX=icpc CC=icc CXXFLAGS=-fPIC CFLAGS=-fPIC  ./configure --disable-precomp-headers --prefix=/opt/WX/intel-dynamic --with-libnotify=no \
#     --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin \
#     --disable-compat28 --without-liblzma --without-libjbig --with-gtk=3 --disable-sys-libs  && \
#     make -j$n_threads && \
#     make install 

# Install wxFormbuilder
RUN cd /tmp && wget https://github.com/wxFormBuilder/wxFormBuilder/releases/download/v3.10.0/wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && \
    apt-get update && apt install -y ./wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && rm wxformbuilder_3.10.0_ubuntu-20.04_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# Install clang format
# 14 - these goa t the end of /etc/apt/sources.list
RUN echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-14 main" | tee -a /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add - && \
    apt-get update && apt-get install -y clang-format-14 -y && \
    cd /usr/bin && ln -s clang-format-14 clang-format \
    && rm -rf /var/lib/apt/lists/*

# Install pytorch
RUN cd /tmp && wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip && \
    mv libtorch /opt

# Install cutensor
ARG CUTENSOR_VERSION=libcutensor-linux-x86_64-1.6.0.3-archive
ARG CUTENSOR_LIB=11
RUN cd /tmp &&   wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${CUTENSOR_VERSION}.tar.xz && \
    mkdir /opt/cuTensor && tar -xf ${CUTENSOR_VERSION}.tar.xz --strip 1 -C /opt/cuTensor && \
    cd /opt/cuTensor/lib && \
    ls -d | grep -v ${CUTENSOR_VERSION} | grep -v "." | grep -v ".." | while read not_version; do rm -rf /opt/cuTensor/lib/${not_version} ; done && \
    ln -s ${CUTENSOR_LIB} cistem_version && \
    rm -f ${CUTENSOR_LIB}/libcuten*_static.a && \
    echo 'export CUTENSOR_ROOT=/opt/cuTensor' >> /home/cisTEMx/.bashrc 

# Include the lib path in LD_RUN_PATH so on linking, the correct path is known 

ENV LD_RUN_PATH=/opt/cuTensor/lib/cistem_version:/usr/local/cuda/lib64:/opt/libtorch/lib:${LD_RUN_PATH}


# Install cuda (when the web is live)
ARG CUDA_VER=11.7.0
ARG DRIVER_VER=515.43.04
# Install cuda (when the web is live)
RUN cd /tmp && wget https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/cuda_${CUDA_VER}_${DRIVER_VER}_linux.run && \
    sh cuda_${CUDA_VER}_${DRIVER_VER}_linux.run --silent --toolkit && \
    rm cuda_${CUDA_VER}_${DRIVER_VER}_linux.run && \
    cd /usr/local/cuda/lib64 && rm libcuspars* libcusolver*

# TODO: move this to the base container once we confirm dynamic linking is working okay
# Use the basename bit to ensure no rm -rf foibles with root dir in empyt string case
RUN ls /usr/local/cuda/lib64/lib*_static.a | grep -v cufft_static.a | while read a; do rm -rf /usr/local/cuda/lib64/$(basename $a); done && \
    rm -rf /usr/local/cuda/lib64/libcufft_static_nocallback.a

RUN echo 'alias lt="ls -lrth"' >> /home/cisTEMx/.bashrc && \
    echo 'alias dU="du -ch --max-depth=1 | sort -h"' >> /home/cisTEMx/.bashrc && \
    echo 'source /opt/intel/oneapi/setvars.sh' >> /home/cisTEMx/.bashrc && \
    echo 'bind "set bell-style none"' >> /home/cisTEMx/.bashrc && \
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/cisTEMx/.bashrc 

# # First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
# RUN tf=`tempfile` && cp /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h.orig && \
#     awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h.orig > $tf && \
#     mv $tf /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h && \
#     chmod a+r /opt/WX/intel-dynamic/include/wx-3.0/wx/longlong.h

# First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
RUN tf=`tempfile` && cp /usr/include/wx-3.1-unofficial/wx/longlong.h /usr/include/wx-3.1-unofficial/wx/longlong.h.orig && \
    awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /usr/include/wx-3.1-unofficial/wx/longlong.h.orig > $tf && \
    mv $tf /usr/include/wx-3.1-unofficial/wx/longlong.h && \
    chmod a+r /usr/include/wx-3.1-unofficial/wx/longlong.h




