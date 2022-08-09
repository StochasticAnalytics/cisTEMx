FROM bhimesbhimes/cistem_build_env:base_image_v1.2


# Get reference images for testing and debugging
RUN pip3 install gdown && cd / && gdown https://drive.google.com/drive/folders/1PO9tBU7lKqKUuSb8qdEPvlEk7xeektXv?usp --folder
ENV CISTEM_REF_IMAGES /cistem_reference_images

# Install newest gcc
# RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
#     apt-get --allow-releaseinfo-change  update && apt install -y gcc-${GCC_VER} g++-${GCC_VER} \
#     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VER} 100 --slave \
#     /usr/bin/g++ g++ /usr/bin/g++-${GCC_VER} --slave /usr/bin/gcov gcov /usr/bin/gcov-${GCC_VER} \
#     && rm -rf /var/lib/apt/lists/*


# Here for the record if you want to build and link static binaries to avoid the cointainerized distribution
# RUN . /opt/intel/oneapi/setvars.sh  &&  cd /tmp/wxWidgets-3.0.5 && CXX=icpc CC=icc CXXFLAGS=-fPIC CFLAGS=-fPIC  ./configure --disable-precomp-headers --prefix=/opt/WX/intel-static --with-libnotify=no --disable-shared \
#     --without-gtkprint --with-libjpeg=builtin --with-libpng=builtin --with-libtiff=builtin --with-zlib=builtin --with-expat=builtin \
#     --disable-compat28 --without-liblzma --without-libjbig --with-gtk=2 --disable-sys-libs  && \
#     make -j$n_threads && \
#     make install && make clean 



# # Install Node 16
# RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
#     apt-get update && apt-get install -y nodejs && \
#     rm -rf /var/lib/apt/lists/*  && \
#     npm install -g pnpm


# Install cutensor
ARG CUTENSOR_VERSION=libcutensor-linux-x86_64-1.6.0.3-archive
ARG CUTENSOR_LIB=11
RUN cd /tmp &&   wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${CUTENSOR_VERSION}.tar.xz && \
    mkdir /opt/cuTensor && tar -xf ${CUTENSOR_VERSION}.tar.xz --strip 1 -C /opt/cuTensor && \
    cd /opt/cuTensor/lib && ln -s ${CUTENSOR_LIB} cistem_version && \
    echo 'export CUTENSOR_ROOT=/opt/cuTensor' >> /home/cisTEMx/.bashrc && \ 
    echo 'export LD_LIBRARY_PATH=${CUTENSOR_ROOT}/lib/${cistem_version}/:${LD_LIBRARY_PATH}' >> /home/cisTEMx/.bashrc 



USER cisTEMx
WORKDIR /home/cisTEMx

