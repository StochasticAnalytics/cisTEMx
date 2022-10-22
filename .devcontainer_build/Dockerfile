FROM bhimesbhimes/cistem_build_env:base_image_v1.4.5

# some rebuild comment

ENV CISTEM_REF_IMAGES /cistem_reference_images

# Get reference images for testing and debugging
RUN pip3 install gdown && cd / && gdown https://drive.google.com/drive/folders/1PO9tBU7lKqKUuSb8qdEPvlEk7xeektXv?usp --folder


# # Install Node 16
# RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
#     apt-get update && apt-get install -y nodejs && \
#     rm -rf /var/lib/apt/lists/*  && \
#     npm install -g pnpm

# TODO: move this to the base container once we confirm dynamic linking is working okay
# Use the basename bit to ensure no rm -rf foibles with root dir in empyt string case
RUN ls /usr/local/cuda/targets/x86_64-linux/lib/lib*_static.a | grep -v cufft_static.a | while read a; do rm -rf /usr/local/cuda/targets/x86_64-linux/lib/$(basename $a); done
RUN rm -f /usr/local/cuda/targets/x86_64-linux/lib/libcufft_static_nocallback.a


# Install cutensor
ARG CUTENSOR_VERSION=libcutensor-linux-x86_64-1.6.0.3-archive
ARG CUTENSOR_LIB=11
RUN cd /tmp &&   wget https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${CUTENSOR_VERSION}.tar.xz && \
    mkdir /opt/cuTensor && tar -xf ${CUTENSOR_VERSION}.tar.xz --strip 1 -C /opt/cuTensor && \
    cd /opt/cuTensor/lib && ln -s ${CUTENSOR_LIB} cistem_version && \
    rm -rf 10.2 ${CUTENSORLIB}/libcuten*_static.a && \
    echo 'export CUTENSOR_ROOT=/opt/cuTensor' >> /home/cisTEMx/.bashrc && \ 
    echo 'export LD_LIBRARY_PATH=${CUTENSOR_ROOT}/lib/cistem_version/:${LD_LIBRARY_PATH}' >> /home/cisTEMx/.bashrc 



USER cisTEMx
WORKDIR /home/cisTEMx

