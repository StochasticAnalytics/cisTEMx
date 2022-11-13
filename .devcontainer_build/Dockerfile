FROM bhimesbhimes/cistem_build_env:base_image_v1.5.0

# some rebuild comment

ENV CISTEM_REF_IMAGES /cisTEMx/cistem_reference_images

# Get reference images for testing and debugging
RUN mkdir -p /cisTEMx && pip3 install gdown && cd /cisTEMx && gdown https://drive.google.com/drive/folders/1PO9tBU7lKqKUuSb8qdEPvlEk7xeektXv?usp --folder


# Install pytorch
# libtorch
ARG LIBTORCH_VERSION=13.0
ARG LIBTORCH_CUDA_VERSION=117

# NOTE: after leaving base_iamge:v1.5.0 the rm -rf /opt/libtorch will no longer be needed
RUN cd /tmp && wget https://download.pytorch.org/libtorch/cu${LIBTORCH_CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-1.13.0%2Bcu${LIBTORCH_CUDA_VERSION}.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.${LIBTORCH_VERSION}+cu${LIBTORCH_CUDA_VERSION}.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.${LIBTORCH_VERSION}+cu${LIBTORCH_CUDA_VERSION}.zip && \
    rm -rf /opt/libtorch && \
    mv libtorch /opt
# python version
# Install pytorch
ARG TORCHVISION_VERSION=14.0
RUN pip install torch==1.${LIBTORCH_VERSION}+cu${LIBTORCH_CUDA_VERSION} \
    torchvision==0.${TORCHVISION_VERSION}+cu${LIBTORCH_CUDA_VERSION} \
    torchaudio==0.${LIBTORCH_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu${LIBTORCH_CUDA_VERSION}



# # Install Node 16
# RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
#     apt-get update && apt-get install -y nodejs && \
#     rm -rf /var/lib/apt/lists/*  && \
#     npm install -g pnpm





USER cisTEMx
WORKDIR /home/cisTEMx

