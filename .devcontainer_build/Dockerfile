FROM bhimesbhimes/cistem_build_env:base_image_v1.5.0

# some rebuild comment

ENV CISTEM_REF_IMAGES /cisTEMx/cistem_reference_images

# Get reference images for testing and debugging
RUN mkdir -p /cisTEMx && pip3 install gdown && cd /cisTEMx && gdown https://drive.google.com/drive/folders/1PO9tBU7lKqKUuSb8qdEPvlEk7xeektXv?usp --folder


# # Install Node 16
# RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
#     apt-get update && apt-get install -y nodejs && \
#     rm -rf /var/lib/apt/lists/*  && \
#     npm install -g pnpm





USER cisTEMx
WORKDIR /home/cisTEMx

