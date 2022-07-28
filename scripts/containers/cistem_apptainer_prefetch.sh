#!/bin/bash


# TODO: setup so both the apptainer def and this file source the same blocks here

CISTEM_CACHE_DIR=${HOME}
mkdir -p ${CISTEM_CACHE_DIR}/.cistem_apptainer_cache

export wx_location=${CISTEM_CACHE_DIR}/.cistem_apptainer_cache/wxWidgets-3.0.5.tar.bz2
echo "Checking for cached wxWidgets at $wx_location"
if [[ -f $wx_location ]] ; then
    echo "Found cached wxWidgets, extracting" 
else
    echo "Cached wxWidgets not found, downloading"
    wget  https://github.com/wxWidgets/wxWidgets/releases/download/v3.0.5/wxWidgets-3.0.5.tar.bz2 -O $wx_location
fi

export CUDA_VER=11.7.0
export DRIVER_VER=515.43.04

# Install cuda (when the web is live)
export cuda_toolkit_location=${CISTEM_CACHE_DIR}/.cistem_apptainer_cache/cuda_toolkit_${CUDA_VER}_${DRIVER_VER}.run
echo "Checking for cached cuda toolkit at $cuda_toolkit_location"
if [[ -f $cuda_toolkit_location ]] ; then
    echo "Found cached cuda-toolkit, installing" 
else
    echo "Cached cuda-toolkit not found, downloading"
    wget  https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/cuda_${CUDA_VER}_${DRIVER_VER}_linux.run -O $cuda_toolkit_location
fi

export pytorch_location=${CISTEM_CACHE_DIR}/.cistem_apptainer_cache/libtorch-cxx11-abi-shared-with-deps-1.11.0+cu113.zip
if [[ -f $pytorch_location ]] ; then
    echo "Found cached cuda-toolkit, installing"
else
    echo "Cached cuda-toolkit not found, downloading"
    wget  https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcu113.zip -O $pytorch_location
fi