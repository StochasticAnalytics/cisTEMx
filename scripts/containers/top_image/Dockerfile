# syntax=docker/dockerfile:1
FROM fake_repo

# To allow a variety of common options, arguments may be passed to the build_container script, 
# which have the side-effect that this top-level layer is rebuilt without cache every time.
#
# Generally, the thinking is that all top-level additions are relatively cheap to build, although this may
# not be quite true for WX.

# Options are still somewhat restricted
#   build-type = linking is static or dynamic
#   build-compiler = icpc or g++
#   build-wx-version = old (3.0.5) or new (3.1.5) NOTE: we probably want to test 3.2 instead
#   build-npm = false [ default ] or true if specified
#   build-ref-images = false [ default ] or true if specified
ARG n_threads=12
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York
ARG GCC_VER=11
# note "-" in the variable seems to break the conditional statements.
ARG build_type="static"
ARG build_compiler="icpc"
ARG build_wx_version="stable"
ARG build_npm="false"
ARG build_ref_images="false"
ARG build_claude="false"
ARG build_libtorch="true"
ARG build_docs="true"

SHELL ["/bin/bash", "-c"]
# some rebuild comment
ENV CISTEM_REF_IMAGES=/cisTEMdev/cistem_reference_images

# Python 3.10 is native to Ubuntu 22.04, set up virtual environment at /opt/venv
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev && \
    python3.10 -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel && \
    chmod -R a+rX /opt/venv && \
    rm -rf /var/lib/apt/lists/*

# Default shell uses the venv Python for subsequent RUN steps
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Verify
RUN python --version && pip --version

# Install wxWidgets and Bear
COPY install_wx_3.1.5.sh install_node_16.sh install_node_22_and_claude.sh requirements.txt install_libtorch.sh install_documentation_tooling.sh install_bear.sh install_llm_optimization_tools.sh /tmp/


# wxWidgets installations - multiple versions/compilers available in base container
# DO NOT create /usr/bin/wx-config symlink - use explicit --with-wx-config paths
# Available installations:
#   wx305-{icc,icpx,gcc,clang}-static-gtk2 (3.0.5 stable with gtk2)
#   wx328-{icc,icpx,gcc,clang}-static-gtk3 (3.2.8.1 stable with gtk3)
# OBSOLETE: install_wx_3.1.5.sh installs system wx packages - DO NOT USE (set build_wx_version != "dev")
# wxWidgets 3.0.5 and 3.2.8.1 are now built from source in base container
RUN echo "wxWidgets multi-version/multi-compiler matrix available in /opt/WX/" && \
    if [[ "x${build_wx_version}" == "xdev" ]]; then echo "ERROR: build_wx_version=dev is obsolete and installs system wx packages!" && exit 1; fi

# Install Python packages from requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install Bear for compilation database generation (needed for clang-tidy)
# Note: shellcheck is already available in base image
# TODO: Move Bear installation to base container in future update
RUN /tmp/install_bear.sh

# Get reference images for testing and debugging
RUN mkdir -p /opt && cd /opt && gdown --fuzzy https://drive.google.com/file/d/1QqeDbv9dUrl6FertvKbDukJ6wjG2PsDi/view?usp=sharing && unzip FastFFT.zip && rm FastFFT.zip
RUN if [[ "x${build_ref_images}" == "xtrue" ]]; then mkdir -p /cisTEMdev && cd /cisTEMdev && gdown --fuzzy https://drive.google.com/file/d/197sE_pO4FWmjCo0zlqRJXAxN2BLbmHS_/view?usp=sharing && tar -xjvf cistem_reference_images_fp32.tar.bz2 && rm cistem_reference_images_fp32.tar.bz2 ;fi


# Install wxFormBuilder v4.2.1 from SourceForge pre-built .deb for Ubuntu 22.04
RUN wget -q https://sourceforge.net/projects/wxformbuilder.mirror/files/v4.2.1/wxformbuilder_4.2.1_ubuntu-22.04_amd64.deb/download -O /tmp/wxformbuilder.deb && \
    apt-get update && \
    apt-get install -y /tmp/wxformbuilder.deb || apt-get install -f -y && \
    rm -f /tmp/wxformbuilder.deb && \
    rm -rf /var/lib/apt/lists/*

#  TODO: move this to the base container. Very nice screen shot tool for layout editing with LLM
# xdotool x11-utils wmctrl for a script I have to get push notifications on hooks when claude stalls (TODO add notes for others)
# sox is for Whisper (local voice trasncription), still testing
RUN apt-get update && apt install -y flameshot xdotool x11-utils wmctrl sox && rm -rf /var/lib/apt/lists/*

# # Install Node 16 
RUN echo "build npm" && if [[ "x${build_npm}" == "xtrue" ]] ; then /tmp/install_node_16.sh ; fi

# Download and install LibTorch 2.5.0 CPU-only (cxx11 ABI for compatibility with GCC 11+)
# This is required for blush regularization and other ML-based tools
RUN if [[ "x${build_libtorch}" == "xtrue" ]]; then echo "installing lib torch" && /tmp/install_libtorch.sh ; else echo "NOT installing libtorch" ; fi


# Include the lib path in LD_RUN_PATH so on linking, the correct path is known 
# TODO: could this be avoided as the installation? Will there be any problems if used without the actual install
# Only set LD_RUN_PATH for non-static builds to avoid warnings
RUN if [[ "x${build_type}" != "xstatic" ]]; then echo "export LD_RUN_PATH=/opt/libtorch/lib:\${LD_RUN_PATH}" >> /etc/profile.d/ld_run_path.sh; fi

# TODO: if adding dynamic linking for the cuda libs, we can save a bunch of space in the image by removing these large static libs
# Use the basename bit to ensure no rm -rf foibles with root dir in empyt string case
# RUN ls /usr/local/cuda/lib64/lib*_static.a | grep -v cufft_static.a | while read a; do rm -rf /usr/local/cuda/lib64/$(basename $a); done && \
#     rm -rf /usr/local/cuda/lib64/libcufft_static_nocallback.a

RUN echo 'source /opt/venv/bin/activate' >> /home/cisTEMdev/.bashrc

RUN if [[ "x${build_docs}" == "xtrue" ]] ; then /tmp/install_documentation_tooling.sh  && /tmp/install_llm_optimization_tools.sh ; fi

USER cisTEMdev
WORKDIR /home/cisTEMdev

RUN echo "set filename-display basename" > /home/cisTEMdev/.gdbinit
RUN echo "build claude" && if [[ "x${build_claude}" == "xtrue" ]] ; then /tmp/install_node_22_and_claude.sh ; fi

# Make sure we open claude in verbose mode (can be turned off) so the startup hook ensureing skills are loaded is clear
RUN echo "alias claude='claude --verbose'" >> /home/cisTEMdev/.bashrc