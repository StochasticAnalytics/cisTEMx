FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# /groups/himesb/.config/Code\ -\ Insiders/User/globalStorage/ms-vscode-remote.remote-containers/cli-bin/code-insiders build --image-name bhimesbhimes/cistem_build_env:latest ./.devcontainer_build

# For compiling wxWidgets
ARG n_threads=16
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=America/New_York
ARG GCC_VER=11

# By default this will create cisTEMdev as uid==1000 which is not what we want as it is also going to be the user id
# of a host user when building the top layer with singularity. < 1000 is reserved for system users. 814 is a somewhat random choice.
RUN apt-get update &&\
    apt-get -y install tzdata sudo &&\
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone &&\
    apt-get install -y locales &&\
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 &&\
    useradd --uid 814 -ms /bin/bash cisTEMdev &&\
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

# TEMPORARY: Truncated for quick build test
CMD ["/bin/bash"]
