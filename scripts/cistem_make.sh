#!/bin/bash

build_dir=${1}

cpp_standard=${2}

if [ $3 ] ; then
    n_threads=${3}
else
    n_threads=$(($(nproc) / 2))
fi 

if [[ $cpp_standard == 17 ]]; then
    cd ${build_dir}/
    make -j${n_threads}
else
    if [[ $cpp_standard == 20 ]]; then
        # We need to first  build any modules
        cd ${build_dir}/src

        make -j${n_threads} libmodules.a

        # Now do the normal build
        cd ..
        make -j${n_threads} 

        # Get the top level directory relative to the build dir
        top_srcdir=$(awk '/^top_srcdir/ {print $3}' Makefile)

        # Finally we need to fix the dependency files which are somewhat broken for modules as of now
        cd src
        ${top_srcdir}/../scripts/module_deps_find.sh ${n_threads} ${top_srcdir}/../scripts/module_deps_fix.sh
    else
        echo "cpp_standard must be either 17 or 20"
        exit 1
    fi
fi




