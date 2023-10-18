#!/bin/bash

# Called from the top layer Dockerfile
# Used to make conditionals easier

n_threads=16

# Make sure we got the arg
if [[ $# -ne 1 ]] ; then
    echo "Usage: $0 <compiler>"
    exit 1
fi
# Make sure it is icpc or g++
if [[ $1 != "icpc" && $1 != "g++" ]] ; then
    echo "Invalid compiler: ($1) - must be icpc or g++"
    exit 1
fi
compiler=${1}

if [[ $compiler == 'icpc' ]] ; then
    src_cmd=". /opt/intel/oneapi/setvars.sh"
    dir_name='intel'
else
    src_cmd=""
    dir_name='gnu'
fi

#--prefix=/opt/WX/intel-dynamic
$src_cmd 

# We install both the static and dynamic libs, dynamic second, so that there is a wx-config at /usr/bin, which is needed for wxFormbuilder
# This does create a possible conflict if a user wants static configures cisTEM without --wx-config=/opt/WX/intel-static/bin/wx-config, but that is unlikely (since the system wx-config will be grabbed)
for linkage in "static" "dynamic" ; do
    cd /opt/WX/${dir_name}-${linkage}/wxWidgets-3.0.5 && make install && make clean && ldconfig
    rm -rf /opt/WX/${dir_name}-${linkage}/wxWidgets-3.0.5

    # First noticed outside container with g++9, several errors in longlong.h seem to be fixed by this extra include  /usr/include/wx-3.1-unofficial
    tf=`tempfile` && cp /opt/WX/${dir_name}-${linkage}/include/wx-3.0/wx/longlong.h /opt/WX/${dir_name}-${linkage}/include/wx-3.0/wx/longlong.h.orig && \
        awk '{if(/#include "wx\/defs.h"/){ print $0 ;print "#include <wx/txtstrm.h>"} else print $0}' /opt/WX/${dir_name}-${linkage}/include/wx-3.0/wx/longlong.h.orig > $tf && \
        mv $tf /opt/WX/${dir_name}-${linkage}/include/wx-3.0/wx/longlong.h && \
        chmod a+r /opt/WX/${dir_name}-${linkage}/include/wx-3.0/wx/longlong.h
done



