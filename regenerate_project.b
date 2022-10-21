rm -fr m4
mkdir m4
cd m4
ln -sf scripts/m4/aclocal.m4 aclocal.m4
ln -s ../scripts/m4/ax_cuda.m4 ax_cuda.m4
ln -s ../scripts/m4/submodule_FastFFT.m4 submodule_FastFFT.m4
ln -s ../scripts/m4/additional_programs.m4 additional_programs.m4
cd ..
libtoolize --force || glibtoolize
aclocal
autoheader --force
autoconf
automake --add-missing --copy

