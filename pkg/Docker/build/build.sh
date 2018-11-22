#!/bin/bash

# ccache
export PATH=/usr/lib/ccache/bin:$PATH
export CCACHE_DIR=$(find /home/ -type d -name .ccache)
export CCACHE_LOGFILE=/workdir/cache.debug
ccache -F 0
ccache -M 0
ccache -s

if [ $# -eq 0 ]
then
	[ -e .git ] && git submodule update --init --recursive

	if [ -e runme.sh ]
	then
	    chmod +x runme.sh
	    ./runme.sh
    elif [ $(find -iname compile.sh 2>/dev/null) ]
    then
        source "$(find -iname compile.sh)"
	elif [ -e CMakeLists.txt ]
	then
	    mkdir build
	    cd build
        export CMAKE_CXX_FLAGS="-Wall -rdynamic -finstrument-functions -fprofile-arcs -ftest-coverage -g"
        export LDFLAGS="-lgcov"
        export CMAKE_EXPORT_COMPILE_COMMANDS=ON
        cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS" -E env LDFLAGS="$LDFLAGS" ..
	    VERBOSE=1 make
	    #make install
	elif [ -e *.pro ]
	then
	    qmake-qt5
	    make
	    #make install
	elif [ -e configure ]
	then
	    ./configure
	    make
	    #make install
	elif [ -e Makefile ]
	then
	    make
	elif [ -e Makefile.am ]
	then
	    autoreconf -i
	    automake
	    autoconf
	    ./configure
	    make
	    #make install
	else
	     echo "Unknown structure"
	fi
else
	$*
fi
