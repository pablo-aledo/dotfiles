if [ $# -eq 0 ]
then
	if [ -e runme.sh ]
	then
	    chmod +x runme.sh
	    ./runme.sh
	elif [ -e CMakeLists.txt ]
	then
	    mkdir build
	    cd build
	    cmake ..
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
	sh -c "$*"
fi
