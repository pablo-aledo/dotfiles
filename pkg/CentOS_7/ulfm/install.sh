cd
git clone https://bitbucket.org/icldistcomp/ulfm2.git
cd ulfm2
./autogen.pl
./configure --with-ft
#    use --with-ft (default) to enable ULFM,
#        --without-ft to disable it
make all install
