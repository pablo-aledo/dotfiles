sudo yum install -y perl-Data-Dumper.x86_64 autoconf libtool gcc-c++ flex bison
cd
git clone https://bitbucket.org/icldistcomp/ulfm2.git
cd ulfm2
./autogen.pl
./configure --with-ft
#    use --with-ft (default) to enable ULFM,
#        --without-ft to disable it
make all
sudo make install
echo "alias umpirun='mpirun --oversubscribe -mca btl tcp,self'" | tee -a ~/.shell
