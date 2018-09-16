FROM debian:jessie

# need 32 bit libraries to run binaries in MAUS distribution
RUN dpkg --add-architecture i386
RUN apt-get update && apt-get -y --force-yes install libstdc++6:i386 libgcc1:i386 wget tcsh sox

RUN mkdir /home/maus && cd /home/maus && \
    wget -q ftp://ftp.phonetik.uni-muenchen.de/pub/BAS/SOFTW/MAUS/maus-1608051212.tgz &&\
    tar -xzf maus-1608051212.tgz &&\
    rm maus-1608051212.tgz

# need to fix up some paths in maus files
RUN sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/maus && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/maus.corpus && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/maus.iter && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/maus.trn && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/maus.web && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/mausbpf2emuR && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/mausbpfDB2emuRDB && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/par2TextGrid && \
    sed -i  -e 's+/homes/schiel/MAUS/TOOL+/home/maus/+' /home/maus/par2emu

ENTRYPOINT []
