mksrcdir /usr/src/fsl-community-bsp
cd /usr/src/fsl-community-bsp
repo init -u https://github.com/Freescale/fsl-community-bsp-platform -b jethro
repo sync

source setup-environment build

bitbake core-image-minimal
