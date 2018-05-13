### About
Pwnbox is a Docker container with tools for binary reverse engineering and exploitation. It's primarily geared towards Capture The Flag competitions. 

### Installation
You can grab the container from Docker Hub: `docker pull superkojiman/pwnbox`
 1. Make sure you have Docker installed. For OS X users, you'll need to create a Docker machine. Pick one depending on your hypervisor:

        # VMware Fusion
        docker-machine create --driver vmwarefusion \
            --vmwarefusion-disk-size 4000 \
            --vmwarefusion-memory-size 1000 \
            --vmwarefusion-no-share ctf

        # VirtualBox
        docker-machine create --driver virtualbox \
            --virtualbox-disk-size 4000 \
            --virtualbox-memory 1000 \
            --virtualbox-no-share ctf

 1. Optional: Create a ./rc directory. Your custom configuration files in $HOME go here. Eg: .gdbinit, .radare2rc, .bashrc, .vimrc, etc. The contents of rc gets copied into /root on the container. 
 1. Get the `run.sh` script from [https://raw.githubusercontent.com/superkojiman/pwnbox/master/run.sh](https://raw.githubusercontent.com/superkojiman/pwnbox/master/run.sh). 
 1. Execute `run.sh` script which creates a container named `ctfname-ctf`. Eg:

        $ ./run.sh defcon
        f383e644c0e2504f30487f1d658d8b61a66fca2bdb961fabb0277b05660f5367
                                 ______
        ___________      ___________  /___________  __
        ___  __ \_ | /| / /_  __ \_  __ \  __ \_  |/_/
        __  /_/ /_ |/ |/ /_  / / /  /_/ / /_/ /_>  <
        _  .___/____/|__/ /_/ /_//_.___/\____//_/|_|
        /_/                           by superkojiman

        #
 1. When you're ready to delete the container, use the `ctfname-ctf-stop.sh` script.

### Limitations
 1. If you need to edit anything in /proc, you must edit `run.sh` to use the `--privileged` option to `docker` instead of `--security-opt seccomp:unconfined`. 
 1. The container is designed to be isolated so no directories are mounted from the host. This allows you to have multiple containers hosting files from different CTFs. 


### Go forth, and CTF 
•_•)

( •_•)>⌐■-■

(⌐■_■)
