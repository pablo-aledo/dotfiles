sudo mhwd -a pci nonfree 0300
sudo pacman -S mesa
sudo pacman -S lib32-mesa
sudo pacman -S libva-mesa-driver mesa-vdpau opencl-mesa vulkan-mesa-layers mesa-demos vulkantools
sudo pacman -S lib32-libva-mesa-driver lib32-mesa-vdpau lib32-opencl-mesa lib32-vulkan-mesalayers lib32-mesa-demos

glxinfo | grep "OpenGL version"
