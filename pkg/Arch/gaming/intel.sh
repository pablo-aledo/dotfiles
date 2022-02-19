sudo pacman -S mesa lib32-mesa
sudo pacman -S libva-mesa-driver mesa-vdpau opencl-mesa vulkan-mesa-layers mesa-demos vulkantools vulkan-intel
sudo pacman -S lib32-libva-mesa-driver lib32-mesa-vdpau lib32-opencl-mesa lib32-vulkan-intel lib32vulkan-mesa-layers lib32-mesa-demos

glxinfo | grep "OpenGL version"
