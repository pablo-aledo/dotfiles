mkd ~/vulkansdk
wget 'https://sdk.lunarg.com/sdk/download/1.2.189.0/linux/vulkansdk-linux-x86_64-1.2.189.0.tar.gz?Human=true' -O vulkansdk-linux-x86_64-1.2.189.0.tar.gz
wget 'https://sdk.lunarg.com/sdk/download/1.3.290.0/linux/vulkansdk-linux-x86_64-1.3.290.0.tar.xz?Human=true' -O vulkansdk-linux-x86_64-1.3.290.0.tar.xz
tar -xf vulkansdk-linux-x86_64-1.3.290.0.tar.xz
export VULKAN_SDK=$(pwd)/1.3.290.0/x86_64
