git clone https://github.com/symisc/tiny-dream.git
cd tiny-dream
export PKG_CONFIG_PATH=~/ncnn/build/install/lib/pkgconfig
g++ -o tinydream boilerplate.cpp -funsafe-math-optimizations -Ofast -flto=auto  -funroll-all-loops -pipe -march=native -std=c++17 -Wall -Wextra `pkg-config --cflags --libs ncnn` -L~/vulkansdk/1.3.290.0/x86_64/lib -lglslang -lstdc++ -pthread -flto -fopt-info-vec-optimized -fopenmp
./tinydream "pyramid, desert, palm trees, river, (landscape), (high quality)"
