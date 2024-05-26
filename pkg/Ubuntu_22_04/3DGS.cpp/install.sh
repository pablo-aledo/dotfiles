cd
git clone https://github.com/shg8/3DGS.cpp/
cd 3DGS.cpp
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release -S ./ -B ./build
cmake --build ./build -j4
