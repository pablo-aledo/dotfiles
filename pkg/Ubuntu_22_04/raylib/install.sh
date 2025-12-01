sudo apt update
sudo apt install -y git build-essential cmake
sudo apt install -y libasound2-dev mesa-common-dev libx11-dev libxrandr-dev libxi-dev libxinerama-dev libxcursor-dev
git clone https://github.com/raysan5/raylib.git
cd raylib
mkdir build
cd build
cmake ..
make
sudo make install

cat << EOF > test.c
#include "raylib.h"
int main() {
    InitWindow(800, 450, "Raylib Test");
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);
        DrawText("Hello, Raylib!", 200, 200, 20, BLACK);
        EndDrawing();
    }
    CloseWindow();
    return 0;
}
EOF

gcc test.c -lraylib -lm -lpthread -ldl -lrt -lX11
