CMake flags tested with

cmake_minimum_required(VERSION 2.8.9)
project (hello)
add_executable(hello helloworld.cpp)

#include <iostream>

int main(int argc, char *argv[]){
    std::cout << "Hello World!" << std::endl;
    return 0;
}

