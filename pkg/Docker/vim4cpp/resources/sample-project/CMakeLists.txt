cmake_minimum_required(VERSION 3.7)
project(sample-project) 

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER "clang++") 
set(CMAKE_EXE_LINKER_FLAGS "-lc++ -lc++abi") 

add_subdirectory(src bin)
