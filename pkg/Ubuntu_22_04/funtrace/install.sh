./simple-example/build.sh
./simple-example/run.sh
pip install viztracer
rehash
vizviewer out/funtrace-fi-gcc.json
vizviewer out/funtrace-pg.json
vizviewer out/funtrace-fi-clang.json
vizviewer out/funtrace-xray.json
