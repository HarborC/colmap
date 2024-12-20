#!/bin/bash

BASE_DIR=$(cd $(dirname $0);pwd)
cd ${BASE_DIR}

# rm -rf build/
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install