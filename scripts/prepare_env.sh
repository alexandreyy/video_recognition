#!/usr/bin/env sh
# Update system.
sudo apt-get update
sudo apt-get upgrade

# Install dependencies.
sudo apt-get install build-essential rar libsqlite3-dev sqlite3 bzip2 libbz2-dev zlib1g-dev libssl-dev openssl libgdbm-dev liblzma-dev libreadline-dev libncursesw5-dev libffi-dev uuid-dev cmake git pkg-config

# Install OpenCV.
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran
sudo python3.6 -mpip install numpy

git clone https://github.com/opencv/opencv
cd opencv
git checkout 3.4.1
cd ..
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.1

cd ../opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D CMAKE_CXX_FLAGS="-fpermissive" \
      -D BUILD_EXAMPLES=ON ..
make -j4
sudo make install
sudo ldconfig
cd ../../../
sudo rm -R opencv
sudo rm -R opencv_contrib

# Install python packages.
sudo python3.6 -mpip install numpy opencv-python opencv-contrib-python tensorflow-gpu
