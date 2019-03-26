FROM nvidia/cuda:8.0-cudnn6-devel-centos7

# Conda / initial dependencies
RUN yum -y install cmake python-pip python-devel autoconf automake bzip2 freetype-devel git libtool pkgconfig zlib-devel yasm-devel libtheora-devel libvorbis-devel libX11-devel gtk2-devel protobuf-devel leveldb-devel snappy-devel opencv-devel lmdb-devel python-devel gflags-devel glog-devel kernel-devel wget unzip gmp-devel mpfr-devel libmpc-devel numactl-devel.x86_64 && \
    wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
    bash Miniconda2-latest-Linux-x86_64.sh -b -p $HOME/miniconda
ENV PATH $PATH:/root/miniconda/bin
RUN cd /root/miniconda/bin && \
    conda update conda && \
    yum -y groupinstall 'development tools'
RUN conda create -n r21d python=2.7 && \
    source activate r21d && \
    conda install setuptools lmdb numpy flask future graphviz hypothesis jupyter matplotlib protobuf pydot pyyaml requests scikit-image scipy six tornado typing && \
    conda install -c conda-forge python-nvd3  && \
    conda install -c mingfeima mkldnn && \
    conda install -c pytorch magma-cuda80 

#GCC
RUN curl https://ftp.gnu.org/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.bz2 -O && \
    tar xvfj gcc-5.4.0.tar.bz2 && \
    mkdir gcc-5.4.0-build && cd gcc-5.4.0-build && \
    ../gcc-5.4.0/configure --enable-languages=c,c++ --disable-multilib && \
    make -j16 && make install 

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib64:$LD_LIBRARY_PATH
RUN wget https://cmake.org/files/v3.11/cmake-3.11.4.tar.gz && \
    tar xvzf cmake-3.11.4.tar.gz && \
    cd cmake-3.11.4 && \
    ./bootstrap && \
    make && \
    make install 

ENV PATH $PATH:/usr/bin/wget:/usr/bin/unzip:/usr/bin/pip

# FFMPEG
RUN wget http://www.nasm.us/pub/nasm/releasebuilds/2.13.01/nasm-2.13.01.tar.xz && \
    tar -xvf nasm-2.13.01.tar.xz && \
    cd nasm-2.13.01 && \
    ./configure && \
    make && make install && \
    cd /usr/local && \
    git clone http://git.videolan.org/git/x264.git && \
    cd x264 && \
    ./configure --enable-shared --enable-pic && \
    make -j16 && make install && \
    cd /usr/local && \
    curl -O -L http://downloads.xiph.org/releases/ogg/libogg-1.3.3.tar.gz && \
    tar xzvf libogg-1.3.3.tar.gz && \
    cd libogg-1.3.3 && \
    ./configure && \
    make -j16 && make install && \ 
    cd /usr/local && \
    git clone http://git.videolan.org/git/ffmpeg.git && \
    cd ffmpeg && \
    git checkout a3a6d4da62539c38e89a5e427c21bfe94dfd3564 && \
    ./configure --enable-gpl --enable-nonfree --enable-libtheora --enable-libvorbis  --enable-libx264  --enable-postproc --enable-version3 --enable-pthreads --enable-shared --enable-pic --cc="gcc -m64 -fPIC" && \
    make -j16 && make install && \
    hash -r && \
    ldconfig
ENV PATH=/usr/local/cuda/bin:$PATH LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH 
RUN cd /usr/local/ffmpeg/ && find . -type f -iname \*.a -delete

# OPENCV
RUN cd /usr/local && \
    wget https://github.com/opencv/opencv/archive/3.4.4.zip -O opencv-3.4.4.zip && \
    unzip opencv-3.4.4.zip && \
    cd opencv-3.4.4 && mkdir build && cd build && \
    cmake   -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/root/miniconda/envs/r21d \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_PYTHON_SUPPORT=ON \
    -D BUILD_OPENCV_PYTHON2=ON \
    -D PYTHON2_EXECUTABLE=/root/miniconda/envs/r21d/bin/python2.7 \
    -D PYTHON2_INCLUDE_DIR=/root/miniconda/envs/r21d/include/python2.7 \
    -D PYTHON2_PACKAGES_PATH=/root/miniconda/envs/r21d/lib/python2.7/site-packages \
    -D WITH_CUDA=ON \
    -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs .. && \
    make -j16 && make install && \
    ldconfig 

# NCCL
RUN cd /usr/local && \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl && \
    make -j src.build && \
    yum install -y rpm-build rpmdevtools && \
    make pkg.redhat.build

# CAFFE2
RUN source activate r21d && \
    ln -s /root/miniconda/envs/r21d/python/cv2/python-2.7/cv2.so /root/miniconda/envs/r21d/lib/python2.7/site-packages/cv2.so && \
    conda install numpy pyyaml mkl mkl-include setuptools cython cffi typing pandas python-lmdb && \
    cd /usr/local && \
    git clone --recursive https://github.com/pytorch/pytorch.git && \
    cd pytorch && git checkout 'c2a75926cac36b770c9afcca894d07feb307592e' && mkdir build && cd build && \
    cmake -DPYTHON_EXECUTABLE=/root/miniconda/envs/r21d/bin/python \
    -DPYTHON_LIBRARY=/root/miniconda/envs/r21d/lib/libpython2.7.so.1.0 \
    -DPYTHON_INCLUDE_DIR=/root/miniconda/envs/r21d/include/python2.7 \
    -DBUILDING_WITH_TORCH_LIBS=ON \
    -DTORCH_BUILD_VERSION=1.0.0a0+db5d313 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TORCH=ON \
    -DBUILD_PYTHON=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_BINARY=ON \
    -DBUILD_TEST=ON \
    -DINSTALL_TEST=ON \
    -DBUILD_CAFFE2_OPS=ON \
    -DONNX_NAMESPACE=onnx_torch \
    -DUSE_CUDA=1 \
    -DUSE_DISTRIBUTED=OFF \
    -DUSE_FBGEMM=0  \
    -DUSE_NUMPY=  \
    -DNUMPY_INCLUDE_DIR=/root/miniconda/envs/r21d/lib/python2.7/site-packages/numpy/core/include \
    -DUSE_SYSTEM_NCCL=OFF  \
    -DNCCL_INCLUDE_DIR=  \
    -DNCCL_ROOT_DIR=  \
    -DNCCL_SYSTEM_LIB=  \
    -DCAFFE2_STATIC_LINK_CUDA=0  \
    -DUSE_ROCM=0 \
    -DUSE_NNPACK=0 \
    -DUSE_NCCL=OFF \
    -DUSE_LEVELDB=OFF \
    -DUSE_LMDB=ON \
    -DUSE_OPENCV=ON \
    -DUSE_QNNPACK=0 \
    -DUSE_FFMPEG=ON \
    -DUSE_GLOG=OFF \
    -DUSE_GFLAGS=OFF \
    -DUSE_SYSTEM_EIGEN_INSTALL=OFF \
    -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
    -DCUDNN_LIB_DIR=/usr/local/cuda/lib64 \
    -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so.6 \
    -DUSE_MKLDNN=0 \
    -DNCCL_EXTERNAL=0 \
    -DCMAKE_INSTALL_PREFIX=/root/miniconda/envs/r21d \
    -DCMAKE_C_FLAGS= \
    -DCMAKE_CXX_FLAGS= '-DCMAKE_EXE_LINKER_FLAGS= \
    -Wl,-rpath,$ORIGIN ' '-DCMAKE_SHARED_LINKER_FLAGS= \
    -Wl,-rpath,$ORIGIN ' \
    -DTHD_SO_VERSION=1 '-DCMAKE_PREFIX_PATH=/root/miniconda/envs/r21d/' .. && make \
    -j16 install 

# Download the app
ADD . /home/code

# Download the weights
RUN cd /home && wget https://dl.dropbox.com/s/y6c6jf7myv2w6iq/r2.5d_d34_l32.pkl?dl=1

# Additional libs for opencv demo
RUN yum install -y libcanberra-gtk2

# Add to the path
ENV PYTHONPATH=$PYTHONPATH:/home/code/src/lib

