FROM ubuntu:18.04

WORKDIR /root

ENV OPENCV_VERSION=4.3.0 \
    PYTHON_VERSION=3.6 \
    PYTHON_VERSION_SHORT=36 

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get remove -y ffmpeg x264 libx264-dev

RUN apt-get install -y \
    build-essential \
    cmake \
    libjack-jackd2-dev \
    libmp3lame-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libsdl1.2-dev \
    libtheora-dev \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libx11-dev \
    libxfixes-dev \
    libxvidcore-dev \
    texi2html \
    zlib1g-dev \
    wget \
    unzip \
    yasm \
    pkg-config \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    python${PYTHON_VERSION} \
    python3-pip \
    libpq-dev

RUN apt-get remove -y x264 ffmpeg libx264-dev && \
    apt-get install -y x264 libx264-dev && \
    apt-get install -y ffmpeg && \
    pip3 install numpy

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    mkdir opencv-${OPENCV_VERSION}/build

WORKDIR /root/opencv-${OPENCV_VERSION}/build

RUN cmake \
  -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python${PYTHON_VERSION} -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python${PYTHON_VERSION}) \
  -DPYTHON_INCLUDE_DIR=$(python${PYTHON_VERSION} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python${PYTHON_VERSION} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  -DPYTHON_DEFAULT_EXECUTABLE=$(which python${PYTHON_VERSION}) \
  -DBUILD_NEW_PYTHON_SUPPORT=ON \
  -DBUILD_opencv_python3=ON \
  -DHAVE_opencv_python3=ON \
  -DBUILD_opencv_gapi=OFF \
  -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python${PYTHON_VERSION}/dist-packages/numpy/core/include \
  .. \
 && make install

RUN mv \
  /root/opencv-${OPENCV_VERSION}/build/lib/python3/cv2.cpython-${PYTHON_VERSION_SHORT}m-x86_64-linux-gnu.so \
  /usr/local/lib/python${PYTHON_VERSION}/dist-packages/cv2.so

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /root

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/opencv-${OPENCV_VERSION}

RUN apt update && apt install -y --no-install-recommends\
    libgtk2.0-dev \
    libgl1-mesa-glx \
     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "discord_bot.py" ]