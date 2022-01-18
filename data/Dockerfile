FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         expat \
         autoconf \
         automake \
         libass-dev \
         libfreetype6-dev \
         libtool \
         libvorbis-dev \
         pkg-config \
         texinfo \
         wget \
         zlib1g-dev \
         nasm \
         yasm \
         libx264-dev \
         libx265-dev \
         libnuma-dev \
         libvpx-dev \
         libfdk-aac-dev \
         libmp3lame-dev \
         libopus-dev \
         libgomp1  && \
         apt-get autoremove -y && \
        apt-get clean -y

RUN mkdir -p /opt/ffmpeg_sources && cd /opt/ffmpeg_sources && \
    git -C aom pull 2> /dev/null || git clone --depth 1 https://aomedia.googlesource.com/aom && \
    mkdir -p aom_build && \
    cd aom_build && \
    PATH="/opt/ffmpeg/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="/opt/ffmpeg_build" -DENABLE_SHARED=off -DENABLE_NASM=on ../aom && \
PATH="/opt/ffmpeg/bin:$PATH" make && \
make install


RUN mkdir -p /opt/ffmpeg_sources && \
    wget -O ffmpeg-snapshot.tar.bz2 https://ffmpeg.org/releases/ffmpeg-snapshot.tar.bz2 && \
    tar xjvf ffmpeg-snapshot.tar.bz2 && \
    cd ffmpeg && \
    PATH="/opt/ffmpeg/bin:$PATH" PKG_CONFIG_PATH="/opt/ffmpeg_build/lib/pkgconfig" ./configure \
      --prefix="/opt/ffmpeg_build" \
      --pkg-config-flags="--static" \
      --extra-cflags="-I/opt/ffmpeg_build/include" \
      --extra-ldflags="-L/opt/ffmpeg_build/lib" \
      --extra-libs="-lpthread -lm" \
      --bindir="/opt/ffmpeg/bin" \
      --enable-gpl \
      --enable-libaom \
      --enable-libass \
      --enable-libfdk-aac \
      --enable-libfreetype \
      --enable-libmp3lame \
      --enable-libopus \
      --enable-libvorbis \
      --enable-libvpx \
      --enable-libx264 \
      --enable-libx265 \
      --enable-nonfree && \
    PATH="/opt/ffmpeg/bin:$PATH" make && \
    make install && \
    hash -r

# FFMPEG DONE.

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh  && \
 chmod +x ~/miniconda.sh && \
 ~/miniconda.sh -b -p /opt/conda && \
 rm ~/miniconda.sh && \
 /opt/conda/bin/conda install -y python=3.7.3 tqdm numpy pyyaml ipython cython typing pandas matplotlib lxml h5py

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:/opt/ffmpeg/bin:/usr/local/nvidia/bin/:/usr/local/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/opt/ffmpeg/lib:/usr/local/nvidia/lib64:/usr/local/lib
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
#RUN conda install -c conda-forge accimage

# install apex
#RUN git clone https://github.com/NVIDIA/apex.git \
# && cd apex \
# && python setup.py install --cuda_ext --cpp_ext
#
## install pycocotools
#RUN git clone https://github.com/cocodataset/cocoapi.git \
# && cd cocoapi/PythonAPI \
# && python setup.py build_ext install

RUN pip install requests ninja yacs matplotlib opencv-python google-api-python-client oauth2client gcsfs google-api-python-client google-auth-oauthlib google-auth-httplib2 youtube-dl requests scikit-video google-cloud-storage beautifulsoup4 ujson scikit-video scikit-image opencv-python-headless google-auth-oauthlib

## install PyTorch Detection
#ARG FORCE_CUDA="1"
#ENV FORCE_CUDA=${FORCE_CUDA}
#RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
# && cd maskrcnn-benchmark \
# && python setup.py build develop
RUN apt-get update
RUN apt-get install -y parallel libsndfile1-dev libprotobuf-dev protobuf-compiler


# NOTE: Grover denoise will not work any more
RUN pip install tensorflow==2.5.0 regex
RUN pip install editdistance==0.5.3 tslearn==0.5.0.5

RUN pip install ftfy demoji
RUN python -c "import demoji; demoji.download_codes()"
RUN pip install tokenizers==0.10.2
RUN pip install youtube-dl==2021.6.6
RUN pip install gcld3
RUN pip install librosa==0.8.1
RUN pip install zstandard==0.15.2
RUN pip install numpy==1.21.1
RUN pip install scipy==1.7.0
RUN pip install unidecode
RUN pip install num2words

# set the working directory
WORKDIR /youtubescraper

# RUN mkdir /tmp/build
# COPY . /tmp/build
# RUN find /tmp/build

# add the code as the final step so that when we modify the code
# we don't bust the cached layers holding the dependencies and
# system packages.
ADD . .
ENV PYTHONPATH /youtubescraper
CMD ["/bin/bash"]


