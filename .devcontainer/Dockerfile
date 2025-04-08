FROM devrt/ros-devcontainer-vscode:noetic-desktop


RUN sudo apt-get update \
    && sudo apt-get install -y python3-pip vim build-essential ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control ros-noetic-pcl-ros pcl-tools \
    ros-noetic-ros-controllers python3-tk ros-noetic-boost-sml

RUN pip3 install KDEpy numpy==1.21


RUN sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libpoco-dev \
    libeigen3-dev \
    ros-noetic-libfranka \ 
    ros-noetic-franka-ros \
    && sudo rm -rf /var/lib/apt/lists/*

# # Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV LIBFRANKA_VERSION=0.9.0
#  ENV LIBFRANKA_BUILD_DIR=/home/developer/libfranka

# # Build and install libfranka
# RUN git clone https://github.com/frankaemika/libfranka.git ${LIBFRANKA_BUILD_DIR} && \
#     cd ${LIBFRANKA_BUILD_DIR} && \
#     git checkout tags/${LIBFRANKA_VERSION} -b build-branch && \
#     mkdir build && cd build && \
#     cmake -DCMAKE_BUILD_TYPE=Release .. && \
#     cmake --build . --parallel && \
#     sudo cmake --install . --prefix /usr/local


# # Set Franka_DIR for CMake to find it
# ENV Franka_DIR=/usr/local/lib/cmake/Franka
# ENV CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$Franka_DIR
