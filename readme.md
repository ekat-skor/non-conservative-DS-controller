[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## Build Status
[![Test Template](https://github.com/acfr/ros2_template_pkg/actions/workflows/ci_actions.yml/badge.svg)](https://github.com/acfr/ros2_template_pkg/actions/workflows/ci_actions.yml)

## About


## How to build
### Prerequisites
- Docker
- Preferably an X86_X64 computer with Linux 
- VSCode 

### Installation and Setup

To get started with development, install docker desktop on your computer by following instructions on the docker website.
 If you are using Linux as your base operating system ( which is wonderful :) ) just install docker engine and don't bother installting the docker desktop. https://docs.docker.com/engine/install/ and make sure you do the post-installation steps https://docs.docker.com/engine/install/linux-postinstall/ .

If you are a Windows or Mac user :(, Then you can follow the instructions on this link to install docker sektop on your machine. https://www.docker.com/get-started/ 

Next up, install vscode from this link if you don't already have it installed. https://code.visualstudio.com/

After docker and vscode installation, you will need to have the devcontainers extension installed in vscode to make building and running docker containers easy. you can go to the extensions panel in vscode, search and install the devcontainers extension. 
https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers

why do all this ? 
    we will be running all the code on a docker container and this avoids tedious installation and configuration of ROS2 and all other dependancies that are needed to run the code. Hopefully, with minimal effort the container should build without any issues and run smoothly. The first build will take some time as docker pulls the base image and installs all the required packages and libraries. This will get stored on your local machine as a docker image that will have everything we need. Once it is built the next time you open vscode it should run almost instantly. you can read more about docker and devcontainers here - https://code.visualstudio.com/docs/devcontainers/containers



make sure your git ssh keys are setup. if not follow this link on how to do it - https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

### Clone the repo
``` bash
git clone git@github.com:ekat-skor/non-conservative-DS-controller.git
cd non-conservative-DS-controller

# make sure to checkout to the right branch 
git checkout <branch_name> 
```
Make sure the docker daemon is running 
``` bash
systemctl status docker.service 
```
This should show that the docker service is loaded, active and running. \

Open VSCode from the git folder and the devcontainer plugin should detect the configuration and give you a pop up to 
open in container. 

Before you run the code in the docker container, you should edit few things in the devcontainer.json file. on the file menu in vscode you should see a .devcontainer folder that has a devcontainer.json and a Dockerfile.  \
In the devcontainer.json file replace the remoteuser and USERNAME to your computer's user name . 

Press Ctrl+Shift+P and look for the command rebuild and launch container and run it. 
This will trigger the docker build and deploy process and it should end up with a terminal access to the docker container where you can build and run the code.

### Compile the Code

Once inside the container, you can go into the ros workspace and compile the code. The source file is already added into the system wide bash so you dont have to source the ros setup file.
``` bash
cd ros_ws
catkin_make

source devel/setup.bash
```

## Usage

Once the code is built, you can source the environment and run the launch files to start the simulation or run the franka panda robot

## Running the sim
``` bash
roslaunch franka_interactive_controllers simulate_panda_gazebo.launch
```

## Running on franka panda
``` bash
roslaunch franka_interactive_controllers franka_interactive_bringup.launch
```