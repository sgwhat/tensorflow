# Steps to generate a container with Intel® Optimization for TensorFlow

This guide will help you generate a container with Intel's icx-base branch build.

## Steps:

1. Clone intel-tensorflow icx-base branch:

    ```
    $ git clone https://github.com/Intel-tensorflow/tensorflow.git --branch=icx-base --single-branch
    $ cd tensorflow
    $ git checkout icx-base
    # Run "git log" and check for the right git hash
    ```

2.  Go to the directory that has Intel mkl docker files:

    ```
    $ cd tensorflow/tools/ci_build/linux/mkl/
    ```

3.  Run build-dev-container.sh by passing the following env parameters:

    For ICX-SERVER containers:

    ```
    $ env  ROOT_CONTAINER=tensorflow/tensorflow \
    	ROOT_CONTAINER_TAG=devel \
    	TF_DOCKER_BUILD_DEVEL_BRANCH=icx-base	 \
    	TF_REPO=https://github.com/Intel-tensorflow/tensorflow \
    	BUILD_ICX_SERVER_CONTAINERS=yes \
    	BUILD_TF_V2_CONTAINERS=yes \    	
    	BAZEL_VERSION=3.7.2 \    	
    	ENABLE_SECURE_BUILD=yes \
        ENABLE_HOROVOD=yes \
	BUILD_SSH=yes \
	TF_NIGHTLY_FLAG=--nightly_flag \
	ENABLE_GCC8=yes \
	RELEASE_CONTAINER=yes \
	OPENMPI_VERSION=openmpi-4.0.5 \
	OPENMPI_DOWNLOAD_URL=https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz \
	HOROVOD_VERSION=56183ca42c43aad7afd619f0cc8bc4842336f3ec \
	INSTALL_HOROVOD_FROM_COMMIT=yes \
	BUILD_SSH=yes \
    	./build-dev-container.sh > ./container_build.log
    ```  

4.  Open a second terminal session at the same location and run `tail -f container_build.log` to monitor container build progress
    or wait until the build finishes and then open the log file <container_build.log> ...

    ```
	INFO: Build completed successfully, 17731 total actions.
    ```

    Below output indicates that the container has intel-optimized tensorflow:

    ```
    PASS: MKL enabled test in <intermediate container name>
    ```

5.  Check if the image was built successfully and tag it:

    ICX-SERVER container:

    ```
    $ docker images
	intel-mkl/tensorflow:nightly-icx-server-devel-mkl
    $ docker tag intel-mkl/tensorflow:nightly-icx-server-devel-mkl intel/intel-optimized-tensorflow:2.4.0-devel-mkl
    ```   
