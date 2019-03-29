# Docker instructions
To build the Docker image in the current directory, type in BASH: docker build -t <image_name> ./

To run the Docker image, type in BASH: docker run -ti -v ${PWD}:${PWD} -w ${PWD} <image_name>
