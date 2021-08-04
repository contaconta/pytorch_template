IMAGE_NAME=dl_gpu
SERVER_URL=my-server-url
SERVER_ROOT_DIR=/path/to/server_root
DOCKER_WORKDIR=/mnt/workspace

deploy:
	rsync -avz --progress \
	./ \
	${SERVER_URL}:${SERVER_ROOT_DIR}/ \
	--exclude .git \
	--exclude .venv \
	--exclude outputs

build_docker:
	docker build -t ${IMAGE_NAME} -f dockerfiles/Dockerfile.gpu .

run_docker:
	docker run -d --ipc=host --gpus=all \
	-w ${DOCKER_WORKDIR} \
	-v ${SERVER_ROOT_DIR}:${DOCKER_WORKDIR} \
	-p 6006:6006 \
	-p 8000:8000 \
	-p 8888:8888 \
	-t ${IMAGE_NAME}:latest \
	bash

install_poetry:
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -	

install_pytorch:
	poetry run pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html