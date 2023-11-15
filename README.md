# mlops_project2
1. Install docker desktop: https://www.docker.com/products/docker-desktop/ <br>
2. Clone this repository
3. Run the following command to build the image:<br>
`docker build --build-arg WANDB_KEY=[WANDB-KEY] -t [IMAGE_NAME] .` <br>
<b>WANDB_KEY</b>: insert your Weights & Biases API-Key. <br> You can find it in the User settings: https://wandb.ai/settings <br>
<b>IMAGE_NAME</b>: Define your Docker Image Name (e.g. mlops-project2)
4. To run the Docker and start one training run use the following command:<br>
`docker run [IMAGE_NAME]`

