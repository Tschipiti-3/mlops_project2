FROM python:3.11

ARG WANDB_KEY

ADD main.py .
ADD GLUE.py .
ADD checkpoints .
ADD requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN wandb login $WANDB_KEY

CMD ["python", "./main.py", "--checkpoint_dir", "checkpoints", "--lr", "1e3", "--adam_epsilon", "0.00001", "--weight_decay", "0.00001"]
