import wandb
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from GLUE import GLUEDataModule, GLUETransformer

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--adam_epsilon', type=float, default=0.00001, help='Adam epsilon')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--training_runs', type=int, default=1, help='number of training runs')

args = parser.parse_args() 

def train():
    # Set the seed
    seed_everything(42)

    # Define your GLUEDataModule, GLUETransformer, and Trainer
    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
    )
    dm.setup("fit")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir, 
        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
    )
    
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=wandb.config.learning_rate,
        adam_epsilon=wandb.config.adam_epsilon,
        weight_decay=wandb.config.weight_decay,
    )

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=WandbLogger(),  # Use WandbLogger to log metrics
        callbacks=[checkpoint_callback],
    )

    # Fit the model
    trainer.fit(model, datamodule=dm)

def main():
    for i in range(args.training_runs):
        config = dict(
            learning_rate=args.lr,
            adam_epsilon=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        wandb.init(project='mlops_project2', config=config)     
        train()
        wandb.finish()

if __name__ == '__main__':
    main()