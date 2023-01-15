import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
import sys
#from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback #importing Callbacks class

sys.path.append("/Users/mac/Documents/GitHub/final_exercise/src")
from data.data import CorruptMnistDataModule
#sys.path.append("/Users/mac/Documents/GitHub/final_exercise/src/models")
from models.model import MyAwesomeModel

import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import ModelCheckpoint

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Start Training...")
    #client = secretmanager.SecretManagerServiceClient()
    #PROJECT_ID = "dtu-mlops-project"

    #secret_id = "WANDB"
    #resource_name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    #response = client.access_secret_version(name=resource_name)
    #api_key = response.payload.data.decode("UTF-8")
    #os.environ["WANDB_API_KEY"] = api_key
    #WANDB
    wandb.init(project="final_exercise", entity="beatrice-marrano", config= config)
    
    data_module = CorruptMnistDataModule(
        os.path.join(hydra.utils.get_original_cwd(), config.data.path),###
        batch_size=config.train.batch_size,
    )
    data_module.setup()
    model = MyAwesomeModel(config)

    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="train_loss", mode="min")
    
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        max_epochs=config.train.epochs,
        logger=pl.loggers.WandbLogger(project="final_exercise", config=config),#Add the wandb logger
        enable_checkpointing=True, #Callbacks ModelCheckpoint
        callbacks=[checkpoint_callback],
        limit_test_batches=0.25, # run through only 25% of the test set each epoch
        check_val_every_n_epoch=1,
        gradient_clip_val=1.0,
    )
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
        #val_dataloaders=data_module.test_dataloader(),
    )

    torch.save(model, os.path.join('/Users/mac/Documents/GitHub/final_exercise/models', "trained_model.pt"))    
    torch.save(model.state_dict(), os.path.join('/Users/mac/Documents/GitHub/final_exercise/models', "checkpoint.pth"))           


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()