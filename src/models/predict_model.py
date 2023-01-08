import glob
import logging
import os
from pathlib import Path
from time import time

import hydra
import numpy as np
import omegaconf
import torch.quantization
from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig
from torch import nn
import sys

sys.path.append("/Users/mac/Documents/GitHub/final_exercise/src")
from data.data import CorruptMnist
from models.model import MyAwesomeModel

@hydra.main(config_path="../../config", config_name="default_config.yaml")
def main(config: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Executing predict model script.")

    # %% Validate output folder
    #PRIMA PARTE: trova il folder dove c è il tuo modello
    output_dir = os.path.join(
        hydra.utils.get_original_cwd(), config.predict.model_output_dir
    )
    if not os.path.isdir(output_dir):
        raise Exception(
            'The "model_output_dir" path ({}) could not be found'.format(output_dir)
        )
    #output_dir = "/Users/mac/Documents/GitHub/final_exercise/models"

    # %% Load local config in output directory
    # INDICA DOVE MANDARE LE PREDIZIONI
    output_config_path= " /Users/mac/Documents/GitHub/final_exercise/src/models"

    # %% Load model
    # CARICA IL MODELLO
    model= torch.load(os.path.join(output_dir, "trained_model.pt"))
    
    # %% Load data module and use Validation data
    #CARICA I DATI DA MANDARE IN INPUT
    data_module = CorruptMnist(train=False)
    data_test= data_module.data

    # %% Predict and save to output directory
    output_prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(output_prediction_dir, exist_ok=True)

    pred= model(data_test.float())
    pred_np = pred.detach().numpy()    
    output_prediction_file = os.path.join(output_prediction_dir, "predictions.csv")
    np.savetxt(output_prediction_file, pred_np, delimiter=",")



if __name__ == "__main__":
    #log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()