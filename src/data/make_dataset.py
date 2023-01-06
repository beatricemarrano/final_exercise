# -*- coding: utf-8 -*-
#import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
import torch 

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    
    content_train= []
    content_test= []
    #os.path.isdir("../data/raw")
    
    for i in range(5):
        content_train.append(np.load(os.path.join(input_filepath, f"train_{i}.npz") , allow_pickle=True))
        data_train = torch.tensor(np.concatenate([c['images'] for c in content_train])).reshape(-1, 1, 28, 28)
        data_train= data_train/data_train.max()
        targets_train = torch.tensor(np.concatenate([c['labels'] for c in content_train]))
    
    content_test = np.load(os.path.join(input_filepath, "test.npz"), allow_pickle=True)
    data_test = torch.tensor(content_test['images']).reshape(-1, 1, 28, 28)
    data_test= data_test/data_train.max()
    targets_test = torch.tensor(content_test['labels'])
    
    torch.save(data_train, os.path.join(output_filepath, "data_train.pkl"))
    torch.save(data_test, os.path.join(output_filepath, "data_test.pkl"))
    torch.save(targets_train, os.path.join(output_filepath, "targets_train.pkl"))
    torch.save(targets_test, os.path.join(output_filepath, "targets_test.pkl"))
    
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main('/Users/mac/Documents/GitHub/final_exercise/data/raw/corruptmnist', '/Users/mac/Documents/GitHub/final_exercise/data/processed')
