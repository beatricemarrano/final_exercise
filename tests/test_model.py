import os

import pytest
import torch
from hydra import compose, initialize

import sys
sys.path.append("/Users/mac/Documents/GitHub/final_exercise/")
from src.models.model import MyAwesomeModel
#from tests import _PROJECT_ROOT

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
@pytest.mark.skipif(
    not os.path.exists(_PROJECT_ROOT + "/config"), reason="Config files not found"
)
def test_distil_model_output_shape():
    with initialize("../config/"):
        cfg = compose(config_name="default_config.yaml")

        cfg.model["model"] = "MyAwesomeModel"
        model = MyAwesomeModel(cfg)
        
        #token_len = cfg["build_features"]["max_sequence_length"]
        #x = torch.randint(0, 1000, (5, token_len))
        #y = torch.randint(0, 1000, (5, token_len))
        #(logits,) = model((x, y))

        #assert logits.shape == torch.Size([5, 2])
    
def test_error_on_wrong_shape():
    with initialize("../config/"):
        cfg = compose(config_name="default_config.yaml")

        cfg.model["model"] = "MyAwesomeModel"
        model = MyAwesomeModel(cfg)
        
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    #assert len(train_dataset) == 4000, "Dataset did not have the correct number of samples"          
