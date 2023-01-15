import os

import pytest
from hydra import compose, initialize
import sys

sys.path.append("/Users/mac/Documents/GitHub/final_exercise/")
from src.data.data import CorruptMnistDataModule
#sys.path.append("/Users/mac/Documents/GitHub/final_exercise/tests")
#from tests import _PATH_DATA, _PROJECT_ROOT

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

@pytest.mark.skipif(
    not os.path.exists(_PATH_DATA + "/processed")
    or not os.path.exists(_PROJECT_ROOT + "/config"),
    reason="Data and config files not found",
)
def test_loaders_len_split():
    with initialize("../config/"):
        cfg = compose(config_name="default_config.yaml")
        data_module = CorruptMnistDataModule(
            _PATH_DATA,
            batch_size=cfg.train.batch_size,
        )
        data_module.setup()
        train_loader = data_module.train_dataloader()
        test_loader = data_module.test_dataloader()

        train_set_len = len(train_loader.dataset)
        test_set_len = len(test_loader.dataset)
        assert train_set_len + test_set_len == 30000
        assert train_set_len == 25000
        assert test_set_len == 5000
  