import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn, optim


class MyAwesomeModel(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        
        self.config = config
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3), # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3), # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3), # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, 3), # [N, 8, 20]
            nn.LeakyReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, 128),
            nn.Dropout(),
            nn.Linear(128, 10)
        )
        self.criterium= nn.CrossEntropyLoss()
        
    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        return self.classifier(self.backbone(x))
    
    def training_step(self, batch, bathc_idx):
        data, target= batch
        preds= self(data)
        loss= self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr= 1e-2)

    
    def save_jit(self, file: str = "deployable_model.pt") -> None:
        token_len = 64#self.config["build_features"]["max_sequence_length"]##
        tokens_tensor = torch.ones(1, token_len).long()
        mask_tensor = torch.ones(1, token_len).float()
        script_model = torch.jit.trace(self.model, [tokens_tensor, mask_tensor])
        script_model.save(file)