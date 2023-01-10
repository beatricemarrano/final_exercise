from torch import nn, optim
from pytorch_lightning import LightningModule

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.classifier(self.backbone(x))
    
    def training_step(self, batch, bathc_idx):
        data, target= batch
        preds= self(data)
        loss= self.criterium(preds, target)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr= 1e-2)