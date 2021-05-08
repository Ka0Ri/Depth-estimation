import torch.nn.functional as F
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn, optim, rand, sum as tsum, reshape, save
from torch.utils.data import DataLoader, Dataset

SAMPLE_DIM = 21000

class CustomDataset(Dataset):
    def __init__(self, samples=42):
        self.dataset = rand(samples, SAMPLE_DIM).cpu().float() * 2 - 1

    def __getitem__(self, index):
        return (self.dataset[index], (tsum(self.dataset[index]) > 0).cpu().float())

    def __len__(self):
        return self.dataset.size()[0]

class OurModel(LightningModule):
    def __init__(self):
        super(OurModel, self).__init__()
        # Network layers
        self.linear = nn.Linear(SAMPLE_DIM, 2048)
        self.linear2 = nn.Linear(2048, 1)
        self.output = nn.Sigmoid()
        # Hyper-parameters, that we will auto-tune using lightning!
        self.lr = 0.000001
        self.batch_size = 512

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        output = self.output(x)
        return reshape(output, (-1,))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        loader = DataLoader(CustomDataset(samples=43210), batch_size=self.batch_size, shuffle=True)
        return loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.binary_cross_entropy(self(x), y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def val_dataloader(self):
        loader = DataLoader(CustomDataset(samples=1234), batch_size=self.batch_size, shuffle=False)
        return loader

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = F.binary_cross_entropy(self(x), y)
        return {'val_loss': loss, 'log': {'val_loss': loss}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_acc in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        print("OUR LR:",self.lr)
        return results

if __name__ == '__main__':
    seed_everything(42)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='auto')
    model = OurModel()
    trainer = Trainer(gpus=1, max_epochs=100, min_epochs=1, auto_lr_find=False, auto_scale_batch_size=False,
                      progress_bar_refresh_rate=10, callbacks=[early_stop_callback])
    trainer.tune(model)

    trainer.fit(model)
    save(model.state_dict(), 'Location of our saved model')