import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from model import Generator, Discriminator

class CycleGANTransferModule(pl.LightningModule):
    def __init__(self, config, normative_loader, non_normative_loader):
        super(CycleGANTransferModule, self).__init__()
        self.normative_loader = normative_loader
        self.non_normative_loader = non_normative_loader
        self.use_reconst_loss = config.use_reconst_loss
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lr = config.lr or 10e-3

        self.g_to_non_normative = Generator()
        self.g_to_normative = Generator()
        self.d_non_normative = Discriminator()
        self.d_normative = Discriminator()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, direction):
        if direction == 'to_normative':
            return self.g_to_non_normative(x)
        elif direction == 'to_non_normative':
            return self.g_to_normative(x)
        else:
            raise ValueError("Invalid direction specified. Choose 'normative' or 'non_normative'.")

    def configure_optimizers(self):

        g_params = list(self.g_to_non_normative.parameters()) + list(self.g_to_normative.parameters())
        d_params = list(self.d_normative.parameters()) + list(self.d_non_normative.parameters())
        
        g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
    
        return [g_optimizer, d_optimizer]
	
	def training_step(self, batch, batch_idx, optimizer_idx):
		normative = batch['normative']
		non_normative = batch['non_normative']

		# Train discriminators
		if optimizer_idx == 1:
			d_normative_real_out = self.d_normative(normative)
			d_non_normative_real_out = self.d_non_normative(non_normative)
			d_normative_real_loss = torch.mean((d_normative_real_out - 1) ** 2)
			d_non_normative_real_loss = torch.mean((d_non_normative_real_out - 1) ** 2)
			d_real_loss = d_normative_real_loss + d_non_normative_real_loss

			fake_non_normative = self.g_to_non_normative(normative)
			fake_normative = self.g_to_normative(non_normative)
			d_non_normative_fake_out = self.d_non_normative(fake_non_normative)
			d_normative_fake_out = self.d_normative(fake_normative)
			d_non_normative_fake_loss = torch.mean(d_non_normative_fake_out ** 2)
			d_normative_fake_loss = torch.mean(d_normative_fake_out ** 2)
			d_fake_loss = d_normative_fake_loss + d_non_normative_fake_loss

			d_loss = d_real_loss + d_fake_loss
			self.log('d_loss', d_loss)
			return d_loss

		# Train generators
		if optimizer_idx == 0:
			fake_non_normative = self.g_to_non_normative(normative)
			fake_normative = self.g_to_normative(non_normative)
			d_non_normative_fake_out = self.d_non_normative(fake_non_normative)
			d_normative_fake_out = self.d_normative(fake_normative)
			g_loss = torch.mean((d_non_normative_fake_out - 1) ** 2)
			g_loss += torch.mean((d_normative_fake_out - 1) ** 2)

			if self.use_reconst_loss:
				reconst_normative = self.g_to_normative(fake_non_normative)
				reconst_non_normative = self.g_to_non_normative(fake_normative)
				g_loss += F.mse_loss(normative, reconst_normative) + F.mse_loss(non_normative, reconst_non_normative)

			self.log('g_loss', g_loss)
			return g_loss

    def train_dataloader(self):
        return {
            'normative': self.normative_loader,
            'non_normative': self.non_normative_loader
        }

# Assuming the config object and data loaders (svhn_loader and mnist_loader) are defined elsewhere:
config = ...  # Your configuration object
normative_loader = ...  # Your SVHN data loader
non_normative_loader = ...  # Your MNIST data loader

# Instantiate the Lightning module
model = CycleGANTransferModule(config, normative_loader, non_normative_loader)

# Train the model using PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=config.train_iters)
trainer.fit(model)

