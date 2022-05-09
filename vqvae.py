import matplotlib.pyplot as plt
import numpy as np
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import os
import scipy
from datasets import DatasetEMNIST
from tensorboardX import SummaryWriter

MAX_LEN = 256

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batch_size = 32
epochs = 100
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
learning_rate = 1e-3

writer = SummaryWriter(
    logdir='tmp'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# training_data = datasets.CIFAR10(root="data", train=True, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#                                   ]))
# validation_data = datasets.CIFAR10(root="data", train=False, download=True,
#                                   transform=transforms.Compose([
#                                       transforms.ToTensor(),
#                                       transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
#                                   ]))
training_data = DatasetEMNIST(is_train=True, len=100000)
validation_data = DatasetEMNIST(is_train=False, len=MAX_LEN)

# TODO: why for variance calculation only one CPU used? try numba here
all_training_data = [] 
for i in range(len(training_data)):
    all_training_data.append(training_data[i][0])
result_arr = np.stack(all_training_data, axis=0)
x_variance = np.var(all_training_data)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))          
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        loss_encoder = F.mse_loss(quantized.detach(), inputs) * self._commitment_cost
        loss_codebook = F.mse_loss(quantized, inputs.detach())
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss_encoder, loss_codebook, quantized.permute(0, 3, 1, 2).contiguous(), perplexity

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens, num_residual_layers=num_residual_layers, num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)   
        x = self._residual_stack(x)  
        x = self._conv_trans_1(x)
        x = F.relu(x)   
        return self._conv_trans_2(x)
             
class VQVAE(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self._encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss_encoder, loss_codebook, quantized, perplexity = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss_encoder, loss_codebook, x_recon, perplexity


def main():
    data_loader_train = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    data_loader_validation = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    tb_x_norm, _ = next(iter(data_loader_validation))
    # MNIST tmp
    tb_x_norm = torch.squeeze(torch.stack([tb_x_norm, tb_x_norm, tb_x_norm], dim=1))
    tb_x = ((tb_x_norm.cpu().numpy() + 0.5) * 255.0).astype(np.uint8)


    model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    metrics = {}
    for stage in ['train', 'validation']:
        for metric in ['loss', 'loss_rec', 'loss_encoder', 'loss_codebook', 'perplexity']:
            metrics[f'{stage}/{metric}'] = []

    for epoch in range(1, epochs):
        for data_loader in [data_loader_train, data_loader_validation]:
            metrics_epoch = {key: [] for key in metrics.keys()}
            stage = 'train'
            model = model.train()
            torch.set_grad_enabled(True)
            if data_loader == data_loader_validation:
                stage = 'validation'
                model = model.eval()
                torch.set_grad_enabled(False)
            for x, _ in data_loader:
                # MNIST tmp
                x = torch.squeeze(torch.stack([x,x,x], dim=1))
                x = x.to(device)

                loss_encoder, loss_codebook, x_prim, perplexity = model.forward(x)
                loss_rec = F.mse_loss(x_prim, x) / x_variance
                loss = loss_rec + loss_encoder + loss_codebook
                
                metrics_epoch[f'{stage}/loss'].append(loss.cpu().item())
                metrics_epoch[f'{stage}/loss_rec'].append(loss_rec.cpu().item())
                metrics_epoch[f'{stage}/loss_encoder'].append(loss_encoder.cpu().item())
                metrics_epoch[f'{stage}/loss_codebook'].append(loss_codebook.cpu().item())
                metrics_epoch[f'{stage}/perplexity'].append(perplexity.cpu().item())

                if data_loader == data_loader_train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            metrics_strs = []
            for key in metrics_epoch.keys():
                if stage in key:
                    value = np.mean(metrics_epoch[key])
                    metrics[key].append(value)
                    metrics_strs.append(f'{key}: {round(value, 2)}')
                    writer.add_scalar(key, value, epoch)

            print(f'epoch: {epoch} {" ".join(metrics_strs)}')

        _, _, tb_x_prim_norm, _ = model.forward(tb_x_norm)
        tb_x_prim = (tb_x_prim_norm.cpu().numpy() * 255.0).astype(np.uint8)
        tb_x_prim = np.clip(tb_x_prim, 0, 255)
        for i in range(len(tb_x)):
            tb_x_merge = np.concatenate((tb_x[i][:1], tb_x_prim[i][:1]), axis=2)
            writer.add_image(f'idx_{i}', tb_x_merge, epoch)

if __name__ == "__main__":
    main()

# TODO: add UMAP on each epoch
# TODO: rename codebook and private vairables
"""
proj = umap.UMAP(n_neighbors=3,
                 min_dist=0.1,
                 metric='cosine').fit_transform(model._vq_vae._embedding.weight.data.cpu())
plt.scatter(proj[:,0], proj[:,1], alpha=0.3)                
"""

# TODO: try different image normalizations on input
# TODO: make code run with local GPU