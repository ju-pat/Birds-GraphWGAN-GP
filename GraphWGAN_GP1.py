import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn import GraphConv, BatchNorm # GCNConv, GATConv
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torchvision

import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import hsluv
import matplotlib.colors as mcolors

from to_graph_utils import edge_indexes
import importlib
import tools
importlib.reload(tools)         # pour recharger les modifications
from tools import *

colony_abr= [-17.97, -38.70]

## For plotting:

index = np.arange(50)
cmap = plt.colormaps['RdYlGn_r'] # Green to red
# cmap = plt.cm.get_cmap('viridis')
cmap.set_under('black')
colors = cmap(index / max(index))
hex_colors = [mcolors.to_hex(c) for c in colors]
hsluv_colors = [hsluv.hex_to_hsluv(c) for c in hex_colors]
colors = [hsluv.hsluv_to_hex((h, s, min(l, 70))) for h, s, l in hsluv_colors]



def plot_tensor(tensor, x_min, x_max, y_min, y_max):
    tensor = tensor.detach().cpu().numpy()
    x = tensor[:, 0]
    y = tensor[:, 1]

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.scatter(x, y, c=colors, marker='o')
    plt.scatter(x[0], y[0], color='blue')
    plt.scatter(x[-1], y[-1], color='BlueViolet')
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Return the buffer as an image
    return Image.open(buf)

def plot_dist(dist_true, dist_false):
    x = np.arange(dist_false.shape[0])
    mean = np.mean(dist_true, axis=0)
    min_val = np.min(dist_true, axis=0)
    max_val = np.max(dist_true, axis=0)
    plt.plot(mean, label='Mean True', color='blue')
    plt.fill_between(x, min_val, max_val, color='blue', alpha=0.2, label='Min to Max True')

    dist_false = dist_false.detach().cpu().numpy()

    plt.plot(dist_false, label='False')
    plt.legend(loc = "best")
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Return the buffer as an image
    return Image.open(buf)


# class SpectralRegularization(nn.Module):
#     def __init__(self, lambda_reg, spectral_type):
#         super(SpectralRegularization, self).__init__()
#         self.lambda_reg = lambda_reg
#         self.spectral_type = spectral_type

#     def forward(self, x, x_real):
#         if self.spectral_type == 'freq':
#             # Compute frequency-domain representation using FFT
#             fft_x = torch.fft.rfft(x)
#             fft_x_real = torch.fft.rfft(x_real)
#             # Take the magnitude of the complex numbers
#             fft_x = torch.abs(fft_x)
#             fft_x_real = torch.abs(fft_x_real)
#             S = F.mse_loss(fft_x, fft_x_real)
#         elif self.spectral_type == 'eig':
#             # Compute covariance matrix and eigenvalues
#             cov_x = torch.cov(x, rowvar=False)
#             cov_x_real = torch.cov(x_real, rowvar=False)
#             eig_x = torch.eig(cov_x, eigenvectors=True)[0]
#             eig_x_real = torch.eig(cov_x_real, eigenvectors=True)[0]
#             S = F.mse_loss(eig_x, eig_x_real)
#         return S


# Define the Generator and Discriminator models
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        super(Generator, self).__init__()
        # Pour enregistrer les params dans le fichier
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)

        self.conv1 = GraphConv(input_dim, hidden_dim*4)
        # self.bn1 = nn.BatchNorm1d(hidden_dim*8)
        # self.bn1 = nn.LayerNorm(hidden_dim*8)
        self.bn1 = pyg_nn.norm.BatchNorm(hidden_dim*4)
        
        self.conv2 = GraphConv(hidden_dim*4, hidden_dim*8)
        # self.bn2 = nn.LayerNorm(hidden_dim*4)
        self.bn2 = pyg_nn.norm.BatchNorm(hidden_dim*8)

        self.conv3 = GraphConv(hidden_dim*8, hidden_dim*2)
        # self.bn3 = nn.LayerNorm(hidden_dim*2)
        self.bn3 = pyg_nn.norm.BatchNorm(hidden_dim*2)

        self.conv4 = GraphConv(hidden_dim*2, hidden_dim)
        # self.bn4 = nn.LayerNorm(hidden_dim)
        self.bn4 = pyg_nn.norm.BatchNorm(hidden_dim)

        self.conv5 = GraphConv(hidden_dim, output_dim)
        # self.bn5 = nn.LayerNorm(output_dim)
        self.bn5 = pyg_nn.norm.BatchNorm(output_dim)


    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = self.dropout(x)
        x = torch.relu(x)

        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        # x = self.dropout(x)

        # x = torch.hstack(( x, calculate_step_length( x.reshape(int(x.size(0)/50), 50, x.size(1)) ).reshape(x.size(0), 1) ))
        # x = torch.hstack(( x, calculate_dist_nest0( x.reshape(int(x.size(0)/50), 50, x.size(1)) ).reshape(x.size(0), 1) ))

        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(Discriminator, self).__init__()
        # Pour enregistrer les params dans le fichier
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        # self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = GraphConv(hidden_dim, hidden_dim*2)
        self.norm2 = nn.LayerNorm(hidden_dim*2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = GraphConv(hidden_dim*2, hidden_dim*4)
        self.norm3 = nn.LayerNorm(hidden_dim*4)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.conv4 = GraphConv(hidden_dim*4, hidden_dim*8)
        self.norm4 = nn.LayerNorm(hidden_dim*8)
        self.dropout4 = nn.Dropout(dropout_rate)

        self.conv5 = GraphConv(hidden_dim*8, hidden_dim*8)
        self.norm5 = nn.LayerNorm(hidden_dim*8)
        self.dropout5 = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim*8, 1)
        


    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv4(x, edge_index)
        x = self.norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv5(x, edge_index)
        x = self.norm5(x)
        x = self.dropout5(x)
        x = F.leaky_relu(x, 0.2)

        x = torch.sum(x, dim=0)

        x = self.fc(x)

        return x


# Define the Graph-GAN class
class GraphGAN:
    def __init__(self, generator, discriminator, data_loader, device, lr_d, lr_g, input_dim, lambda_gp, 
                critic_it, gene_it, logs_dir, lambda_reg, spectral_type, lambda_rep=0.1, sigma=1.0):
        self.generator = generator
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.device = device
        self.input_dim = input_dim
        self.lr_g = lr_g
        self.lr_d = lr_d
    
        self.lambda_gp = lambda_gp
        self.critic_it = critic_it
        self.gene_it = gene_it
        self.logs_dir = logs_dir

        self.generator.to(device)
        self.discriminator.to(device)

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

        self.lambda_reg = lambda_reg
        # self.spectral_type = spectral_type
        # self.spectral_reg = SpectralRegularization(lambda_reg, spectral_type)

        self.lambda_rep = lambda_rep
        self.sigma = sigma

    
    def named_parameters(self):
        for name, param in self.generator.named_parameters():
            yield f'generator.{name}', param
        for name, param in self.discriminator.named_parameters():
            yield f'discriminator.{name}', param


    def compute_wgan_gp_loss(self, real_data, fake_data, lambda_rep, sigma, edge_index):
        batch_size = real_data.num_graphs
        alpha = torch.rand(real_data.x.size(0), real_data.num_features).to(self.device)

        interpolated_data = alpha * real_data.x + (1 - alpha) * fake_data
        interpolated_data.requires_grad_()

        d_interpolated = self.discriminator(interpolated_data, edge_index)
        grad_outputs = torch.ones_like(d_interpolated)
        gradients = torch.autograd.grad(outputs=d_interpolated, 
                                        inputs=interpolated_data,
                                        grad_outputs=grad_outputs, 
                                        create_graph=True, 
                                        retain_graph=True)[0]
        grad_flat   = gradients.view(gradients.size(0), -1)
    
        grad_norm   = grad_flat.norm(2, dim=1)
        grad_penalty = torch.mean((grad_norm - 1) ** 2) 

        # # Add Repulsion Term
        # repulsion_term = 0
        # for i in range(batch_size):
        #     for j in range(i+1, batch_size):
        #         repulsion_term += torch.exp(-torch.sum((fake_data[i] - fake_data[j]) ** 2) / (2 * sigma ** 2))
        # repulsion_term = -repulsion_term / (batch_size * (batch_size - 1) / 2)
        # loss = grad_penalty + lambda_rep * repulsion_term
        # print(f"Grad Penalty: {grad_penalty.item()}, Repulsion Term: {repulsion_term.item()}")

        loss = grad_penalty
        return loss

    def train(self, num_epochs):
        writer = SummaryWriter(log_dir=self.logs_dir)
        writer_real = SummaryWriter(f""+ self.logs_dir +"/real")
        writer_fake = SummaryWriter(f""+ self.logs_dir +"/fake")
        writer_fake_dist = SummaryWriter(f""+ self.logs_dir +"/fake_dist")
        writer_fake_steps = SummaryWriter(f""+ self.logs_dir +"/fake_steps")
        step = 0

        for epoch in tqdm(range(num_epochs), desc="Training", mininterval=10):
            for data in self.data_loader:
                real_data = data.to(self.device)
                batch_size = real_data.num_graphs
                num_features = real_data.num_features
                num_nodes = real_data.num_nodes
                graph_nodes = int(num_nodes/batch_size)
                
                edge_index = edge_indexes(batch_size, graph_nodes, 2)  # Think to modify also the value in the part of generating trajectories in Main.py
    
                # # # ATENTION : d_input_dim à modifier dans le notebook "Main.ipynb" aussi si on ajoute des features calculées
                # real_data.x = torch.hstack(( real_data.x, calculate_step_length( real_data.x.reshape(batch_size, graph_nodes, num_features) ).reshape(num_nodes, 1) ))
                # real_data.x = torch.hstack(( real_data.x, calculate_dist_nest0( real_data.x.reshape(batch_size, graph_nodes, real_data.x.size(1)) ).reshape(num_nodes, 1) ))


                # Train the discriminator

                for _ in range(self.critic_it):
                    self.discriminator_optimizer.zero_grad()

                    noise = torch.randn(num_nodes, self.input_dim).to(self.device)
                    with torch.no_grad():
                        fake_data = self.generator(noise, edge_index).to(self.device)

                    grad_penalty = self.compute_wgan_gp_loss(real_data, fake_data, self.lambda_rep, self.sigma, edge_index)

                    critics_fake = self.discriminator(fake_data, edge_index).reshape(-1)
                    critics_real = self.discriminator(real_data.x, edge_index).reshape(-1)
                    discriminator_loss = - (critics_real.mean() - critics_fake.mean()) + self.lambda_gp*grad_penalty.mean()

                    discriminator_loss.backward()
                    self.discriminator_optimizer.step()

                    for name, param in self.discriminator.named_parameters():
                        if torch.isnan(param).any():
                            print(f"NaN in Discriminator {name}")


                # Train the generator
            
                for _ in range(self.gene_it) :
                    self.generator_optimizer.zero_grad()

                    noise = torch.randn(num_nodes, self.input_dim).to(self.device)
                    fake_data = self.generator(noise, edge_index).to(self.device)
                    
                    critics_fake = self.discriminator(fake_data, edge_index).reshape(-1)
                    # loss_reg = self.lambda_reg * self.spectral_reg(fake_data, real_data.x)
                    loss_reg = 0
                    generator_loss = -critics_fake.mean() + loss_reg
                    generator_loss.backward()
                    self.generator_optimizer.step()

                    for name, param in self.generator.named_parameters():
                        if torch.isnan(param).any():
                            print(f"NaN in Generator {name}")


            ## Print, plot and save training progression                 

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item():.5f}, Discriminator Loss: {discriminator_loss.item():.5f}")

            if epoch % 1 == 0:
                # Enregistrer les événements pour TensorBoard
                writer.add_scalar('G_loss', generator_loss.item(), global_step=epoch)
                writer.add_scalar('D_loss', discriminator_loss.item(), global_step=epoch)
                for name, param in self.generator.named_parameters():
                    writer.add_histogram('generator/' + name, param, global_step=epoch)
                for name, param in self.discriminator.named_parameters():
                    writer.add_histogram('discriminator/' + name, param, global_step=epoch)
                
            if epoch % 5 == 0:
                with torch.no_grad():
                    
                    lon_concatenated = real_data.x[:, 0].cpu().numpy()
                    x_min = np.min(lon_concatenated)*1.2
                    x_max = np.max(lon_concatenated)*1.2

                    lat_concatenated = real_data.x[:, 1].cpu().numpy()
                    y_min = np.min(lat_concatenated)*1.2
                    y_max = np.max(lat_concatenated)*1.2

                    for i in range(min(4, real_data.num_graphs)):
                        img = plot_tensor(real_data[i].x[:, 0:2], x_min, x_max, y_min, y_max)
                        writer_real.add_image(f"Real_Graph_{i}", torchvision.transforms.ToTensor()(img), global_step=epoch)

                    lon_concatenated = fake_data[:, 0].cpu().numpy()
                    x_min = np.min(lon_concatenated)*1.2
                    x_max = np.max(lon_concatenated)*1.2

                    lat_concatenated = fake_data[:, 1].cpu().numpy()
                    y_min = np.min(lat_concatenated)*1.2
                    y_max = np.max(lat_concatenated)*1.2

                    for i in range(min(4, real_data.num_graphs)):
                        img = plot_tensor(fake_data[i*graph_nodes : (i+1)*graph_nodes, 0:2], x_min, x_max, y_min, y_max)
                        writer_fake.add_image(f"Fake_Graph_{i}", torchvision.transforms.ToTensor()(img), global_step=epoch)
                    

                
                # use the tools
                # real_dist = np.mean([get_dist_nest([0,0], real_data[i].x.cpu().numpy()) for i in range(real_data.num_graphs)], axis=0)
                real_dist = np.array([get_dist_nest([0,0], real_data[i].x.cpu().numpy()) for i in range(real_data.num_graphs)])
                real_steps = np.array([get_step_length(real_data[i].x.cpu().numpy()) for i in range(real_data.num_graphs)])

                with torch.no_grad():
                    for i in range(min(4, real_data.num_graphs)):
                        fake_traj = fake_data[i*graph_nodes : (i+1)*graph_nodes, :].cpu().numpy()

                        img = plot_dist(real_dist, torch.tensor(get_dist_nest(fake_traj[0, :2], fake_traj)))
                        writer_fake_dist.add_image(f"Fake_Dist_{i}", torchvision.transforms.ToTensor()(img), global_step=epoch)
                
                        img = plot_dist(real_steps, torch.tensor(get_step_length(fake_traj)))
                        writer_fake_steps.add_image(f"Fake_Steps_{i}", torchvision.transforms.ToTensor()(img), global_step=epoch)    

        print("Fin du training")

        writer.close()
    



       