# This script is corresponding to the paper below:
# Jian, X., Xia, Y., Duthé, G., Bacsa, K., Liu, W., & Chatzi, E. (2024). Using Graph Neural Networks and Frequency Domain Data for Automated Operational Modal Analysis of Populations of Structures. arXiv preprint arXiv:2407.06492.
# Link of the preprint: https://www.arxiv.org/abs/2407.06492
#%% Training
# use the two lines to solve "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/."
import os
# from sqlite3 import InterfaceError
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# pytorch lib
import random
import torch
import torch.nn as nn
from torch.nn import Parameter as Param
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as init
from torch_scatter import scatter_add
# geometric lib
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn import EdgeWeightNorm, GraphConv, GINConv
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
# other lib
import pandas as pd
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import copy
# load .mat file
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.sparse import csc_matrix, hstack, vstack
from scipy.sparse.linalg import inv
from scipy.linalg import sqrtm
import networkx as nx
import wandb
# from util import *
# torch.set_default_tensor_type(torch.DoubleTensor)

# fix Seed to make results reproducible
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
dgl.seed(seed)
dgl.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

# Define a GNN model
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats, hid_feats, aggregator_type='pool')
        self.conv2 = dglnn.SAGEConv(hid_feats, hid_feats, aggregator_type='pool')
        self.conv3 = dglnn.SAGEConv(hid_feats, out_feats, aggregator_type='pool')
        self.norm = EdgeWeightNorm(norm='both')
    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs) 
        h = F.relu(h)
        for i in range(1):
            h = self.conv2(graph, h)
            h = F.relu(h)
        h = self.conv3(graph, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hid_feats)
        self.conv2 = GraphConv(hid_feats, hid_feats)
        self.conv3 = GraphConv(hid_feats, out_feats)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for i in range(1):
            h = self.conv2(g, h)
            h = F.relu(h)
        h = self.conv3(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.GATConv(in_feats, hid_feats, num_heads=1)
        self.conv2 = dglnn.GATConv(hid_feats, hid_feats, num_heads=1)
        self.conv3 = dglnn.GATConv(hid_feats, out_feats, num_heads=1)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        for i in range(1):
            h = self.conv2(g, h)
            h = F.relu(h)
        h = self.conv3(g, h)
        return h

# Define a MLP as the encoder and decoder
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
                                nn.Linear(in_dim, hid_dim),   
                                nn.Dropout(p=0.1),  
                                nn.ReLU(),
                                nn.Linear(hid_dim, hid_dim),  
                                nn.Dropout(p=0.1),  
                                nn.ReLU(),
                                nn.Linear(hid_dim, out_dim)
                                        )
    def forward(self, x):
        # h contains the node representations computed from the GNN
        y = self.mlp(x)
        return y

modeN = 4  # number of modes to be identified

# Define the entire model
class Model(nn.Module):  # original model
    def __init__(self, encoder_hid_dim, encoder_out_dim, GNN_hid_dim, GNN_out_dim, decoder_hid_dim):
        super().__init__()
        self.freq_estimator = MLP(encoder_out_dim, encoder_hid_dim, modeN) # 
        self.damping_estimator = MLP(encoder_out_dim, encoder_hid_dim, modeN)
        self.node_encoder = MLP(input_dim, encoder_hid_dim, encoder_out_dim) # Define a MLP to encode the nodal features into node representations
        self.GNN = SAGE(encoder_out_dim, GNN_hid_dim, GNN_out_dim) # Message passing with GraphSAGE
        # self.GNN = GCN(encoder_out_dim, GNN_hid_dim, GNN_out_dim) # Message passing with GraphSAGE
        # self.GNN = GAT(encoder_out_dim, GNN_hid_dim, GNN_out_dim) # Message passing with GraphSAGE
        self.decoder = MLP(GNN_out_dim, decoder_hid_dim, modeN) # Define a MLP to decode node representations into mode shapes       
    def forward(self, g):
        h_node = self.node_encoder(g.ndata['acc_Y'])
        g.ndata['h'] = h_node
        h_graph = dgl.readout_nodes(g, 'h', op='mean')
        f = self.freq_estimator(h_graph)
        zeta = self.damping_estimator(h_graph)
        h_GNN = self.GNN(g, h_node)
        ms = self.decoder(h_GNN)
        return f, zeta, ms # return frequencies and mode shapes

# class Model(nn.Module):  # model without encoder
#     def __init__(self, encoder_hid_dim, encoder_out_dim, GNN_hid_dim, GNN_out_dim, decoder_hid_dim):
#         super().__init__()
#         self.freq_estimator = MLP(input_dim, encoder_hid_dim, modeN) # 
#         self.damping_estimator = MLP(input_dim, encoder_hid_dim, modeN)
#         self.GNN = SAGE(input_dim, GNN_hid_dim, GNN_out_dim) # Message passing with GraphSAGE
#         self.decoder = MLP(GNN_out_dim, decoder_hid_dim, modeN) # Define a MLP to decode node representations into mode shapes       
#     def forward(self, g):
#         h_graph = dgl.readout_nodes(g, 'acc_Y', op='mean')
#         f = self.freq_estimator(h_graph)
#         zeta = self.damping_estimator(h_graph)
#         h_GNN = self.GNN(g, g.ndata['acc_Y'])
#         ms = self.decoder(h_GNN)
#         return f, zeta, ms # return frequencies and mode shapes

# class Model(nn.Module):  # model using MLP to identify mode shapes
#     def __init__(self, encoder_hid_dim, encoder_out_dim, GNN_hid_dim, GNN_out_dim, decoder_hid_dim):
#         super().__init__()
#         self.freq_estimator = MLP(encoder_out_dim, encoder_hid_dim, modeN) # 
#         self.damping_estimator = MLP(encoder_out_dim, encoder_hid_dim, modeN)
#         self.node_encoder = MLP(input_dim, encoder_hid_dim, encoder_out_dim) # Define a MLP to encode the nodal features into node representations
#         self.MS_MLP = MLP(encoder_out_dim, encoder_hid_dim, GNN_out_dim)
#         self.decoder = MLP(encoder_out_dim, decoder_hid_dim, modeN) # Define a MLP to decode node representations into mode shapes       
#     def forward(self, g):
#         h_node = self.node_encoder(g.ndata['acc_Y'])  
#         h_MLP = self.MS_MLP(h_node)
#         ms = self.decoder(h_MLP)
#         g.ndata['h'] = h_node
#         h_graph = dgl.readout_nodes(g, 'h', op='mean')
#         f = self.freq_estimator(h_graph)
#         zeta = self.damping_estimator(h_graph)
#         return f, zeta, ms # return frequencies and mode shapes

# Define the Feature Propagation Func to fill unknown nodal features (without measurement)
def get_propagation_matrix(edge_index, n_nodes):
    # Initialize all edge weights to ones if the graph is unweighted)
    """
    Given an edge_index, return the same edge_index and edge weights computed as
    \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2}.
    """
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=n_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    DAD = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] 
    adj = torch.sparse.FloatTensor(edge_index, values=DAD, size=(n_nodes, n_nodes)).to(edge_index.device)
    return adj

def Feature_Propagation(x, mask, edge_index, n_nodes):
    # out is inizialized to 0 for missing values. However, its initialization does not matter for the final
    # value at convergence
    out = x
    if mask is not None:
        out = torch.zeros_like(x)
        out[mask] = x[mask]
    num_iterations = 30
    adj = get_propagation_matrix(edge_index, n_nodes)
    for _ in range(num_iterations):
        # Diffuse current features
        out = torch.sparse.mm(adj, out)
        # Reset original known features
        out[mask] = x[mask]
    return out

# Define the Modal Assurance Criterion funcion
def MAC(x, y):
    assert x.shape == y.shape, "Mode shapes must have the same shape"
    numerator = np.abs(np.dot(x, y.T))**2
    denominator_x = np.dot(x, x.T)
    denominator_y = np.dot(y, y.T)
    mac = numerator / (denominator_x * denominator_y)
    return mac

# Load data from the current folder
mat_contents = sio.loadmat("./trapezoid_pwelch_bottom_input.mat") # pupolation 1

acc_input = mat_contents['acceleration_pwelch'][:, 0]
input_dim = 1025
freq = mat_contents['frequency_out'][:, 0]
zeta = mat_contents['damping_out'][:, 0]
phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
node = mat_contents['node_out'][:, 0]
element = mat_contents['element_out'][:, 0]

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# prepare dataset
class Dataset(DGLDataset):
    def __init__(self, graph_ids):
        self.graph_ids = graph_ids # these definition of ids must be put above the super(), otherwise error happens 
        super(Dataset, self).__init__(name='ModalGNN')   
    def process(self):
        self.graphs = []
        self.freqs = []
        self.zetas = []
        # For each graph ID...
        for graph_id in self.graph_ids:
            # Create a graph and add it to the list of graphs and labels.  
            src = np.concatenate((element[graph_id][:,0], element[graph_id][:,1]), axis=0)-1 # bi-directional edge, left-end node no. (python starts from 0 so minus 1)
            dst = np.concatenate((element[graph_id][:,1], element[graph_id][:,0]), axis=0)-1 # bi-directional edge, right-end node no.
            graph_sub = dgl.graph((src, dst))  #       
            graph_sub.ndata['acc_Y'] = torch.tensor(acc_input[graph_id][:, :input_dim], dtype = torch.float)
            graph_sub.ndata['phi_Y'] = torch.tensor(phi[graph_id][:, 0:modeN], dtype = torch.float)  
            g = graph_sub.to(device)
            graph_freq = freq[graph_id][:modeN].squeeze()
            graph_zeta = zeta[graph_id][:modeN].squeeze()
            self.graphs.append(g)
            self.freqs.append(graph_freq)
            self.zetas.append(graph_zeta)
        # Convert the label list to tensor for saving.
        # self.freqs = torch.LongTensor(self.freqs)
        self.freqs = torch.tensor(self.freqs, dtype = torch.float).to(device)
        self.zetas = torch.tensor(self.zetas, dtype = torch.float).to(device)
    def __getitem__(self, i):
        return self.graphs[i], self.freqs[i], self.zetas[i]
    def __len__(self):
        return len(self.graphs)

N_all = 500
N_train = int(320*1)
N_valid = int(80*1)

# k-fold cross validation
kk = 5  # from 1 to 5 #####################################################
valid_no = np.array(range(0, N_valid)) + (kk-1)*N_valid
train_no = np.setdiff1d(np.array(range(0,N_train+N_valid)), valid_no)

valid_set = Dataset(graph_ids=valid_no)
train_set = Dataset(graph_ids=train_no)

hyper_factor = 2
model = Model(encoder_hid_dim = 32 * hyper_factor,
              encoder_out_dim = 32 * hyper_factor,
              GNN_hid_dim = 32 * hyper_factor,
              GNN_out_dim = 32 * hyper_factor,
              decoder_hid_dim = 32 * hyper_factor)  # each node has 1 output features corresponding to the first order of modeshapes

bs = 64
dataloader_train = dgl.dataloading.GraphDataLoader(train_set, batch_size=bs,
                              drop_last=True, shuffle=True)
dataloader_valid = dgl.dataloading.GraphDataLoader(valid_set, batch_size=bs,
                              drop_last=True, shuffle=True)
#%% Train model
plt.close('all')
model.to(device)
loss_mse = nn.MSELoss()

n_epoch = 5000
learning_rate = 0.001
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
start_time = time.time()

loss_meter_train = []
loss_meter_valid = []

loss_meter_train_phi = []
loss_meter_train_zeta = []
loss_meter_train_freq = []

loss_meter_valid_phi = []
loss_meter_valid_zeta = []
loss_meter_valid_freq = []

model.train()

lambda1 = 2
lambda2 = 1
lambda3 = 1

for epoch in range(n_epoch):
    epoch_loss_valid_phi = 0
    epoch_loss_valid_zeta = 0
    epoch_loss_valid_freq = 0
    epoch_loss_valid = 0
    for graph_valid, freq_valid, zeta_valid in dataloader_valid:
        freq_pred_valid, zeta_pred_valid, phi_pred_valid = model(graph_valid)  # valid with complete observations 
        # loss about phi
        loss_valid_phi = loss_mse(phi_pred_valid.squeeze(), graph_valid.ndata['phi_Y']) * lambda1
        # loss about zeta
        loss_valid_zeta = loss_mse(zeta_pred_valid/zeta_valid, torch.ones(zeta_valid.shape).to(device)) * lambda2
        # loss about freq
        loss_valid_freq = loss_mse(freq_pred_valid/freq_valid, torch.ones(freq_valid.shape).to(device)) * lambda3
        # total loss
        loss_valid =  loss_valid_phi + loss_valid_zeta + loss_valid_freq
        epoch_loss_valid_phi += loss_valid_phi.detach()
        epoch_loss_valid_zeta += loss_valid_zeta.detach()
        epoch_loss_valid_freq += loss_valid_freq.detach()
        epoch_loss_valid += loss_valid.detach()
    epoch_loss_valid_phi /= len(dataloader_valid)
    epoch_loss_valid_zeta /= len(dataloader_valid)
    epoch_loss_valid_freq /= len(dataloader_valid)
    epoch_loss_valid /= len(dataloader_valid)
    loss_meter_valid_phi.append(epoch_loss_valid_phi.cpu().numpy())
    loss_meter_valid_freq.append(epoch_loss_valid_freq.cpu().numpy())
    loss_meter_valid_zeta.append(epoch_loss_valid_zeta.cpu().numpy())
    loss_meter_valid.append(epoch_loss_valid.cpu().numpy())
    
    epoch_loss_train_phi = 0
    epoch_loss_train_zeta = 0
    epoch_loss_train_freq = 0
    epoch_loss_train = 0
    for graph_train, freq_train, zeta_train in dataloader_train:
        freq_pred_train, zeta_pred_train, phi_pred_train = model(graph_train)  # train with complete observations
        # loss about phi
        loss_train_phi = loss_mse(phi_pred_train.squeeze(), graph_train.ndata['phi_Y']) * lambda1
        # loss about zeta
        loss_train_zeta = loss_mse(zeta_pred_train/zeta_train, torch.ones(zeta_train.shape).to(device)) * lambda2
        # loss about freq
        loss_train_freq = loss_mse(freq_pred_train/freq_train, torch.ones(freq_train.shape).to(device)) * lambda3
        # total loss
        loss_train = loss_train_phi + loss_train_zeta + loss_train_freq
        epoch_loss_train_phi += loss_train_phi.detach()
        epoch_loss_train_zeta += loss_train_zeta.detach()
        epoch_loss_train_freq += loss_train_freq.detach()
        epoch_loss_train += loss_train.detach()
        opt.zero_grad()
        loss_train.backward()
        opt.step()
    epoch_loss_train_phi /= len(dataloader_train)
    epoch_loss_train_zeta /= len(dataloader_train)
    epoch_loss_train_freq /= len(dataloader_train)
    epoch_loss_train /= len(dataloader_train)
    loss_meter_train_phi.append(epoch_loss_train_phi.cpu().numpy())
    loss_meter_train_freq.append(epoch_loss_train_freq.cpu().numpy())
    loss_meter_train_zeta.append(epoch_loss_train_zeta.cpu().numpy())
    loss_meter_train.append(epoch_loss_train.cpu().numpy())
    
    if epoch % 50 == 0:
        print('epoch: {}, loss_train: {:.6f}, loss_valid: {:.6f}' .format(epoch, epoch_loss_train, epoch_loss_valid))
    
time_train = time.time() - start_time
print("--- %s seconds ---" % time_train)

# plot loss curves
plt.close('all')

plt.figure()
plt.plot(np.log10(loss_meter_train), label='training', color='#FF1F5B')
plt.plot(np.log10(loss_meter_valid), label='validation', color='#00CD6C')
title_text = "Final train loss={:.6f}, Final valid loss={:.6f}, Time={:.3f}".format(loss_meter_train[-1], loss_meter_valid[-1], time_train)
plt.title(title_text)
plt.xlabel('Epoch', fontsize=18, fontname='Times New Roman')
plt.ylabel('log10(Loss)', fontsize=18, fontname='Times New Roman')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(4*2.54/2.54, 3*2.54/2.54))
# plt.plot(np.log10(loss_meter_train), label='training', color='#FF1F5B')
# plt.plot(np.log10(loss_meter_valid), label='validation', color='#00CD6C')
plt.semilogy(loss_meter_train, label='training', color='#FF1F5B')
plt.semilogy(loss_meter_valid, label='validation', color='#00CD6C')
plt.ylim([-4, 2])
plt.xlabel('Epoch', fontsize=18, fontname='Times New Roman')
plt.ylabel('log10(Loss)', fontsize=18, fontname='Times New Roman')
plt.xticks(np.arange(0, n_epoch+1, 2500))
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.2, labelspacing=0.1)
plt.tight_layout()
plt.grid()
plt.show()


plt.figure(figsize=(4*2.54/2.54, 3*2.54/2.54))
plt.plot(np.log10(loss_meter_train_phi), label='$\hat{\u03A6}$', color='#FF1F5B')
plt.plot(np.log10(loss_meter_train_zeta), label='$\hat{Z}$', color='#00CD6C')
plt.plot(np.log10(loss_meter_train_freq), label='$\hat{F}$', color='#AF58BA')
# title_text = "Final phi loss={:.6f}, zeta loss={:.6f}, freq loss={:.6f}".format(loss_meter_valid_phi[-1], loss_meter_valid_zeta[-1], loss_meter_valid_freq[-1])
# plt.title(title_text)
plt.ylim([-4, 2])
plt.xlabel('Epoch', fontsize=18, fontname='Times New Roman')
plt.ylabel('log10(Loss)', fontsize=18, fontname='Times New Roman')
plt.xticks(np.arange(0, n_epoch+1, 2500))
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=0.6, borderpad=0.2, columnspacing=0.2, ncol=3)
plt.tight_layout()
plt.grid()
plt.show()

plt.figure(figsize=(4*2.54/2.54, 3*2.54/2.54))
plt.plot(np.log10(loss_meter_valid_phi), label='$\hat{\u03A6}$', color='#FF1F5B')
plt.plot(np.log10(loss_meter_valid_zeta), label='$\hat{Z}$', color='#00CD6C')
plt.plot(np.log10(loss_meter_valid_freq), label='$\hat{F}$', color='#AF58BA')
# title_text = "Final phi loss={:.6f}, zeta loss={:.6f}, freq loss={:.6f}".format(loss_meter_valid_phi[-1], loss_meter_valid_zeta[-1], loss_meter_valid_freq[-1])
# plt.title(title_text)
plt.ylim([-4, 2])
plt.xlabel('Epoch', fontsize=18, fontname='Times New Roman')
plt.ylabel('log10(Loss)', fontsize=18, fontname='Times New Roman')
plt.xticks(np.arange(0, n_epoch+1, 2500))
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=0.6, borderpad=0.2, columnspacing=0.2, ncol=3)
plt.tight_layout()
plt.grid()
plt.show()

print(kk)
# %% save model if needed
# PATH = "model_SAGE.pt"
# torch.save(model.state_dict(), PATH)
# %% test trained model with population 1
N_test = 100 # use the last 100 trusses for testing

# load test data
# pupolation 1 - original
mat_contents = sio.loadmat("./trapezoid_pwelch_bottom_input.mat") # pupolation 1
# pupolation 1 - noise polluted
# mat_contents = sio.loadmat("./trapezoid_pwelch_bottom_input_noise10.mat")

acc_input = mat_contents['acceleration_pwelch'][:, 0]
input_dim = 1025
freq = mat_contents['frequency_out'][:, 0]
zeta = mat_contents['damping_out'][:, 0]
phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
node = mat_contents['node_out'][:, 0]
element = mat_contents['element_out'][:, 0]

# load trained model
kk = 5  # cross validation, from 1 to 5, here using 5 because the model is trained with the 5th fold
# test the model trained with 40 trusses
# PATH = "model_SAGE_training40.pt"
# N_train = int(320*0.1)
# N_valid = int(80*0.1)
# test the model trained with 200 trusses
PATH = "model_SAGE_training200.pt"
N_train = int(320*0.5)
N_valid = int(80*0.5)
# test the model trained with 400 trusses
# PATH = "model_SAGE_training400.pt"
# N_train = int(320*1)
# N_valid = int(80*1)

valid_no = np.array(range(0, N_valid)) + (kk-1)*N_valid
train_no = np.setdiff1d(np.array(range(0,N_train+N_valid)), valid_no)

model.load_state_dict(torch.load(PATH))
model.eval()
model = model.cpu()

# test with a single sample #################################################
caseN = 10 - 1 + 400
test_data = Dataset(graph_ids = [caseN])[0]
graph_test = test_data[0].cpu()
freq_test_true = test_data[1].cpu()
zeta_test_true = test_data[2].cpu()
    
node_test = node[caseN]
element_test = element[caseN] - 1

node_mask = torch.ones(len(node_test), dtype=torch.bool)

# uncomment the first two missing indices: 66% unknown node features
# uncomment three missing indices: 82% unknown node features
# missing_indices = np.array(range(1, len(node_test), 2))
# node_mask[missing_indices] = False
# missing_indices = np.array(range(1, len(node_test), 3))
# node_mask[missing_indices] = False
# missing_indices = np.array(range(2, len(node_test), 3))
# node_mask[missing_indices] = False
# missing_indices = np.array(range(3, len(node_test), 3))
# node_mask[missing_indices] = False

missing_ratio = np.count_nonzero(node_mask == False)/len(node_mask)
print('missing_ratio =', missing_ratio)
src = np.concatenate((element_test[:,0], element_test[:,1]), axis=0) # bi-directional edge, left-end node no. 
dst = np.concatenate((element_test[:,1], element_test[:,0]), axis=0) # bi-directional edge, right-end node no.
edge_index =  torch.LongTensor([list(row) for row in list(zip(src, dst))]).T

acc_test = graph_test.ndata['acc_Y']
acc_test_FP = Feature_Propagation(acc_test, node_mask, edge_index, len(node_test))
acc_test_FP[acc_test[:,1]==0, :] = 0
graph_test.ndata['acc_Y'] = acc_test_FP

magnify = 3 # magnify the mode shapes for better visualization
marker_size = 5
plt.close('all')
fig, axs = plt.subplots(2, 2, figsize=(9, 3.5), layout="constrained")
for mode_order, ax in enumerate(axs.flat):
    phi_test_true = graph_test.ndata['phi_Y'][:, mode_order]
    freq_test_pred, zeta_test_pred, phi_test_pred = model(graph_test)
    phi_test_pred = phi_test_pred[:, mode_order] / phi_test_pred[:, mode_order].max()
    MSE_ms_model = loss_mse(phi_test_pred, phi_test_true)

    # plot the truss
    node_pred = np.zeros([len(node_test), 2])
    node_pred[:, 0] = node_test[:, 0]
    node_pred[:, 1] = node_test[:, 1] + phi_test_pred.detach().numpy().squeeze() * magnify
    node_true = np.zeros([len(node_test), 2])
    node_true[:, 0] = node_test[:, 0]
    node_true[:, 1] = node_test[:, 1] + phi_test_true.detach().numpy().squeeze() * magnify
    
    MAC_phi = MAC(phi_test_pred.detach().numpy().squeeze(), phi_test_true.detach().numpy().squeeze())
    
    # Plot mode shapes
    for ele in element_test:
        node1 = node_test[ele[0]]
        node2 = node_test[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='#FFC61E')
    ax.plot(node_test[:, 0], node_test[:, 1], 'o', markersize=marker_size, label='undeformed', color='#FFC61E')
    for ele in element_test:
        node1 = node_true[ele[0]]
        node2 = node_true[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], '--', color='#FF1F5B')
    ax.plot(node_true[:, 0], node_true[:, 1], 'o', markersize=marker_size, label='true', color='#FF1F5B')
    for ele in element_test:
        node1 = node_pred[ele[0]]
        node2 = node_pred[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='#AF58BA')
    # complete measurements    
    ax.plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=marker_size, label='identified', color='#AF58BA')
    # incomplete measurements  
    # ax.plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=marker_size, label='identified_known', color='#AF58BA')
    # ax.plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', markersize=3, label='identified_unknown', color='#00CD6C')
    
    plt.setp(ax.get_xticklabels(), fontsize=10, fontname='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontsize=10, fontname='Times New Roman')
    
    ax.set_xlim(-42,42)
    ax.set_ylim(-3,10)
    ax.set_xlabel('X (m)', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('Y (m)', fontsize=12, fontname='Times New Roman')
    title_text = "Mode {:.0f}, MAC={:.5f}".format(mode_order+1, MAC_phi)
    ax.set_title(title_text, fontsize=12, fontname='Times New Roman')   
    # ax.set_aspect('equal')
    ax.grid()
    
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, prop={'family': 'Times New Roman'}, fontsize=10, frameon=False)
plt.tight_layout(rect=[0, 0.01, 1, 0.92])
plt.show()

print("True Freq:", freq_test_true)
print("Pred Freq:", freq_test_pred.squeeze())

print("True Zeta:", zeta_test_true)
print("Pred Zeta:", zeta_test_pred.squeeze())

# test trained model - entire dataset ########################################
freq_pred = np.zeros([N_all, modeN])
zeta_pred = np.zeros([N_all, modeN])
phi_pred = []

start_time = time.time()
# modal identification
for caseN in range(N_all):
    if caseN % 50 == 0:
        print(caseN)   
    test_data = Dataset(graph_ids = [caseN])[0]
    graph_test = test_data[0].cpu()           
    node_test = node[caseN]
    element_test = element[caseN] - 1   
    node_mask = torch.ones(len(node_test), dtype=torch.bool)
    
    # uncomment the first two missing indices: 66% unknown node features
    # uncomment three missing indices: 82% unknown node features
    # missing_indices = np.array(range(1, len(node_test), 2))
    # node_mask[missing_indices] = False
    # missing_indices = np.array(range(1, len(node_test), 3))
    # node_mask[missing_indices] = False
    # missing_indices = np.array(range(2, len(node_test), 3))
    # node_mask[missing_indices] = False
    
    missing_ratio = np.count_nonzero(node_mask == False)/len(node_mask)
    
    src = np.concatenate((element_test[:,0], element_test[:,1]), axis=0) # bi-directional edge, left-end node no. 
    dst = np.concatenate((element_test[:,1], element_test[:,0]), axis=0) # bi-directional edge, right-end node no.
    edge_index =  torch.LongTensor([list(row) for row in list(zip(src, dst))]).T
    
    acc_test = graph_test.ndata['acc_Y']
    acc_test_FP = Feature_Propagation(acc_test, node_mask, edge_index, len(node_test))
    acc_test_FP[acc_test[:,1]==0, :] = 0
    graph_test.ndata['acc_Y'] = acc_test_FP
    
    freq_test_pred, zeta_test_pred, phi_test_pred = model(graph_test)
    freq_pred[caseN, :] = freq_test_pred.squeeze().detach().numpy()
    zeta_pred[caseN, :] = zeta_test_pred.squeeze().detach().numpy()
    phi_pred.append(phi_test_pred.detach().numpy())
        
print("--- %s seconds ---" % (time.time() - start_time))
print('missing_ratio =', missing_ratio)

# calculate evaluation indicators
freq_true = np.zeros([N_all, modeN])
zeta_true = np.zeros([N_all, modeN])
MAC_phi = np.zeros([N_all, modeN])
RE_freq = np.zeros([N_all, modeN])
RE_zeta = np.zeros([N_all, modeN])
for mode_order in range(modeN):
    for caseN in range(N_all):
        if caseN % 50 == 0:
            print(caseN)    
        
        freq_test_true = freq[caseN][mode_order]
        freq_true[caseN, mode_order] = freq_test_true
        zeta_test_true = zeta[caseN][mode_order]
        zeta_true[caseN, mode_order] = zeta_test_true
        phi_test_true = phi[caseN][:, mode_order]

        RE_freq[caseN, mode_order] = (freq_pred[caseN, mode_order] - freq_test_true) / freq_test_true  # relative error in percentage
        RE_zeta[caseN, mode_order] = (zeta_pred[caseN, mode_order] - zeta_test_true) / zeta_test_true
        MAC_phi[caseN, mode_order] = MAC(phi_pred[caseN][:, mode_order], phi_test_true)

# plt.close('all')
fig, ax = plt.subplots(3, 1, layout="constrained")
for k in range(3):
    if k == 0:
        ax[k].plot(MAC_phi[:, 3], label='model')
        # ax[k].plot(MAC_pp, label='peak picking')
        ax[k].set_ylabel('MAC', fontsize=14)
        ax[k].set_title("Mode Shape MAC", fontsize=14)
        ax[k].legend(fontsize=14)
    elif k == 1:
        ax[k].plot(RE_freq[:, 3])
        ax[k].set_ylabel('Relative Error', fontsize=14)
        ax[k].set_title("Frequency MSE", fontsize=14)
        ax[k].set_xlabel('Sample No.', fontsize=14)
    else:
        ax[k].plot(RE_zeta[:, 3])
        ax[k].set_ylabel('Relative Error', fontsize=14)
        ax[k].set_title("Damping ratio MSE", fontsize=14)
        ax[k].set_xlabel('Sample No.', fontsize=14)
    ax[k].grid()

statistics_phi = np.zeros([3, modeN])
statistics_freq = np.zeros([3, modeN])
statistics_zeta = np.zeros([3, modeN])

for j in range(modeN):
    # mode shape
    statistics_phi[0, j] = np.mean(MAC_phi[-N_test:, j])
    statistics_phi[1, j] = np.std(MAC_phi[-N_test:, j])
    statistics_phi[2, j] = np.min(MAC_phi[-N_test:, j])
    # frequency
    statistics_freq[0, j] = np.mean(RE_freq[-N_test:, j])
    statistics_freq[1, j] = np.std(RE_freq[-N_test:, j])
    statistics_freq[2, j] = np.max(abs(RE_freq[-N_test:, j]))
    # damping
    statistics_zeta[0, j] = np.mean(RE_zeta[-N_test:, j])
    statistics_zeta[1, j] = np.std(RE_zeta[-N_test:, j])
    statistics_zeta[2, j] = np.max(abs(RE_zeta[-N_test:, j]))

statistics_phi = np.transpose(statistics_phi)
statistics_zeta = np.transpose(statistics_zeta) * 100
statistics_freq = np.transpose(statistics_freq) * 100
    
print("Mode shape")
print(statistics_phi)
print("Damping")
print(statistics_zeta)
print("Freq")
print(statistics_freq)

# mode shape MAC results of training set
marker_size = 10
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(np.zeros([N_train, 1])+1, MAC_phi[train_no, 0], color='#FF1F5B', label='mode 1', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_train, 1])+2, MAC_phi[train_no, 1], color='#00CD6C', label='mode 2', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_train, 1])+3, MAC_phi[train_no, 2], color='#FFC61E', label='mode 3', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_train, 1])+4, MAC_phi[train_no, 3], color='#AF58BA', label='mode 4', s=marker_size, alpha=1)
plt.boxplot(MAC_phi[train_no, :], 0, '')
plt.xticks([1, 2, 3, 4], ['Mode1', 'Mode2', 'Mode3', 'Mode4'])
plt.ylim([0, 1.05])
plt.xticks(fontsize=17, fontname='Times New Roman')
plt.yticks(fontsize=17, fontname='Times New Roman')
plt.ylabel('MAC', fontname='Times New Roman', fontsize=17)
plt.grid()
plt.tight_layout()
plt.show()

# mode shape MAC results of testing set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(np.zeros([N_test, 1])+1, MAC_phi[-N_test:, 0], color='#FF1F5B', label='mode 1', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+2, MAC_phi[-N_test:, 1], color='#00CD6C', label='mode 2', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+3, MAC_phi[-N_test:, 2], color='#FFC61E', label='mode 3', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+4, MAC_phi[-N_test:, 3], color='#AF58BA', label='mode 4', s=marker_size, alpha=1)
plt.boxplot(MAC_phi[-N_test:, :], 0, '')
plt.xticks([1, 2, 3, 4], ['Mode1', 'Mode2', 'Mode3', 'Mode4'])
plt.ylim([0, 1.05])
plt.xticks(fontsize=17, fontname='Times New Roman')
plt.yticks(fontsize=17, fontname='Times New Roman')
plt.ylabel('MAC', fontname='Times New Roman', fontsize=17)
plt.grid()
plt.tight_layout()
plt.show()

# frequency results of training set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(freq_true[train_no, 0], freq_pred[train_no, 0], color='#FF1F5B', label='mode 1', s=marker_size, alpha=1)
plt.scatter(freq_true[train_no, 1], freq_pred[train_no, 1], color='#00CD6C', label='mode 2', s=marker_size, alpha=1)
plt.scatter(freq_true[train_no, 2], freq_pred[train_no, 2], color='#FFC61E', label='mode 3', s=marker_size, alpha=1)
plt.scatter(freq_true[train_no, 3], freq_pred[train_no, 3], color='#AF58BA', label='mode 4', s=marker_size, alpha=1)
plt.plot([0,30], [0,30], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,30], [0,30*0.9], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,30*0.9], [0,30], linestyle='--', color='blue')
plt.xlim([0,25])
plt.ylim([0,25])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()

# frequency results of testing set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(freq_true[-N_test:, 0], freq_pred[-N_test:, 0], color='#FF1F5B', label='mode 1', s=marker_size)
plt.scatter(freq_true[-N_test:, 1], freq_pred[-N_test:, 1], color='#00CD6C', label='mode 2', s=marker_size)
plt.scatter(freq_true[-N_test:, 2], freq_pred[-N_test:, 2], color='#FFC61E', label='mode 3', s=marker_size)
plt.scatter(freq_true[-N_test:, 3], freq_pred[-N_test:, 3], color='#AF58BA', label='mode 4', s=marker_size)
plt.plot([0,30], [0,30], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,30], [0,30*0.9], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,30*0.9], [0,30], linestyle='--', color='blue')
plt.xlim([0,25])
plt.ylim([0,25])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()

zeta_limit = 0.011
# damping ratio results of training set %
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(zeta_true[train_no, 0]*100, zeta_pred[train_no, 0]*100, color='#FF1F5B', label='mode 1', s=marker_size)
plt.scatter(zeta_true[train_no, 1]*100, zeta_pred[train_no, 1]*100, color='#00CD6C', label='mode 2', s=marker_size)
plt.scatter(zeta_true[train_no, 2]*100, zeta_pred[train_no, 2]*100, color='#FFC61E', label='mode 3', s=marker_size)
plt.scatter(zeta_true[train_no, 3]*100, zeta_pred[train_no, 3]*100, color='#AF58BA', label='mode 4', s=marker_size)
plt.plot([0,zeta_limit*100], [0,zeta_limit*100], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,zeta_limit*100], [0,zeta_limit*0.9*100], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,zeta_limit*0.9*100], [0,zeta_limit*100], linestyle='--', color='blue')
plt.xlim([0.003*100,zeta_limit*100])
plt.ylim([0.003*100,zeta_limit*100])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Damping Ratio (%)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Damping Ratio (%)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()

# damping ratio results of testing set %
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(zeta_true[-N_test:, 0]*100, zeta_pred[-N_test:, 0]*100, color='#FF1F5B', label='mode 1', s=marker_size)
plt.scatter(zeta_true[-N_test:, 1]*100, zeta_pred[-N_test:, 1]*100, color='#00CD6C', label='mode 2', s=marker_size)
plt.scatter(zeta_true[-N_test:, 2]*100, zeta_pred[-N_test:, 2]*100, color='#FFC61E', label='mode 3', s=marker_size)
plt.scatter(zeta_true[-N_test:, 3]*100, zeta_pred[-N_test:, 3]*100, color='#AF58BA', label='mode 4', s=marker_size)
plt.plot([0,zeta_limit*100], [0,zeta_limit*100], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,zeta_limit*100], [0,zeta_limit*0.9*100], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,zeta_limit*0.9*100], [0,zeta_limit*100], linestyle='--', color='blue')
plt.xlim([0.003*100,zeta_limit*100])
plt.ylim([0.003*100,zeta_limit*100])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Damping Ratio (%)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Damping Ratio (%)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()
# %% test trained model with population 2
# load test data
# pupolation 2- cantilever
mat_contents = sio.loadmat("./trapezoid_pwelch_bottom_input_cantilever.mat") 
N_all = 100 # since the dataset size if different, need to modify the division of training and testing set
N_test = 100
N_train = 0

acc_input = mat_contents['acceleration_pwelch'][:, 0]
input_dim = 1025
freq = mat_contents['frequency_out'][:, 0]
zeta = mat_contents['damping_out'][:, 0]
phi = abs(mat_contents['modeshape_out'][:, 0])*1  # absolute mode shape
node = mat_contents['node_out'][:, 0]
element = mat_contents['element_out'][:, 0]

# load trained model
PATH = "model_SAGE_training400.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

model = model.cpu()

# test with a single sample
caseN = 10 - 1 + 40
test_data = Dataset(graph_ids = [caseN])[0]
graph_test = test_data[0].cpu()
freq_test_true = test_data[1].cpu()
zeta_test_true = test_data[2].cpu()
    
node_test = node[caseN]
element_test = element[caseN] - 1

node_mask = torch.ones(len(node_test), dtype=torch.bool)

# missing_indices = np.array(range(1, len(node_test), 2))
# node_mask[missing_indices] = False
# missing_indices = np.array(range(1, len(node_test), 3))
# node_mask[missing_indices] = False
# missing_indices = np.array(range(2, len(node_test), 3))
# node_mask[missing_indices] = False
# missing_indices = np.array(range(3, len(node_test), 3))
# node_mask[missing_indices] = False

missing_ratio = np.count_nonzero(node_mask == False)/len(node_mask)
print('missing_ratio =', missing_ratio)
src = np.concatenate((element_test[:,0], element_test[:,1]), axis=0) # bi-directional edge, left-end node no. 
dst = np.concatenate((element_test[:,1], element_test[:,0]), axis=0) # bi-directional edge, right-end node no.
edge_index =  torch.LongTensor([list(row) for row in list(zip(src, dst))]).T

acc_test = graph_test.ndata['acc_Y']
acc_test_FP = Feature_Propagation(acc_test, node_mask, edge_index, len(node_test))
acc_test_FP[acc_test[:,1]==0, :] = 0
graph_test.ndata['acc_Y'] = acc_test_FP

magnify = 3
marker_size = 5
plt.close('all')
fig, axs = plt.subplots(2, 2, figsize=(9, 3.5), layout="constrained")
for mode_order, ax in enumerate(axs.flat):
    phi_test_true = graph_test.ndata['phi_Y'][:, mode_order]
    freq_test_pred, zeta_test_pred, phi_test_pred = model(graph_test)
    phi_test_pred = phi_test_pred[:, mode_order] / phi_test_pred[:, mode_order].max()
    MSE_ms_model = loss_mse(phi_test_pred, phi_test_true)

    # plot the truss
    node_pred = np.zeros([len(node_test), 2])
    node_pred[:, 0] = node_test[:, 0]
    node_pred[:, 1] = node_test[:, 1] + phi_test_pred.detach().numpy().squeeze() * magnify
    node_true = np.zeros([len(node_test), 2])
    node_true[:, 0] = node_test[:, 0]
    node_true[:, 1] = node_test[:, 1] + phi_test_true.detach().numpy().squeeze() * magnify
    
    MAC_phi = MAC(phi_test_pred.detach().numpy().squeeze(), phi_test_true.detach().numpy().squeeze())
    
    # Plot mode shapes
    for ele in element_test:
        node1 = node_test[ele[0]]
        node2 = node_test[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='#FFC61E')
    ax.plot(node_test[:, 0], node_test[:, 1], 'o', markersize=marker_size, label='undeformed', color='#FFC61E')
    for ele in element_test:
        node1 = node_true[ele[0]]
        node2 = node_true[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], '--', color='#FF1F5B')
    ax.plot(node_true[:, 0], node_true[:, 1], 'o', markersize=marker_size, label='true', color='#FF1F5B')
    for ele in element_test:
        node1 = node_pred[ele[0]]
        node2 = node_pred[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='#AF58BA')
    # complete measurements    
    ax.plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=marker_size, label='identified', color='#AF58BA')
    # incomplete measurements  
    # ax.plot(node_pred[:, 0], node_pred[:, 1], 'o', markersize=marker_size, label='identified_known', color='#AF58BA')
    # ax.plot(node_pred[~node_mask, 0], node_pred[~node_mask, 1], 's', markersize=3, label='identified_unknown', color='#00CD6C')
    
    plt.setp(ax.get_xticklabels(), fontsize=10, fontname='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontsize=10, fontname='Times New Roman')
    
    ax.set_xlim(-42,42)
    ax.set_ylim(-3,10)
    ax.set_xlabel('X (m)', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('Y (m)', fontsize=12, fontname='Times New Roman')
    title_text = "Mode {:.0f}, MAC={:.5f}".format(mode_order+1, MAC_phi)
    ax.set_title(title_text, fontsize=12, fontname='Times New Roman')   
    # ax.set_aspect('equal')
    ax.grid()
    
lines, labels = fig.axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=4, prop={'family': 'Times New Roman'}, fontsize=10, frameon=False)
plt.tight_layout(rect=[0, 0.01, 1, 0.92])
plt.show()

print("True Freq:", freq_test_true)
print("Pred Freq:", freq_test_pred.squeeze())

print("True Zeta:", zeta_test_true)
print("Pred Zeta:", zeta_test_pred.squeeze())

# test trained model - entire dataset
freq_pred = np.zeros([N_all, modeN])
zeta_pred = np.zeros([N_all, modeN])
phi_pred = []

start_time = time.time()
# modal identification
for caseN in range(N_all):
    if caseN % 50 == 0:
        print(caseN)   
    test_data = Dataset(graph_ids = [caseN])[0]
    graph_test = test_data[0].cpu()           
    node_test = node[caseN]
    element_test = element[caseN] - 1   
    node_mask = torch.ones(len(node_test), dtype=torch.bool)
    
    # first two indices: 66% unknown, three indices: 82% unknown
    # missing_indices = np.array(range(1, len(node_test), 2))
    # node_mask[missing_indices] = False
    # missing_indices = np.array(range(1, len(node_test), 3))
    # node_mask[missing_indices] = False
    # missing_indices = np.array(range(2, len(node_test), 3))
    # node_mask[missing_indices] = False
    
    missing_ratio = np.count_nonzero(node_mask == False)/len(node_mask)
    
    src = np.concatenate((element_test[:,0], element_test[:,1]), axis=0) # bi-directional edge, left-end node no. 
    dst = np.concatenate((element_test[:,1], element_test[:,0]), axis=0) # bi-directional edge, right-end node no.
    edge_index =  torch.LongTensor([list(row) for row in list(zip(src, dst))]).T
    
    acc_test = graph_test.ndata['acc_Y']
    acc_test_FP = Feature_Propagation(acc_test, node_mask, edge_index, len(node_test))
    acc_test_FP[acc_test[:,1]==0, :] = 0
    graph_test.ndata['acc_Y'] = acc_test_FP
    
    freq_test_pred, zeta_test_pred, phi_test_pred = model(graph_test)
    freq_pred[caseN, :] = freq_test_pred.squeeze().detach().numpy()
    zeta_pred[caseN, :] = zeta_test_pred.squeeze().detach().numpy()
    phi_pred.append(phi_test_pred.detach().numpy())
        
print("--- %s seconds ---" % (time.time() - start_time))
print('missing_ratio =', missing_ratio)

# calculate evaluation indicators
freq_true = np.zeros([N_all, modeN])
zeta_true = np.zeros([N_all, modeN])
MAC_phi = np.zeros([N_all, modeN])
RE_freq = np.zeros([N_all, modeN])
RE_zeta = np.zeros([N_all, modeN])
for mode_order in range(modeN):
    for caseN in range(N_all):
        if caseN % 50 == 0:
            print(caseN)    
        
        freq_test_true = freq[caseN][mode_order]
        freq_true[caseN, mode_order] = freq_test_true
        zeta_test_true = zeta[caseN][mode_order]
        zeta_true[caseN, mode_order] = zeta_test_true
        phi_test_true = phi[caseN][:, mode_order]

        RE_freq[caseN, mode_order] = (freq_pred[caseN, mode_order] - freq_test_true) / freq_test_true  # relative error in percentage
        RE_zeta[caseN, mode_order] = (zeta_pred[caseN, mode_order] - zeta_test_true) / zeta_test_true
        MAC_phi[caseN, mode_order] = MAC(phi_pred[caseN][:, mode_order], phi_test_true)

# plt.close('all')
fig, ax = plt.subplots(3, 1, layout="constrained")
for k in range(3):
    if k == 0:
        ax[k].plot(MAC_phi[:, 3], label='model')
        # ax[k].plot(MAC_pp, label='peak picking')
        ax[k].set_ylabel('MAC', fontsize=14)
        ax[k].set_title("Mode Shape MAC", fontsize=14)
        ax[k].legend(fontsize=14)
    elif k == 1:
        ax[k].plot(RE_freq[:, 3])
        ax[k].set_ylabel('Relative Error', fontsize=14)
        ax[k].set_title("Frequency MSE", fontsize=14)
        ax[k].set_xlabel('Sample No.', fontsize=14)
    else:
        ax[k].plot(RE_zeta[:, 3])
        ax[k].set_ylabel('Relative Error', fontsize=14)
        ax[k].set_title("Damping ratio MSE", fontsize=14)
        ax[k].set_xlabel('Sample No.', fontsize=14)
    ax[k].grid()

statistics_phi = np.zeros([3, modeN])
statistics_freq = np.zeros([3, modeN])
statistics_zeta = np.zeros([3, modeN])

for j in range(modeN):
    # mode shape
    statistics_phi[0, j] = np.mean(MAC_phi[-N_test:, j])
    statistics_phi[1, j] = np.std(MAC_phi[-N_test:, j])
    statistics_phi[2, j] = np.min(MAC_phi[-N_test:, j])
    # frequency
    statistics_freq[0, j] = np.mean(RE_freq[-N_test:, j])
    statistics_freq[1, j] = np.std(RE_freq[-N_test:, j])
    statistics_freq[2, j] = np.max(abs(RE_freq[-N_test:, j]))
    # damping
    statistics_zeta[0, j] = np.mean(RE_zeta[-N_test:, j])
    statistics_zeta[1, j] = np.std(RE_zeta[-N_test:, j])
    statistics_zeta[2, j] = np.max(abs(RE_zeta[-N_test:, j]))

statistics_phi = np.transpose(statistics_phi)
statistics_zeta = np.transpose(statistics_zeta) * 100
statistics_freq = np.transpose(statistics_freq) * 100
    
print("Mode shape")
print(statistics_phi)
print("Damping")
print(statistics_zeta)
print("Freq")
print(statistics_freq)


# mode shape MAC results of testing set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(np.zeros([N_test, 1])+1, MAC_phi[-N_test:, 0], color='#FF1F5B', label='mode 1', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+2, MAC_phi[-N_test:, 1], color='#00CD6C', label='mode 2', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+3, MAC_phi[-N_test:, 2], color='#FFC61E', label='mode 3', s=marker_size, alpha=1)
plt.scatter(np.zeros([N_test, 1])+4, MAC_phi[-N_test:, 3], color='#AF58BA', label='mode 4', s=marker_size, alpha=1)
plt.boxplot(MAC_phi[-N_test:, :], 0, '')
plt.xticks([1, 2, 3, 4], ['Mode1', 'Mode2', 'Mode3', 'Mode4'])
plt.ylim([0, 1.05])
plt.xticks(fontsize=17, fontname='Times New Roman')
plt.yticks(fontsize=17, fontname='Times New Roman')
plt.ylabel('MAC', fontname='Times New Roman', fontsize=17)
plt.grid()
plt.tight_layout()
plt.show()


# frequency results of testing set
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(freq_true[-N_test:, 0], freq_pred[-N_test:, 0], color='#FF1F5B', label='mode 1', s=marker_size)
plt.scatter(freq_true[-N_test:, 1], freq_pred[-N_test:, 1], color='#00CD6C', label='mode 2', s=marker_size)
plt.scatter(freq_true[-N_test:, 2], freq_pred[-N_test:, 2], color='#FFC61E', label='mode 3', s=marker_size)
plt.scatter(freq_true[-N_test:, 3], freq_pred[-N_test:, 3], color='#AF58BA', label='mode 4', s=marker_size)
plt.plot([0,30], [0,30], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,30], [0,30*0.9], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,30*0.9], [0,30], linestyle='--', color='blue')
plt.xlim([0,25])
plt.ylim([0,25])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Frequency (Hz)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()

zeta_limit = 0.011
# damping ratio results of testing set %
plt.figure(figsize=(4*2.54/2.54, 4*2.54/2.54))
plt.scatter(zeta_true[-N_test:, 0]*100, zeta_pred[-N_test:, 0]*100, color='#FF1F5B', label='mode 1', s=marker_size)
plt.scatter(zeta_true[-N_test:, 1]*100, zeta_pred[-N_test:, 1]*100, color='#00CD6C', label='mode 2', s=marker_size)
plt.scatter(zeta_true[-N_test:, 2]*100, zeta_pred[-N_test:, 2]*100, color='#FFC61E', label='mode 3', s=marker_size)
plt.scatter(zeta_true[-N_test:, 3]*100, zeta_pred[-N_test:, 3]*100, color='#AF58BA', label='mode 4', s=marker_size)
plt.plot([0,zeta_limit*100], [0,zeta_limit*100], linestyle='--', color='black', label='\u00B1 0%')
plt.plot([0,zeta_limit*100], [0,zeta_limit*0.9*100], linestyle='--', color='blue', label='\u00B1 10%')
plt.plot([0,zeta_limit*0.9*100], [0,zeta_limit*100], linestyle='--', color='blue')
plt.xlim([0.003*100,zeta_limit*100])
plt.ylim([0.003*100,zeta_limit*100])
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, handlelength=1, borderpad=0.1, labelspacing=0.1)
plt.xticks(fontsize=18, fontname='Times New Roman')
plt.yticks(fontsize=18, fontname='Times New Roman')
plt.xlabel('True Damping Ratio (%)', fontname='Times New Roman', fontsize=18)
plt.ylabel('Identified Damping Ratio (%)', fontname='Times New Roman', fontsize=18)
plt.grid()
plt.tight_layout()
plt.show()
# %% visualize trusses in a population
magnify = 3
marker_size = 5
plt.close('all')
fig, axs = plt.subplots(4, 2, figsize=(9, 4), layout="constrained")
for mode_order, ax in enumerate(axs.flat):
    caseN = mode_order * 60
    test_data = Dataset(graph_ids = [caseN])[0]
    graph_test = test_data[0].cpu()
    freq_test_true = test_data[1].cpu()
    zeta_test_true = test_data[2].cpu()
    node_test = node[caseN]
    element_test = element[caseN] - 1
    
    # Plot mode shapes
    for ele in element_test:
        node1 = node_test[ele[0]]
        node2 = node_test[ele[1]]
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], color='#009ADE')
    ax.plot(node_test[:, 0], node_test[:, 1], 'o', markersize=marker_size, label='undeformed', color='#009ADE')
        
    plt.setp(ax.get_xticklabels(), fontsize=10, fontname='Times New Roman')
    plt.setp(ax.get_yticklabels(), fontsize=10, fontname='Times New Roman')
    
    ax.set_xlim(-42,42)
    ax.set_ylim(-3,7)
    ax.set_xlabel('X (m)', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel('Y (m)', fontsize=12, fontname='Times New Roman')
    # title_text = "Mode {:.0f}, MAC={:.5f}".format(mode_order+1, MAC_phi)
    # ax.set_title(title_text, fontsize=12, fontname='Times New Roman')   
    # ax.set_aspect('equal')
    ax.grid()