import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm


# # define the data, 2-D sphere

# In[3]:


sample_size = 10000
data_dim = 2
train_data = torch.randn(sample_size,data_dim)
train_data /= torch.norm(train_data,dim=1).reshape(-1,1)


# # visualization (if data_dim=2)

# In[4]:


plt.scatter(train_data[:,0],train_data[:,1])


# # define the time-dependent MLP

# In[5]:


class Net(nn.Module):
    
    def __init__(self, input_dim, output_dim, conditon_dim):
        super(Net, self).__init__()
        # First fully connected layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = conditon_dim
        self.fc1 = nn.Linear(input_dim + conditon_dim, 10000)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(10000, 10000)
        self.fc3 = nn.Linear(10000, output_dim)
        
    def forward(self, x, t):
        inp = torch.cat((x, t), dim=1)
        out = self.fc1(inp)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)

        return out


# # define the denoising loss

# In[6]:


def denoising_loss(net, batch_data, eps = 1e-5):
    batch_len = len(batch_data)
    noisy_levels = (torch.rand(batch_len).view(-1,1).cuda()) * (1 - 2 * eps) + eps # 0.001 ~ 1
    noise = torch.randn(batch_data.shape).cuda()
    noisy_data = batch_data * torch.sqrt(1 - noisy_levels) + torch.sqrt(noisy_levels) * noise
    predicted_noise = net(noisy_data, noisy_levels)
    loss  = torch.mean(torch.sum((predicted_noise - noise)**2, dim = 1)) 
    return loss


# In[7]:


def index_iterator(data_len, batch_size, shuffle=True):
    if shuffle:
        indices = np.random.permutation(data_len)
    else:
        indices = np.arange(data_len)
    for i in range(0, data_len, batch_size):
        yield indices[i:i+batch_size]


# # Generation function

# In[8]:


def ddim_sampling(model, sample_size = 100, total_steps = 1000, eps = 1e-5, eta = 1.):
    # generation
    now_coeff = 1 - eps
    interp = now_coeff / total_steps
    #sample_points = np.sqrt(1-now_coeff) * train_data[:100,:] + np.sqrt(now_coeff) * torch.randn(100,2).cuda()
    sample_points = torch.randn(sample_size,2).cuda()

    for _ in range(total_steps-1):
        #nl = eta * np.sqrt((now_coeff - interp)/now_coeff) * np.sqrt(1-(1-now_coeff)/(1-now_coeff+interp))
        nl = eta * np.sqrt(now_coeff - interp) 
        with torch.no_grad():
            direction = model(sample_points, torch.zeros(sample_size,1).cuda() + now_coeff)
        sample_points = np.sqrt(1-now_coeff+interp) * (sample_points - np.sqrt(now_coeff) * direction) / np.sqrt(1-now_coeff) + np.sqrt(max([now_coeff - interp - nl**2, 0])) * direction + nl * torch.randn(sample_size,2).cuda()
        now_coeff -= interp

    return sample_points.cpu().numpy()


# # training with EMA

# In[9]:


EPOCH = 1000

lr=1e-4
batch_size = 32

model = Net(data_dim, data_dim, 1).cuda()
EMA_model = Net(data_dim, data_dim, 1).cuda()
EMA_model.load_state_dict(deepcopy(model.state_dict()))
train_data = train_data.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

progress = tqdm(range(EPOCH))
for epoch in progress:
    avg_loss = 0
    totals = 0
    for batch_idx in index_iterator(len(train_data), batch_size):
        optimizer.zero_grad()
        loss  = denoising_loss(model, train_data[batch_idx])
        loss.backward()
        optimizer.step()
        totals += len(batch_idx)
        avg_loss += loss.item() * len(batch_idx)
    
    for p, ema_p in zip(model.parameters(), EMA_model.parameters()):
        ema_p.data.mul_(0.99).add_(0.01, p.data)

    progress.set_postfix({"avg_loss":avg_loss/totals,"last_loss":loss.item()})

    if (epoch + 1) % 50 == 0:
        ema_sample_points = ddim_sampling(EMA_model, 1000, eta = 1)
        sample_points = ddim_sampling(model, 1000, eta = 1)
        ema_sample_points_eta0 = ddim_sampling(EMA_model, 1000, eta = 0)
        sample_points_eta0 = ddim_sampling(model, 1000, eta = 0)
        plt.figure(figsize=(5,5))
        plt.scatter(sample_points[:,0], sample_points[:,1], s=5, alpha=0.6, label="a")
        plt.scatter(ema_sample_points[:,0], ema_sample_points[:,1], s=5,alpha=0.6, label="b")
        plt.scatter(sample_points_eta0[:,0], sample_points_eta0[:,1], s=5,alpha=0.6, label="c")
        plt.scatter(ema_sample_points_eta0[:,0], ema_sample_points_eta0[:,1], s=5,alpha=0.6, label="d")
        plt.xlim(-2,2)
        plt.ylim(-2,2)
        plt.legend()
        plt.title("epoch: {}".format(epoch))
        plt.savefig("temp_results/ddim_{}.png".format(epoch))
        plt.show()

