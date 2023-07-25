import scipy.io as sio
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


class BurgersDataset:
    def __init__(self, data_path, raw_resolution=[1024, 201], 
                 sample_resolution=[256, 101], eval_resolution=[1024, 201], 
                 v=0.1, start_x=-1, end_x=1, t=2,
                 train_batchsize=256, eval_batchsize=128, 
                 train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, 
                 subset=False, subset_ratio=0.2):
        all_data = self.load_hdf5_data(data_path=data_path)
        
        if subset:
            data_size = all_data.shape[0] * subset_ratio
        else:
            data_size = all_data.shape[0]
        
        train_idx = int(data_size * train_ratio)
        valid_idx = int(data_size * (train_ratio + valid_ratio))
        test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
        
        self.train_dataset = BurgersBase(
            all_data[:train_idx], raw_resolution, sample_resolution,
            start_x, end_x, t, v, mode='train'
        )
        self.valid_dataset = BurgersBase(
            all_data[train_idx:valid_idx], raw_resolution, eval_resolution,
            start_x, end_x, t, v, mode='valid'
        )
        self.test_dataset = BurgersBase(
            all_data[valid_idx:test_idx], raw_resolution, eval_resolution,
            start_x, end_x, t, v, mode='test'
        )
        
        del all_data
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=train_batchsize, shuffle=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batchsize, shuffle=False)
    
    def load_hdf5_data(self, data_path):
        with h5py.File(data_path, 'r') as f:
            all_data = np.array(f['tensor'], dtype=np.float32)
        
        return all_data
    
    def load_data(self):
        all_data = sio.loadmat(self.data_path)
        x = all_data['input'][:, ::self.x_sample_factor]
        y = all_data['output'][:, ::self.t_sample_factor, ::self.x_sample_factor]
        v = all_data['visc']
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.v = torch.from_numpy(v).float().item()


class BurgersBase(Dataset):
    def __init__(self, data, raw_resolution=[1024, 201], sample_resolution=[256, 101], 
                 start_x=-1, end_x=1, t=2, v=0.1, mode='train', **kwargs):
        super().__init__()
        self.x_sample_factor = raw_resolution[0] // sample_resolution[0]
        self.t_sample_factor = (raw_resolution[1] - 1) // (sample_resolution[1] - 1)
        self.num_grid_x = sample_resolution[0]
        self.num_grid_t = sample_resolution[1]
        self.start_x = start_x
        self.end_x = end_x
        self.t = t
        self.v = float(v)
        self.dt = t / (self.num_grid_t - 1)
        self.dx = (end_x - start_x) / self.num_grid_x
        self.mode = mode
        
        self.x, self.y = self.process(data, self.x_sample_factor, self.t_sample_factor, 
                                      self.num_grid_x, self.num_grid_t)
        
        # if mode == "train":
        #     self.pde_x, pde_y  = self.process(data, 1, 1, raw_resolution[0], raw_resolution[1])
        
    def process(self, data, x_sample_factor, t_sample_factor, num_grid_x, num_grid_t):
        x = data[:, 0, ::x_sample_factor]
        y = data[:, ::t_sample_factor, ::x_sample_factor]
    
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        if self.mode == "train":
            grid_x = torch.tensor(np.linspace(self.start_x, self.end_x, num_grid_x + 1)[:-1], dtype=torch.float32)  # (X)
            grid_t = torch.tensor(np.linspace(0, self.t, num_grid_t), dtype=torch.float32)  # (T)
        else:
            grid_x = torch.tensor(np.linspace(self.start_x, self.end_x, num_grid_x), dtype=torch.float32)  # (X)
            grid_t = torch.tensor(np.linspace(0, self.t, num_grid_t + 1)[1:], dtype=torch.float32)  # (T) 
        grid_x = grid_x.reshape(1, 1, num_grid_x)  # (1, 1, X)
        grid_t = grid_t.reshape(1, num_grid_t, 1)  # (1, T, 1)
        
        x = x.reshape(x.shape[0], 1, num_grid_x)   # (N, 1, X)
        x = x.repeat([1, num_grid_t, 1])   # (N, T, X)
        grid_x = grid_x.repeat([x.shape[0], num_grid_t, 1])  # (N, T, X)
        grid_t = grid_t.repeat([x.shape[0], 1, num_grid_x])  # (N, T, X)
        x = torch.stack([x, grid_x, grid_t], dim=3) # (N, T, X, 3)
        
        return x, y
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        # if self.mode == "train":
        #     return self.x[idx], self.y[idx], self.pde_x[idx]
        # else:
        #     return self.x[idx], self.y[idx]
        return self.x[idx], self.y[idx]
