import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


class KMFlowDataset:
    def __init__(self, data_path, raw_resolution=[256, 256, 513], sample_resolution=[64, 64, 257], 
                 pde_resolution = [256, 256, 513], eval_resolution=[64, 64, 257],
                 Re=500, start_x=0, end_x=1, start_y=0, end_y=1, t=1.0, split_factor=8,
                 train_batchsize=10, eval_batchsize=1, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, 
                 subset=False, subset_ratio=0.5):
        self.data_path = data_path
        self.x_sample_factor = raw_resolution[0] // sample_resolution[0]
        self.y_sample_factor = raw_resolution[1] // sample_resolution[1]
        self.t_sample_factor = (raw_resolution[2] - 1) // (sample_resolution[2] - 1)
        self.num_grid_x = sample_resolution[0]
        self.num_grid_y = sample_resolution[1]
        self.num_grid_t = sample_resolution[2]
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.t = t / split_factor
        self.split_factor = split_factor
        self.dt = self.t / (self.num_grid_t - 1)
        self.Re = float(Re)
        self.dx = (end_x - start_x) / self.num_grid_x
        self.dy = (end_y - start_y) / self.num_grid_y
        self.train_batchsize = train_batchsize
        self.eval_batchsize = eval_batchsize
        
        all_data = self.load_np_data()
        
        if subset:
            data_size = all_data.shape[0] * subset_ratio
        else:
            data_size = all_data.shape[0]
        
        train_idx = int(data_size * train_ratio)
        valid_idx = int(data_size * (train_ratio + valid_ratio))
        test_idx = int(data_size * (train_ratio + valid_ratio + test_ratio))
        
        self.train_dataset = KMFlowBase(all_data[:train_idx], raw_resolution, sample_resolution, 
                                        start_x, end_x, start_y, end_y, t, split_factor)
        self.valid_dataset = KMFlowBase(all_data[train_idx:valid_idx], raw_resolution, eval_resolution, 
                                        start_x, end_x, start_y, end_y, t, split_factor)
        self.test_dataset = KMFlowBase(all_data[valid_idx:test_idx], raw_resolution, eval_resolution, 
                                       start_x, end_x, start_y, end_y, t, split_factor)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batchsize, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.eval_batchsize, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.eval_batchsize, shuffle=False)
        
    def load_np_data(self):
        raw_data = np.load(self.data_path, mmap_mode='r')
        all_data = raw_data.copy()
        del raw_data
        
        if self.split_factor != 1.0:
            all_data = self.split_data(all_data)
        
        return all_data
    
    def split_data(self, data):
        N, T, S = data.shape[:3]
        new_data = np.zeros((N * self.split_factor, T // self.split_factor + 1, S, S))
        step = T // self.split_factor
        for i in range(N):
            for j in range(self.split_factor):
                new_data[i * self.split_factor + j] = data[i, j * step: (j + 1) * step + 1]
        
        return new_data
        
    def get_3d_grid(self):
        grid_x = torch.tensor(np.linspace(self.start_x, self.end_x, self.num_grid_x + 1)[:-1], dtype=torch.float32)
        grid_x = grid_x.reshape(1, self.num_grid_x, 1, 1, 1).repeat([1, 1, self.num_grid_y, self.num_grid_t, 1])
        grid_y = torch.tensor(np.linspace(self.start_y, self.end_y, self.num_grid_y + 1)[:-1], dtype=torch.float32)
        grid_y = grid_y.reshape(1, 1, self.num_grid_y, 1, 1).repeat([1, self.num_grid_x, 1, self.num_grid_t, 1])
        grid_t = torch.tensor(np.linspace(0, self.t, self.num_grid_t), dtype=torch.float32)
        grid_t = grid_t.reshape(1, 1, 1, self.num_grid_t, 1).repeat([1, self.num_grid_x, self.num_grid_y, 1, 1])
        
        return grid_x, grid_y, grid_t
        
    def process_data(self, x: torch.Tensor, y: torch.Tensor, mode="train"):
        grid_x, grid_y, grid_t = self.get_3d_grid()
        x = x.reshape(x.shape[0], self.num_grid_x, self.num_grid_y, 1, 1)
        x = x.repeat([1, 1, 1, self.num_grid_t, 1])
        grid_x = grid_x.repeat([x.shape[0], 1, 1, 1, 1])
        grid_y = grid_y.repeat([x.shape[0], 1, 1, 1, 1])
        grid_t = grid_t.repeat([x.shape[0], 1, 1, 1, 1])
        x = torch.cat([x, grid_x, grid_y, grid_t], dim=-1)
        return TensorDataset(x, y)


class KMFlowBase(Dataset):
    def __init__(self, data, raw_resolution=[256, 256, 513], sample_resolution=[64, 64, 257],
                 start_x=0, end_x=1, start_y=0, end_y=1, t=1.0, split_factor=8, mode='train', **kwargs):
        super().__init__()
        self.x_sample_factor = raw_resolution[0] // sample_resolution[0]
        self.y_sample_factor = raw_resolution[1] // sample_resolution[1]
        self.t_sample_factor = (raw_resolution[2] - 1) // (sample_resolution[2] - 1)
        self.num_grid_x = sample_resolution[0]
        self.num_grid_y = sample_resolution[1]
        self.num_grid_t = sample_resolution[2] if split_factor == 1 else sample_resolution[2] // split_factor + 1
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.t = t / split_factor
        self.split_data = split_factor
        self.dt = self.t / (self.num_grid_t - 1)
        self.dx = (end_x - start_x) / self.num_grid_x
        self.dy = (end_y - start_y) / self.num_grid_y
        self.mode = mode
        
        self.process_data(data)
        
    def process_data(self, data):
        data = data[:, ::self.t_sample_factor, ::self.x_sample_factor, ::self.y_sample_factor]
        y = torch.from_numpy(data).float()
        y = y.permute(0, 2, 3, 1)
        x = y[:, :, :, :1, None]
        
        grid_x, grid_y, grid_t = self.get_3d_grid()
        x = x.reshape(x.shape[0], self.num_grid_x, self.num_grid_y, 1, 1)
        x = x.repeat([1, 1, 1, self.num_grid_t, 1])
        grid_x = grid_x.repeat([x.shape[0], 1, 1, 1, 1])
        grid_y = grid_y.repeat([x.shape[0], 1, 1, 1, 1])
        grid_t = grid_t.repeat([x.shape[0], 1, 1, 1, 1])
        x = torch.cat([x, grid_x, grid_y, grid_t], dim=-1)
        
        self.x = x
        self.y = y
    
    def get_3d_grid(self):
        grid_x = torch.tensor(np.linspace(self.start_x, self.end_x, self.num_grid_x + 1)[:-1], dtype=torch.float32)
        grid_x = grid_x.reshape(1, self.num_grid_x, 1, 1, 1).repeat([1, 1, self.num_grid_y, self.num_grid_t, 1])
        grid_y = torch.tensor(np.linspace(self.start_y, self.end_y, self.num_grid_y + 1)[:-1], dtype=torch.float32)
        grid_y = grid_y.reshape(1, 1, self.num_grid_y, 1, 1).repeat([1, self.num_grid_x, 1, self.num_grid_t, 1])
        grid_t = torch.tensor(np.linspace(0, self.t, self.num_grid_t), dtype=torch.float32)
        grid_t = grid_t.reshape(1, 1, 1, self.num_grid_t, 1).repeat([1, self.num_grid_x, self.num_grid_y, 1, 1])
        
        return grid_x, grid_y, grid_t

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]