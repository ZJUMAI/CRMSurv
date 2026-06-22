import torch
import torch.nn as nn
from .kmeans.kmeans_pytorch import kmeans


class Network(nn.Module):
    def __init__(self,
                 cluster = 16,
                 input_dim=1024,
                 dim=256,
                 **kwargs):
        super().__init__()
        self.num_cluster = cluster
        self.seed = kwargs['seed']

    def forward(self, **kwargs):
        x_path = kwargs["x_path"]

        print('cluster center numbers',self.num_cluster) 

        _, bases = kmeans(X=x_path, num_clusters=self.num_cluster, distance='euclidean',
                          device=x_path.device, tqdm_flag=True, seed=self.seed)

        return bases