import torch
import torch.nn as nn

class Mixer(nn.Module): 
    
    def __init__(self, num_mixers: int, max_seq_len: int, hidden_dim: int, mlp_hidden_dim: int, **kwargs):
        super(Mixer, self).__init__(**kwargs)
        self.mixers = nn.Sequential(*[
            MixerLayer(max_seq_len, hidden_dim, mlp_hidden_dim, mlp_hidden_dim) for _ in range(num_mixers)
        ])
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        return self.mixers(inputs)

class MixerLayer(nn.Module): 

    def __init__(self, max_seq_len: int, hidden_dim: int, channel_hidden_dim: int, seq_hidden_dim: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = MlpLayer(max_seq_len, seq_hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp_2 = MlpLayer(hidden_dim, channel_hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        residual = inputs
        outputs = self.layer_norm_1(inputs)
        outputs = outputs.transpose(-1, -2)
        outputs = self.mlp_1(outputs)
        outputs = outputs.transpose(-1, -2) + residual
        residual = outputs
        outputs = self.layer_norm_2(outputs)
        outputs = self.mlp_2(outputs) + residual
        return outputs

class MlpLayer(nn.Module): 

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim), 
            nn.GELU(), 
            nn.Linear(intermediate_dim, hidden_dim)
        ])
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        return self.layers(inputs)
