import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceClassificationLayer(nn.Module): 
    def __init__(self, hidden_dim: int, proj_dim: int, num_classes: int, **kwargs):
        super(SequenceClassificationLayer, self).__init__(**kwargs)
        self.feature_proj = nn.Linear(hidden_dim, proj_dim)
        self.attention_proj = nn.Linear(hidden_dim, proj_dim)
        self.cls_proj = nn.Linear(proj_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        features = self.feature_proj(inputs)
        attention = self.attention_proj(inputs)
        attention = F.softmax(attention, dim=-2)
        seq_repr = torch.sum(attention * features, dim=-2)
        logits = self.cls_proj(seq_repr)
        return logits

class TokenClassificationLayer(nn.Module): 
    def __init__(self, hidden_dim: int, num_classes: int, **kwargs):
        super(TokenClassificationLayer, self).__init__(**kwargs)
        self.cls_layer = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        logits = self.cls_layer(inputs)
        return logits