from classification import SequenceClassificationLayer, TokenClassificationLayer
from mixer import Mixer
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict
import torch
import torch.nn as nn

class PnlpMixerSeqCls(nn.Module): 
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        seq_cls_cfg: DictConfig, 
        **kwargs
    ):
        super(PnlpMixerSeqCls, self).__init__(**kwargs)
        self.pnlp_mixer = PnlpMixer(bottleneck_cfg, mixer_cfg)
        self.seq_cls = SequenceClassificationLayer(**seq_cls_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        reprs = self.pnlp_mixer(inputs)
        seq_logits = self.seq_cls(reprs)
        return seq_logits

class PnlpMixerTokenCls(nn.Module): 
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        token_cls_cfg: DictConfig, 
        **kwargs
    ):
        super(PnlpMixerTokenCls, self).__init__(**kwargs)
        self.pnlp_mixer = PnlpMixer(bottleneck_cfg, mixer_cfg)
        self.token_cls = TokenClassificationLayer(**token_cls_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        reprs = self.pnlp_mixer(inputs)
        token_logits = self.token_cls(reprs)
        return token_logits

class PnlpMixer(nn.Module):
    def __init__(
        self,
        bottleneck_cfg: DictConfig,
        mixer_cfg: DictConfig,
        **kwargs
    ):
        super(PnlpMixer, self).__init__(**kwargs)
        self.bottleneck = nn.Linear((2 * bottleneck_cfg.window_size + 1) * bottleneck_cfg.feature_size, bottleneck_cfg.hidden_dim)
        self.mixer = Mixer(**mixer_cfg)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: 
        features = self.bottleneck(inputs)
        reprs = self.mixer(features)
        return reprs