from dataset import PnlpMixerDataModule
from model import PnlpMixerSeqCls, PnlpMixerTokenCls
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from typing import Any, Dict, List
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class PnlpMixerSeqClsTrainModule(pl.LightningModule): 
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(PnlpMixerSeqClsTrainModule, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = PnlpMixerSeqCls(
            model_cfg.bottleneck,
            model_cfg.mixer,
            model_cfg.sequence_cls,
        )

    def shared_step(self, batch): 
        inputs = batch['inputs']
        targets = batch['targets']
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        corr = torch.sum(logits.argmax(dim=-1) == targets)
        all = logits.size(0)
        return {
            'loss': loss, 
            'corr': corr, 
            'all': all
        }

    def compute_accuracy(self, outputs: List[Dict[str, Any]]): 
        corr = 0
        all = 0
        for output in outputs: 
            corr += output['corr']
            all += output['all']
        return {
            'acc': corr / all, 
        }
    
    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return results
    
    def training_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log('train_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('val_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results
    
    def validation_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log('val_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        results = self.shared_step(batch)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results
    
    def test_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log('test_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer

class PnlpMixerTokenClsTrainModule(pl.LightningModule):
    def __init__(self, optimizer_cfg: DictConfig, model_cfg: DictConfig, **kwargs):
        super(PnlpMixerTokenClsTrainModule, self).__init__(**kwargs)
        self.optimizer_cfg = optimizer_cfg
        self.model = PnlpMixerTokenCls(
            model_cfg.bottleneck,
            model_cfg.mixer,
            model_cfg.token_cls,
        )

    def common_step(self, batch): 
        inputs = batch['inputs']
        targets = batch['targets']
        logits = self.model(inputs)
        loss = F.cross_entropy(logits.transpose(-1, -2), targets, ignore_index=-1)
        corr = torch.sum(torch.logical_and(logits.argmax(dim=-1) == targets, targets > 0))
        all = torch.sum(targets > 0)
        return {
            'loss': loss, 
            'corr': corr, 
            'all': all, 
        }

    def compute_accuracy(self, outputs: List[Dict[str, Any]]): 
        corr = 0
        all = 0
        for output in outputs: 
            corr += output['corr']
            all += output['all']
        return {
            'acc': corr / all 
        }
    
    def training_step(self, batch, batch_idx):
        results = self.common_step(batch)
        self.log('train_loss', results['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return results
    
    def training_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log('train_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        results = self.common_step(batch)
        self.log('val_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results
    
    def validation_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log('val_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        results = self.common_step(batch)
        self.log('test_loss', results['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return results
    
    def test_epoch_end(self, outputs):
        accuracy = self.compute_accuracy(outputs)
        self.log('test_acc', accuracy['acc'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer_cfg = self.optimizer_cfg
        optimizer = torch.optim.Adam(self.parameters(), **optimizer_cfg)
        return optimizer

def parse_args(): 
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--cfg', type=str)
    args.add_argument('-n', '--name', type=str)
    args.add_argument('-p', '--ckpt', type=str)
    args.add_argument('-m', '--mode', type=str, default='train')
    return args.parse_args()

def get_module_cls(type: str): 
    if type == 'mtop': 
        return PnlpMixerTokenClsTrainModule
    if type == 'matis' or type == 'imdb': 
        return PnlpMixerSeqClsTrainModule

if __name__ == '__main__': 
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    vocab_cfg = cfg.vocab
    train_cfg = cfg.train
    model_cfg = cfg.model
    module_cls = get_module_cls(train_cfg.dataset_type)
    if args.ckpt: 
        train_module = module_cls.load_from_checkpoint(args.ckpt, optimizer_cfg=train_cfg.optimizer, model_cfg=model_cfg)
    else: 
        train_module = module_cls(train_cfg.optimizer, model_cfg)
    data_module = PnlpMixerDataModule(cfg.vocab, train_cfg, model_cfg.projection)
    trainer = pl.Trainer(
        # accelerator='ddp',
        # amp_backend='native', 
        # amp_level='O2', 
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_acc',
                save_last=True, 
                save_top_k=5, 
                mode='max'
            )
        ],
        checkpoint_callback=True, 
        gpus=-1,
        log_every_n_steps=train_cfg.log_interval_steps,
        logger=pl.loggers.TensorBoardLogger(train_cfg.tensorboard_path, args.name),
        max_epochs=train_cfg.epochs, 
    )
    if args.mode == 'train':
        trainer.fit(train_module, data_module)
    if args.mode == 'test': 
        trainer.test(train_module, data_module)
