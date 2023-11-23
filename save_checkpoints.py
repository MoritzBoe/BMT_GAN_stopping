import pytorch_lightning as pl
import numpy as np
from copy import deepcopy
from pathlib import Path
import torch

class CheckpointPercent(pl.Callback):
    """Save a new checkpoint only, when the metric improved by X percent.
    Works only for positive metric values!
    
    """

    def __init__(self, dirpath, filename, monitor, percent, mode="min",  save_top_k=-1, save_weights_only=False):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_top_k = save_top_k
        self.percent = percent
        mode_dict = {"min": np.Inf, "max": -np.Inf}
        self.saved_paths = []
        if mode not in mode_dict:
            raise ValueError(f"{mode=} not in {mode_dict=}")
        self.metric = mode_dict[self.mode]
        if mode == "min":
            self.scaling = 1-self.percent/100
        else:
            self.scaling = 1+self.percent/100
        
    def on_validation_end(self, trainer, pl_module):
        metrics = deepcopy(trainer.callback_metrics)
        
        metrics["epoch"] = metrics.get("epoch")
        metrics["epoch"] = metrics["epoch"].int() if isinstance(metrics["epoch"], torch.Tensor) else torch.tensor(trainer.current_epoch)
        
        if self.monitor not in metrics.keys():
            raise ValueError(f"{self.monitor=} not monitored in trainer.callback_metrics. This is the case if metric is not logged with self.log() in the model")
        if self.mode == "min" and (metrics[self.monitor] > self.metric*self.scaling):
            return
        if self.mode == "max" and (metrics[self.monitor] < self.metric*self.scaling):
            return
        filename = pl.callbacks.ModelCheckpoint._format_checkpoint_name(self.filename, metrics)
        filepath = Path(self.dirpath).joinpath(filename+".ckpt")
        trainer.save_checkpoint(filepath=filepath, weights_only=self.save_weights_only)
        self.saved_paths.append(filepath)
        if self.save_top_k != -1:
            if len(self.saved_paths) > self.save_top_k:
                # remove worst checkpoint (first in list)
                trainer.strategy.remove_checkpoint(self.saved_paths[0])
                self.saved_paths.pop(0)
        self.metric = metrics[self.monitor]