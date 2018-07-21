"""
Multi-label classification specific callbacks.
"""
import numpy as np
import ipdb

from keras.callbacks import Callback

class HammingLoss(Callback):
    def __init__(self, datasets, batch_size=128):
        self.datasets = datasets
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.params['metrics'] += ['hl', 'val_hl']

    def on_epoch_begin(self, epoch, logs={}):
        self.metrics = {}
        for dname in self.datasets.keys():
            predicts = self.model.predict(self.datasets[dname], batch_size=self.batch_size)
            if isinstance(predicts,list):
                preds = {self.model.output_names[i]: k for i,k in enumerate(predicts)}
            else:
                preds = {'Y': predicts}
            hl = np.hstack([np.round(v) != self.datasets[dname][k] for k, v in preds.items()]).sum(axis=1).mean()
            self.metrics[dname] = hl

    def on_batch_begin(self, batch, logs={}):
        if 'train' in self.metrics:
            logs['hl'] = self.metrics['train']
        if 'valid' in self.metrics:
            logs['val_hl'] = self.metrics['valid']

    def on_epoch_end(self, epoch, logs={}):
        if 'train' in self.metrics:
            logs['hl'] = self.metrics['train']
        if 'valid' in self.metrics:
            logs['val_hl'] = self.metrics['valid']
