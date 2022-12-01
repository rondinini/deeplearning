import os
from abc import ABC, abstractmethod
import tensorflow as tf
from pathlib import Path
from logging import getLogger

_log = getLogger()

class FunctionalModel(ABC):

    def __init__(self, name, filepath):
        self.name = name
        self.filepath = filepath
    
    @abstractmethod
    def connect_layers(self):
        """ Overwritten by child class"""
        pass
    
    def _build_model(self, inputs, outputs):
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        _log.info(model.summary())
        if not os.path.exists(self.filepath):
            _log.info(f'Creating directory {self.filepath}')
            os.makedirs(self.filepath)
        summary_fn = Path(self.filepath) / Path(self.name + '_summary.txt')
        with open(summary_fn, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        graph_fn = Path(self.filepath) / Path(self.name + '_graph.png')
        tf.keras.utils.plot_model(
            model,
            to_file=graph_fn,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            expand_nested=True,
            show_layer_activations=True
        )
        return model


