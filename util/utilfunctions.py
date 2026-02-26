import pickle
import torch
import json
import io
import pickle

class Config(object):
  """Reads a JSON config file and sets each key-value pair as an attribute."""
  def __init__(self, config_file):
      self._readConfigFile(config_file)
  def _readConfigFile(self, file):
      try:
          with open(file, 'r') as cfg:
              data = json.load(cfg)
          for key, value in data.items():
              setattr(self, key, value)
      except:
          print('Configuration file read error')
          raise
  
class CPU_Unpickler(pickle.Unpickler):
    """Unpickler that forces tensors onto CPU regardless of where they were saved."""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
