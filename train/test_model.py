from os import path
import torch

import config as C
import util
import model

config = C.Config('config.yaml')

M = model.NNUE().to(config.device)

if (path.exists(config.model_save_path)):
  print('Loading model ... ')
  M.load_state_dict(torch.load(config.model_save_path))

M.cpu()

val = M(torch.ones(1), torch.zeros(1, *util.side_size()), torch.zeros(1, *util.side_size()))

print(val)
