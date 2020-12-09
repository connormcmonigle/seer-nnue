import util
import torchvision.transforms.functional as TF
from PIL import Image
from os import path
import torch
import chess

import config as C
import util
import model


def to_image(x):
  x = x.reshape(128, 64, 12, 8, 8)
  x = x.permute(0, 3, 1, 2, 4)
  x = (x.reshape(128*8, 64*12*8) * 6.0).sigmoid()
  im = TF.to_pil_image(x)
  return im




config = C.Config('config.yaml')

M = model.NNUE().to(config.device)

if (path.exists(config.model_save_path)):
  print('Loading model ... ')
  M.load_state_dict(torch.load(config.model_save_path, map_location=config.device))

num_parameters = sum(map(lambda x: torch.numel(x), M.parameters()))

print(num_parameters)

M.cpu()

w = M.white_affine.weight.data
b = M.black_affine.weight.data


to_image(w).save(path.join(config.visual_directory, 'white_affine.png'))
to_image(b).save(path.join(config.visual_directory, 'black_affine.png'))
