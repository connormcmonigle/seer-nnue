import os
import math
import tqdm
import chess.pgn
import random
import torch
import torch.nn.functional as F
import numpy as np
import util

class Data(torch.utils.data.Dataset):
  def __init__(self, config):
    super(Data, self).__init__()
    tgt_dir = config.dataset_path
    num_positions = config.num_positions
    self.config = config
    self.device = config.device
    self.batch_size = config.batch_size
    mm_pov, mm_white, mm_black, mm_score = util.get_memmap_handlers(mode='r', config=config)
    self.mm_pov = mm_pov
    self.mm_white = mm_white
    self.mm_black = mm_black
    self.mm_score = mm_score

  def __len__(self):
    return self.config.num_positions

  def sample(self):
    idx = random.randint(0, len(self)-1)
    vals = [np.array(self.mm_pov[idx]), np.copy(self.mm_white[idx]), np.copy(self.mm_black[idx]), np.array(self.mm_score[idx])]
    return tuple(map(lambda x: torch.from_numpy(x), vals))
    
  def sample_batch(self):
    pov_ls = []
    white_ls = []
    black_ls = []
    score_ls = []
    for _ in range(self.batch_size):
      pov, white, black, score = self.sample()
      pov_ls.append(pov)
      white_ls.append(white)
      black_ls.append(black)
      score_ls.append(score)
    return torch.stack(pov_ls, dim=0).to(self.device).float(),\
      torch.stack(white_ls, dim=0).to(self.device).float(),\
      torch.stack(black_ls, dim=0).to(self.device).float(),\
      torch.stack(score_ls, dim=0).to(self.device).float()
