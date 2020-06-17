import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    # DECLARATION ORDER MUST MATCH C++ IMPLEMENTATION
    self.white_affine = nn.Linear(util.half_kp_numel(), 256)
    self.black_affine = nn.Linear(util.half_kp_numel(), 256)
    self.fc0 = nn.Linear(512, 32)
    self.fc1 = nn.Linear(32, 32)
    self.fc2 = nn.Linear(32, 1)
    

    
  def forward(self, pov, white, black):
    w_256 = self.white_affine(util.half_kp(white))
    b_256 = self.black_affine(util.half_kp(black))
    pov = pov.unsqueeze(-1)
    #print(pov)
    x = pov * torch.cat([w_256, b_256], dim=1) + (1.0 - pov) * torch.cat([b_256, w_256], dim=1)
    x = F.relu(x)
    x = F.relu(self.fc0(x))
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def to_binary_file(self, path):
    joined = np.array([])
    for p in self.parameters():
      print(p.size())
      joined = np.concatenate((joined, p.data.cpu().t().flatten().numpy()))
    print(joined.shape)
    joined.astype('float32').tofile(path)


def loss_fn(score, pred):
  score = score.unsqueeze(-1)
  eta = 0.001
  c_score = score.clamp(eta, 1.0-eta)
  min_value = -(score * c_score.log() + (1.0-score) * (1.0 - c_score).log()).mean()
  return -(score * F.logsigmoid(pred) + (1.0-score) * F.logsigmoid(-pred)).mean() - min_value
