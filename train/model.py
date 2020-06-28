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
    w_256 = self.white_affine(util.half_kp(white, black))
    b_256 = self.black_affine(util.half_kp(black, white))
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


def loss_fn(outcome, score, pred, lambda_):
  q = pred
  t = outcome
  p = util.cp_conversion(score)
  #print(t.size())
  #print(p.size())
  #print(pred.size())
  teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
  outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
  result = lambda_ * teacher_loss + (1.0 - lambda_) * outcome_loss
  #print(result.size())
  return result.sum()
