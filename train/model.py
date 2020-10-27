import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import util


class NNUE(nn.Module):
  def __init__(self):
    super(NNUE, self).__init__()
    # DECLARATION ORDER MUST MATCH C++ IMPLEMENTATION
    BASE = 128
    self.white_affine = nn.Linear(util.half_kp_numel(), BASE)
    self.black_affine = nn.Linear(util.half_kp_numel(), BASE)
    self.fc0 = nn.Linear(2*BASE, 32)
    self.fc1 = nn.Linear(32, 32)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(96, 1)
    

  def forward(self, pov, white, black):
    w_ = self.white_affine(util.half_kp(white, black))
    b_ = self.black_affine(util.half_kp(black, white))
    base = F.relu(pov * torch.cat([w_, b_], dim=1) + (1.0 - pov) * torch.cat([b_, w_], dim=1))
    x = F.relu(self.fc0(base))
    x = torch.cat([x, F.relu(self.fc1(x))], dim=1)
    x = torch.cat([x, F.relu(self.fc2(x))], dim=1)
    x = self.fc3(x)
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
  epsilon = 1e-12
  teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
  outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
  
  teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
  outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))
  result  = lambda_ * teacher_loss    + (1.0 - lambda_) * outcome_loss
  entropy = lambda_ * teacher_entropy + (1.0 - lambda_) * outcome_entropy 
  #print(result.size())
  return result.mean() - entropy.mean()
