import os
import math
import cv2
import functools
import chess
import chess.pgn
import torch
import torch.nn.functional as F
import numpy as np
import random

def side_size():
  return (6, 8, 8)

def side_numel():
  return functools.reduce(lambda a, b: a*b, side_size())

def state_size():
  return (12, 8, 8)

def state_numel():
  return functools.reduce(lambda a, b: a*b, state_size())

def half_kp_numel():
  return 64 * side_numel()

def king_idx():
  return 5


def half_kp(side_batch):
  our_king = side_batch[:, king_idx(), :, :]
  return torch.einsum('bij,bcmn->bcijmn', (our_king, side_batch)).flatten(start_dim=1)


def side_to_tensor(bd, color):
  tensor = torch.zeros(side_size(), dtype=torch.bool)
  for piece_type in range(0, 6):
    plane = torch.zeros(64)
    for sq in bd.pieces(piece_type+1, color):
      plane[sq] = True;
    tensor[piece_type] = plane.reshape(8, 8)
  return tensor.flip(dims=(-1, ))


def to_tensors(bd):
  white = side_to_tensor(bd, color=chess.WHITE)
  black = side_to_tensor(bd, color=chess.BLACK)
  return white, black


def half_kp_indices(bd):
  w, b = to_tensors(bd)
  w = half_kp(w.unsqueeze(0)).squeeze(0).nonzero().squeeze(0)
  b = half_kp(b.unsqueeze(0)).squeeze(0).nonzero().squeeze(0)
  return w, b


def get_memmap_handlers(mode, config):
  num_positions = config.num_positions
  tgt_dir = config.dataset_path
  mm_pov = np.memmap(os.path.join(tgt_dir, 'pov.mm'), dtype='bool', mode=mode, shape=(num_positions))
  mm_white = np.memmap(os.path.join(tgt_dir, 'white.mm'), dtype='bool', mode=mode, shape=(num_positions, *side_size()))
  mm_black = np.memmap(os.path.join(tgt_dir, 'black.mm'), dtype='bool', mode=mode, shape=(num_positions, *side_size()))
  mm_score = np.memmap(os.path.join(tgt_dir, 'score.mm'), dtype='float32', mode=mode, shape=(num_positions))
  return mm_pov, mm_white, mm_black, mm_score

