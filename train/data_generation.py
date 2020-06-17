import os
import tqdm
import chess
import chess.engine
import random
import torch
import torch.nn.functional as F
import numpy as np
import util

def cp_conversion(x, alpha=0.005):
  return (x * alpha).sigmoid()


def self_play_game(config):
  white = []
  black = []
  scores = []
  engine = chess.engine.SimpleEngine.popen_uci(config.uci_engine_path)
  bd = chess.Board()
  while (bd.fullmove_number < config.max_fullmoves) and (not bd.is_game_over()):
    info = engine.analyse(bd, chess.engine.Limit(time=config.uci_engine_time))
    #print(info['pv'][0])
    w_tensor, b_tensor = util.to_tensors(bd)
    white.append(w_tensor)
    black.append(b_tensor)
    scores.append(torch.tensor(float(info['score'].relative.score(mate_score=10000))))
    bd.push(info['pv'][0])
  
  white = torch.stack(white, dim=0)
  black = torch.stack(black, dim=0)
  scores = cp_conversion(torch.stack(scores, dim=0))
  
  outcome = 1.0 if bd.result() == '1-0' else (0.0 if bd.result() == '0-1' else 0.5)
  mask = torch.zeros(bd.fullmove_number+1, 2, dtype=torch.bool)
  mask[:, 0] = True
  mask = mask.flatten()[:len(scores)]
  outcome = mask * outcome + mask.logical_not() * (1.0 - outcome)
  scores = config.interpolation_factor * outcome + (1.0 - config.interpolation_factor) * scores
  engine.quit()
  pov = mask
  return pov, white, black, scores


def self_play_position_generator(num_positions, config):
  count = 0
  while count < num_positions:
    game = self_play_game(config)
    for elem in zip(*game):
      if count >= num_positions:
        break
      else:
        count += 1
        yield elem


def generate_games(config):
  num_positions = config.num_positions
  tgt_dir = config.dataset_path
  mm_pov, mm_white, mm_black, mm_score = util.get_memmap_handlers(mode='w+', config=config)
  for (idx, pos) in tqdm.tqdm(enumerate(self_play_position_generator(num_positions, config)), total=num_positions):
    pov, white, black, score = pos
    mm_pov[idx] = pov.numpy()
    mm_white[idx] = white.numpy()
    mm_black[idx] = black.numpy()
    mm_score[idx] = score.numpy()

