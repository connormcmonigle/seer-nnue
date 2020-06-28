import os
import tqdm
import chess
import chess.engine
import random
import torch
import torch.nn.functional as F
import numpy as np
import util


def self_play_game(engine, config):
  white = []
  black = []
  scores = []
  bd = chess.Board()
  while (bd.fullmove_number < config.max_fullmoves) and (not bd.is_game_over()):
    info = engine.analyse(bd, chess.engine.Limit(depth=config.uci_engine_depth))
    w_tensor, b_tensor = util.to_tensors(bd)
    white.append(w_tensor)
    black.append(b_tensor)
    scores.append(torch.tensor(float(info['score'].relative.score(mate_score=10000))))
    bd.push(info['pv'][0])
  
  white = torch.stack(white, dim=0)
  black = torch.stack(black, dim=0)
  scores = torch.stack(scores, dim=0).float()
  outcome = 1.0 if bd.result() == '1-0' else (0.0 if bd.result() == '0-1' else 0.5)
  mask = torch.zeros(bd.fullmove_number+1, 2, dtype=torch.bool)
  mask[:, 0] = True
  mask = mask.flatten()[:len(scores)]
  outcome = mask * outcome + mask.logical_not() * (1.0 - outcome)
  pov = mask
  return pov, white, black, outcome, scores


def self_play_position_generator(num_positions, config):
  engine = chess.engine.SimpleEngine.popen_uci(config.uci_engine_path)
  engine.configure({"Threads": config.uci_engine_threads})
  count = 0
  while count < num_positions:
    game = self_play_game(engine, config)
    for elem in zip(*game):
      if count >= num_positions:
        break
      else:
        count += 1
        yield elem
  engine.quit()

def generate_games(config):
  num_positions = config.num_positions
  tgt_dir = config.dataset_path
  mm_pov, mm_white, mm_black, mm_outcome, mm_score = util.get_memmap_handlers(mode='w+', config=config)

  for (idx, pos) in tqdm.tqdm(enumerate(self_play_position_generator(num_positions, config)), total=num_positions):
    pov, white, black, outcome, score = pos
    mm_pov[idx] = pov.bool().numpy()
    mm_white[idx] = white.bool().numpy()
    mm_black[idx] = black.bool().numpy()
    mm_outcome[idx] = outcome.float().numpy()
    mm_score[idx] = score.float().numpy()

