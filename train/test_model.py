from os import path
import torch
import chess

import config as C
import util
import model

config = C.Config('config.yaml')

M = model.NNUE().to(config.device)

if (path.exists(config.model_save_path)):
  print('Loading model ... ')
  M.load_state_dict(torch.load(config.model_save_path))

M.cpu()

while True:
  bd = chess.Board(input("fen: "))
  white, black = util.to_tensors(bd);
  val = M(torch.tensor([bd.turn]).float(), white.unsqueeze(0).float(), black.unsqueeze(0).float())
  print(val)
