import os
import math
import bitstring
import struct
import random
import torch
import chess
import torch.nn.functional as F
import numpy as np
import util

PACKED_SFEN_VALUE_BITS = 40*8

# from sfen_packer.cpp
HUFFMAN_MAP = {0b0001 : chess.PAWN, 0b0011 : chess.KNIGHT, 0b0101 : chess.BISHOP, 0b0111 : chess.ROOK, 0b1001: chess.QUEEN}

# Grr. This is bad...
def read_bit(string, idx):
  corrected_idx = 8 * (idx // 8) + 7 - (idx % 8)
  return string[corrected_idx]


def read_n_bit(string, start_idx, n):
  result = 0
  for i in range(n):
    result |= (read_bit(string, start_idx+i) << i)
  return result

def is_quiet(board, from_, to_):
  for mv in board.legal_moves:
    if mv.from_square == from_ and mv.to_square == to_:
      return not board.is_capture(mv)
  return False
        

class NNUEBinData(torch.utils.data.Dataset):
  def __init__(self, config):
    super(NNUEBinData, self).__init__()
    self.config = config
    self.device = config.device
    self.batch_size = config.batch_size
    if not config.copy_data_to_mem:
      self.bits = bitstring.Bits(open(config.nnue_bin_data_path, 'rb'))
    else:
      self.bits = bitstring.Bits(bitstring.Bits(open(config.nnue_bin_data_path, 'rb')).bytes)

  def __len__(self):
    return len(self.bits) // PACKED_SFEN_VALUE_BITS

  def sample_raw(self, idx=None):
    element = random.randint(0, len(self) - 1) if (idx is None) else (len(self) - idx - 1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^~~~~~~~~
    #games are stored backwards relative to the direction they are read by the bitstring module
    start_idx = PACKED_SFEN_VALUE_BITS * element
    end_idx = start_idx + PACKED_SFEN_VALUE_BITS
    return self.bits[start_idx:end_idx]
    
  def sample_data(self, idx=None):
    segment = self.sample_raw(idx)

    board_string = segment[0:256]
    score_string = segment[256:272]
    move_string = segment[272:288]
    game_ply_string = segment[288:304]
    game_result_string = segment[304:312]
    padding_string = segment[312:320]

    bd = chess.Board(fen=None)
    bd.turn = not read_bit(board_string, 0)
    white_king_sq = read_n_bit(board_string, 1, 6)
    black_king_sq = read_n_bit(board_string, 7, 6)
    bd.set_piece_at(white_king_sq, chess.Piece(chess.KING, chess.WHITE))
    bd.set_piece_at(black_king_sq, chess.Piece(chess.KING, chess.BLACK))
    
    assert(black_king_sq != white_king_sq)
    
    bit_idx = 13
    
    for rank_ in range(8)[::-1]:
      for file_ in range(8):
        i = chess.square(file_, rank_)
        if white_king_sq == i or black_king_sq == i:
          continue
        if read_bit(board_string, bit_idx):
          assert(bd.piece_at(i) == None)
          piece_index = read_n_bit(board_string, bit_idx, 4)
          piece = HUFFMAN_MAP[piece_index]
          color = read_bit(board_string, bit_idx+4)
          bd.set_piece_at(i, chess.Piece(piece, not color))
          bit_idx += 5
        else:
          bit_idx += 1
    
    ply = struct.unpack('H', game_ply_string.bytes)[0]
    score = struct.unpack('h', score_string.bytes)[0]
    move = struct.unpack('H', move_string.bytes)[0]
    to_ = move & 63
    from_ = (move & (63 << 6)) >> 6
    
    if self.config.only_quiet:
      if not is_quiet(bd, from_, to_):
        return self.sample_data()
    
    move = chess.Move(from_square=chess.Square(from_), to_square=chess.Square(to_))
    
    # 1, 0, -1
    outcome = {'00000001': 1.0, '00000000': 0.5, '11111111': 0.0}[game_result_string.bin]
    assert(padding_string.bin == '00000000')
    return bd, move, outcome, score
    
  def sample(self, idx=None):
    bd, _, outcome, score = self.sample_data(idx)
    turn_before = bd.turn
    mirror = random.choice([False, True])
    if mirror:
      bd = bd.mirror()
    pov = torch.tensor([bd.turn])
    outcome = torch.tensor([outcome])
    score = torch.tensor([score])
    white, black = util.to_tensors(bd)
    return pov.float(), white.float(), black.float(), outcome.float(), score.float()
  
  def __getitem__(self, idx):
    return self.sample(idx)


