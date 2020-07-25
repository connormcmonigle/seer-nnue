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

HUFFMAN_MAP = {0b0001 : chess.PAWN, 0b0011 : chess.KNIGHT, 0b0101 : chess.BISHOP, 0b0111 : chess.ROOK, 0b1001: chess.QUEEN}

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


def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset
  per_worker = dataset.cardinality() // worker_info.num_workers
  start = worker_id * per_worker
  dataset.set_range(start, start + per_worker)


class NNUEBinData(torch.utils.data.IterableDataset):
  def __init__(self, config):
    super(NNUEBinData, self).__init__()
    self.config = config
    self.batch_size = config.batch_size
    self.shuffle_buffer = [None] * config.shuffle_buffer_size
    self.bits = bitstring.Bits(open(config.nnue_bin_data_path, 'rb'))
    self.start_idx = 0
    self.end_idx = len(self.bits) // PACKED_SFEN_VALUE_BITS

  def cardinality(self):
    return len(self.bits) // PACKED_SFEN_VALUE_BITS

  def set_range(self, start_idx, end_idx):
    self.start_idx = start_idx
    self.end_idx = end_idx

  def get_raw(self, idx):
    first_idx = PACKED_SFEN_VALUE_BITS * idx
    last_idx = first_idx + PACKED_SFEN_VALUE_BITS
    return self.bits[first_idx:last_idx]

  def get_data(self, idx):
    segment = self.get_raw(idx)

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
    bd.fullmove_number = ply // 2
    
    score = struct.unpack('h', score_string.bytes)[0]
    move = struct.unpack('H', move_string.bytes)[0]
    to_ = move & 63
    from_ = (move & (63 << 6)) >> 6
    
    if self.config.only_quiet and (not is_quiet(bd, from_, to_)):
      next_idx = (idx + 1 - self.start_idx) % (self.end_idx - self.start_idx) + self.start_idx
      assert(next_idx >= self.start_idx and next_idx < self.end_idx)
      return self.get_data(next_idx)
    move = chess.Move(from_square=chess.Square(from_), to_square=chess.Square(to_))
    
    # 1, 0, -1
    outcome = {'00000001': 1.0, '00000000': 0.5, '11111111': 0.0}[game_result_string.bin]
    assert(padding_string.bin == '00000000')
    return bd, move, outcome, score

  def get(self, idx):
    bd, _, outcome, score = self.get_data(idx)
    turn_before = bd.turn
    
    if self.config.enable_mirroring:
      mirror = random.choice([False, True])
      if mirror:
        bd = bd.mirror()
    pov = torch.tensor([bd.turn])
    outcome = torch.tensor([outcome])
    score = torch.tensor([score])
    white, black = util.to_tensors(bd)
    return pov.float(), white.float(), black.float(), outcome.float(), score.float()

  def seq_data_iter(self):
    for idx in range(self.start_idx, self.end_idx):
      yield self.get_data(idx)

  def get_shuffled(self, idx):
    shuffle_buffer_idx = random.randrange(0, len(self.shuffle_buffer))
    result = self.shuffle_buffer[shuffle_buffer_idx]
    self.shuffle_buffer[shuffle_buffer_idx] = self.get(idx)
    return result

  def __iter__(self):
    for idx in range(self.start_idx, self.end_idx):
      val = self.get_shuffled(idx)
      if val != None:
        yield val
    for val in self.shuffle_buffer:
      if val != None:
        yield val

