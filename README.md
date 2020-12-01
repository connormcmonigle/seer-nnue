# Seer

Seer is a UCI chess engine using a neural network for position evaluation. Seer's neural network architecture takes inspiration from the halfkp NNUE architecture in that its input features include the outer product of the king position and all other piece positions. Additionally, the network's first layer is incrementally updated as in NNUE. As an engine, Seer stands on the shoulders of many great (and mostly much stronger) predecessors. 
In particular, [Ethereal](https://github.com/AndyGrant/Ethereal), [Winter](https://github.com/rosenthj/Winter), and [Stockfish](https://github.com/official-stockfish/Stockfish) have proven useful in the development of Seer.

### UCI Options
- Clear Hash
- Threads (for every thread doubling, a gain of about 70-80 elo can be expected)
- Hash
- Weights (The absolute path to a binary weights file. This option must be set.)

### Features
- Completely from scratch neural network training and execution (using OpenMP SIMD directives) implementation 
  (training scripts use PyTorch for GPU acceleration and can be found in the /train subdirectory)
- Plain magic bitboard move generation with constexpr compile time generated attack tables.
- Principal variation search inside an iterative deepening framework
- Lockless shared transposition table (board state Zobrist hash is incrementally updated)
- Move Ordering (SEE for captures + Killer Move, Combined Butterfly History, Counter Move History and Follow Up History for quiets)
- History pruning as well as SEE pruning in QSearch
- Null move pruning
- Static null move pruning (reverse futility pruning)
- Late move reduction
- Aspiration windows

### Compiling

```
cd build
cmake ..
make seer
```
