<p align="center">
  <img src="logo/logo_1.png" height="100" width="200">
</p>
<h1 align="center">Seer</h1>

Seer is an original, strong UCI chess engine. Seer relies on a neural network estimating WDL probabilities for position evaluation. Seer's network is trained through a novel retrograde learning approach starting only from 6-man syzygy EGTB WDL values. These initial WDL scores are then iteratively backed up to 32-man chess positions using Seer's search to find continuations from N-man chess positions to N-1-man chess positions ([implementation](https://github.com/connormcmonigle/seer-training)). Seer uses a conventional alpha-beta search combined with "Lazy SMP" (shared transposition table) for multithreading support.

### UCI Options
- OwnBook (specifies whether or not to use a separate opening book)
- BookPath (path to a file containing book positions in a supported format)
- Threads (for every thread doubling, a gain of about 70-80 elo can be expected)
- Hash (the amount of the memory allocated for the transposition table (actual memory usage will be greater))
- Weights (the absolute path to a binary weights file. If the default "EMBEDDED" path is chosen, the embedded weights will be used.)

### Features
- From scratch neural network training and execution (using OpenMP SIMD directives and SIMD intrinsics) implementation 
  (training scripts use PyTorch for GPU acceleration and can be found [here](https://github.com/connormcmonigle/seer-training)).
- Plain magic bitboard move generation with constexpr compile time generated attack tables.
- Principal variation search inside an iterative deepening framework
- Lockless shared transposition table (using Zobrist hashing)
- Move Ordering (SEE for captures + Killer Move, Combined Butterfly History, Counter Move History and Follow Up History for quiets)
- History pruning as well as SEE pruning in QSearch
- History extensions
- Null move pruning
- Static null move pruning (reverse futility pruning)
- Futility pruning
- Late move reductions
- Aspiration windows

### Compiling
The latest network can be found [here](https://github.com/connormcmonigle/seer-training/releases)
```
cd build
wget -O eval.bin https://github.com/connormcmonigle/seer-training/releases/download/0x35bb516b/0x35bb516b.bin
make EVALFILE=eval.bin
```
