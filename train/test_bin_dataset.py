import nnue_bin_dataset
import config as C

config = C.Config('config.yaml')

d = nnue_bin_dataset.NNUEBinData(config)

for i in range(10000):
  print(d.sample_batch())
