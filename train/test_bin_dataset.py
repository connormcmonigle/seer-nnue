import torch
import nnue_bin_dataset
import config as C

config = C.Config('config.yaml')

d = nnue_bin_dataset.NNUEBinData(config.train_bin_data_path, config)
#d = torch.utils.data.DataLoader(d, num_workers=config.num_workers, worker_init_fn=nnue_bin_dataset.worker_init_fn)

for i in d.seq_data_iter():
  print(i)
