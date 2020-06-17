import dataset
import config as C

config = C.Config('config.yaml')

d = dataset.Data(config)

for i in range(100):
  print(d.sample_batch())
