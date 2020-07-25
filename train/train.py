from os import path
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import config as C
import util
import nnue_bin_dataset
import model


def train_step(M, sample, opt, queue, max_queue_size, lambda_, report=False):
  pov, white, black, outcome, score = sample
  pred = M(pov, white, black)
  loss = model.loss_fn(outcome, score, pred, lambda_)
  if report:
    print(loss.item())
  loss.backward()
  if(len(queue) >= max_queue_size):
    queue.pop(0)
  queue.append(loss.item())
  opt.step()
  M.zero_grad()


def main():
  config = C.Config('config.yaml')

  sample_to_device = lambda x: tuple(map(lambda t: t.to(config.device, non_blocking=True), x))

  M = model.NNUE().to(config.device)

  if (path.exists(config.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(config.model_save_path))

  data = nnue_bin_dataset.NNUEBinData(config)
  data_loader = torch.utils.data.DataLoader(data,\
    batch_size=config.batch_size,\
    num_workers=config.num_workers,\
    pin_memory=True,\
    worker_init_fn=nnue_bin_dataset.worker_init_fn)

  opt = optim.Adadelta(M.parameters(), lr=config.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)

  loss_history = []
  queue = []
  
  for epoch in range(1, config.epochs + 1):
    for i, sample in enumerate(data_loader):
      # update visual data
      if (i % config.test_rate) == 0 and i != 0:
        loss_history.append(sum(queue) / len(queue))
        plt.clf()
        plt.plot(loss_history)
        plt.savefig('{}/loss_graph.png'.format(config.visual_directory), bbox_inches='tight')
      
      if (i % config.save_rate) == 0 and i != 0:
        print('Saving model ...')
        M.to_binary_file(config.bin_model_save_path)
        torch.save(M.state_dict(), config.model_save_path)

      train_step(M, sample_to_device(sample), opt, queue, max_queue_size=config.max_queue_size, lambda_=config.lambda_, report=(0 == i % config.report_rate))

    scheduler.step()


if __name__ == '__main__':
  main()
