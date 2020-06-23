from os import path
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

import config as C
import util
import dataset
import model

def train_step(M, sample, opt, loss_history, report=False):
  pov, white, black, score = sample
  pred = M(pov, white, black)
  loss = model.loss_fn(score, pred)
  if report:
    print(loss.item())
  loss.backward()
  loss_history.append(loss.item())
  opt.step()
  M.zero_grad()


def main():
  config = C.Config('config.yaml')

  M = model.NNUE().to(config.device)

  if (path.exists(config.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(config.model_save_path))

  data = dataset.Data(config)

  opt = optim.Adadelta(M.parameters(), lr=config.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)

  loss_history = []
  
  for epoch in range(1, config.epochs + 1):
    M.to_binary_file(config.bin_model_save_path)
    for i in range(config.epoch_length):
      # update visual data
      if (i % config.test_rate) == 0:
        plt.clf()
        plt.plot(loss_history)
        plt.savefig('{}/loss_graph.png'.format(config.visual_directory), bbox_inches='tight')
      
      sample = data.sample_batch()
      train_step(M, sample, opt, loss_history, report=(0 == i % config.report_rate))

    torch.save(M.state_dict(), config.model_save_path)
    scheduler.step()


if __name__ == '__main__':
  main()
