from os import path
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

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


def get_validation_loss(M, sample, lambda_):
  with torch.no_grad():
    pov, white, black, outcome, score = sample
    pred = M(pov, white, black)
    loss = model.loss_fn(outcome, score, pred, lambda_).detach()
  return loss


def main():
  config = C.Config('config.yaml')

  sample_to_device = lambda x: tuple(map(lambda t: t.to(config.device, non_blocking=True), x))

  M = model.NNUE().to(config.device)

  if (path.exists(config.model_save_path)):
    print('Loading model ... ')
    M.load_state_dict(torch.load(config.model_save_path))

  train_data = nnue_bin_dataset.NNUEBinData(config.train_bin_data_path, config)
  validation_data = nnue_bin_dataset.NNUEBinData(config.validation_bin_data_path, config)
  
  train_data_loader = torch.utils.data.DataLoader(train_data,\
    batch_size=config.batch_size,\
    num_workers=config.num_workers,\
    pin_memory=True,\
    worker_init_fn=nnue_bin_dataset.worker_init_fn)

  validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=config.batch_size)
  validation_data_loader_iter = iter(validation_data_loader)

  writer = SummaryWriter(config.visual_directory)

  writer.add_graph(M, sample_to_device(next(iter(train_data_loader)))[:3])

  opt = optim.Adadelta(M.parameters(), lr=config.learning_rate)
  scheduler = optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)

  queue = []
  
  for epoch in range(1, config.epochs + 1):
    for i, sample in enumerate(train_data_loader):
      # update visual data
      if (i % config.test_rate) == 0 and i != 0:
        step = train_data.cardinality() * (epoch - 1) + i * config.batch_size
        train_loss = sum(queue) / len(queue)
        
        validation_sample = next(validation_data_loader_iter, None)
        if validation_sample == None:
          validation_data_loader_iter = iter(validation_data_loader)
          validation_sample = next(validation_data_loader_iter, None)
        validation_sample = sample_to_device(validation_sample)
        
        validation_loss = get_validation_loss(M, validation_sample, lambda_=config.lambda_)
        
        writer.add_scalar('train_loss', train_loss, step)
        writer.add_scalar('validation_loss', validation_loss, step)
      
      if (i % config.save_rate) == 0 and i != 0:
        print('Saving model ...')
        M.to_binary_file(config.bin_model_save_path)
        torch.save(M.state_dict(), config.model_save_path)

      train_step(M, sample_to_device(sample), opt, queue, max_queue_size=config.max_queue_size, lambda_=config.lambda_, report=(0 == i % config.report_rate))

    scheduler.step()


if __name__ == '__main__':
  main()
