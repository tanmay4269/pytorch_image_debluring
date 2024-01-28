import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from utils import *
import config
import networks
import datasets

class Trainer:
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available else "cpu"
    
    """ Model """
    self.models = {}
    self.parameters_to_train = []

    self.models["encoder"] = networks.ResnetEncoder()
    self.models["encoder"].to(self.device)

    self.models["gaussian_decoder"] = networks.Decoder()
    self.models["gaussian_decoder"].to(self.device)
    self.parameters_to_train += list(
      self.models["gaussian_decoder"].parameters())
    
    self.model_optimizer = optim.Adam(self.parameters_to_train, config.learning_rate)
    self.model_lr_scheduler = optim.lr_scheduler.StepLR(
      self.model_optimizer, config.scheduler_step_size, 0.1)
    
    """ Data """
    self.dataset = datasets.GaussianDataset
    
    self.full_datasets = {}
    self.train_datasets = {}
    self.val_datasets = {}

    self.train_loaders = {}
    self.val_loaders = {}

    self.full_datasets["gaussian"] = self.dataset(
      config.dir_path_sharp, 
      config.dir_path_sharp,
      apply_constant_blur=True)
    
    train_size = int(config.train_ratio * len(self.full_datasets["gaussian"]))
    val_size = len(self.full_datasets["gaussian"]) - train_size

    self.train_datasets["gaussian"], self.val_datasets["gaussian"] = torch.utils.data.random_split(self.full_datasets["gaussian"], [train_size, val_size])
    
    self.train_loaders["gaussian"] = DataLoader(
      self.train_datasets["gaussian"], batch_size=config.batch_size, 
      num_workers=config.num_workers, shuffle=True)

    self.val_loaders["gaussian"] = DataLoader(
      self.val_datasets["gaussian"], batch_size=config.batch_size, 
      num_workers=config.num_workers, shuffle=True)
    
    self.num_total_steps = train_size // config.batch_size * config.num_epochs


    """ Logs """
    self.writers = {}
    for mode in ["train", "val"]:
        self.writers[mode] = SummaryWriter(os.path.join(config.log_path, mode))

  def set_train(self):
    for m in self.models.values():
      m.train()

  def set_eval(self):
    for m in self.models.values():
      m.eval()

  def train(self):
    self.epoch = 0
    self.step = 0
    self.start_time = time.time()

    for self.epoch in range(config.num_epochs):
      self.run_epoch()
      if (self.epoch + 1) % config.save_frequency == 0:
        self.save_model()

  def run_epoch(self):
    self.model_lr_scheduler.step()

    print("Training")
    self.set_train()

    for batch_idx, batch in enumerate(self.train_loaders["gaussian"]):
      before_op_time = time.time()

      outputs, loss = self.process_batch(batch)

      self.model_optimizer.zero_grad()
      loss.backward()
      self.model_optimizer.step()

      duration = time.time() - before_op_time

      # log less frequently after the first 2000 steps to save time & disk space
      early_phase = batch_idx % config.log_frequency == 0 and self.step < 2000
      late_phase = self.step % 2000 == 0

      if early_phase or late_phase:
        self.log_time(batch_idx, duration, loss.cpu().data)

        self.log("train", batch, outputs, loss)
        self.val()

      self.step += 1

  def process_batch(self, batch):
    images, labels = batch["image"], batch["label"]

    images = images.to(self.device)
    labels = labels.to(self.device)

    # Forward pass
    features = self.models["encoder"](images)
    outputs = self.models["gaussian_decoder"](features)

    loss_fn = nn.MSELoss()
    loss = loss_fn(outputs, labels)

    return (outputs, loss)
  
  def val(self):
    """
    Validate the model on a single minibatch
    """
    self.set_eval()

    inputs = None
    for batch in self.val_loaders["gaussian"]:
      inputs = batch
      break

    with torch.no_grad():
      outputs, loss = self.process_batch(inputs)

      self.log("val", inputs, outputs, loss)
      del inputs, outputs, loss

    self.set_train()
  
  def log_time(self, batch_idx, duration, loss):
    """
    Print a logging statement to the terminal
    """
    samples_per_sec = config.batch_size / duration
    time_sofar = time.time() - self.start_time
    training_time_left = (
      self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
        " | loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                              sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

  def log(self, mode, batch, outputs, loss):
    writer = self.writers[mode]

    writer.add_scalar("loss", loss, self.step)
    
    for j in range(min(4, config.batch_size)):
      writer.add_image("input_image", batch["image"][j], self.step)
      writer.add_image("label_image", batch["label"][j], self.step)
      writer.add_image("model_output", outputs[j], self.step)

  # understand what's this doing
  def save_model(self):
    """
    Saves model to disk
    """

    save_folder = os.path.join(config.log_path, "models", f"weights_{self.epoch}")
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)

    for model_name, model in self.models.items():
      save_path = os.path.join(save_folder, f"{model_name}.pth")
      to_save = model.state_dict()

      torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "adam.pth")
    torch.save(self.model_optimizer.state_dict(), save_path)

  # understand what's this doing
  def load_model(self):
    """
    Load model(s) from disk
    """

    config.load_weights_folder = os.path.expanduser(config.load_weights_folder)

    assert os.path.isdir(config.load_weights_folder), \
      "Cannot find folder {}".format(config.load_weights_folder)
    print("loading model from folder {}".format(config.load_weights_folder))

    for n in config.models_to_load:
      print("Loading {} weights...".format(n))
      path = os.path.join(config.load_weights_folder, "{}.pth".format(n))
      model_dict = self.models[n].state_dict()
      pretrained_dict = torch.load(path)
      pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
      model_dict.update(pretrained_dict)
      self.models[n].load_state_dict(model_dict)

    # loading adam state
    optimizer_load_path = os.path.join(config.load_weights_folder, "adam.pth")
    if os.path.isfile(optimizer_load_path):
      print("Loading Adam weights")
      optimizer_dict = torch.load(optimizer_load_path)
      self.model_optimizer.load_state_dict(optimizer_dict)
    else:
      print("Cannot find Adam weights so Adam is randomly initialized")


if __name__ == "__main__":
  trainer = Trainer()
  trainer.train()

