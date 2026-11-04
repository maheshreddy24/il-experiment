import logging
import os
import random
import time

import minari
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import Policy


def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f"rvm_trainer.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Full determinism would also require torch.use_deterministic_algorithms(True);
    # without it cudnn.deterministic only buys slower convs, not reproducibility.
    torch.backends.cudnn.benchmark = True


def make_dataloaders(dataset_id, batch_size, test_fraction=0.1):
    """ Split by episode so test transitions come from unseen trajectories """
    dataset = minari.load_dataset(dataset_id, download=True)
    episodes = list(dataset.iterate_episodes())
    random.shuffle(episodes)
    num_test = max(1, int(len(episodes) * test_fraction))

    def to_tensor_dataset(eps):
        # observations has one more entry than actions (the final observation)
        obs = np.concatenate([ep.observations[:-1] for ep in eps])
        act = np.concatenate([ep.actions for ep in eps])
        return TensorDataset(torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(act, dtype=torch.float32))

    train_loader = DataLoader(to_tensor_dataset(episodes[num_test:]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(to_tensor_dataset(episodes[:num_test]), batch_size=batch_size)
    return train_loader, test_loader, dataset


class Optimisation:
    def __init__(self, config, model, train_dataloader, test_dataloader):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.checkpoint_dir = f"checkpoints/exp_{int(time.time())}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "train.log"))
        self.global_step = 0

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pt")
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config,
        }
        torch.save(ckpt, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        total_loss = 0.0
        for obs, act in self.test_dataloader:
            obs, act = obs.to(self.device), act.to(self.device)
            total_loss += self.criterion(self.model(obs), act).item()
        avg_loss = total_loss / len(self.test_dataloader)
        self.logger.info(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self):
        for epoch in range(1, self.config['num_epochs'] + 1):
            self.model.train()
            total_loss = 0.0
            for obs, act in self.train_dataloader:
                obs, act = obs.to(self.device), act.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(obs), act)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                self.global_step += 1

            avg_loss = total_loss / len(self.train_dataloader)
            self.logger.info(f"Epoch [{epoch}/{self.config['num_epochs']}], Train Loss: {avg_loss:.4f}")
            self.eval()
            self.save_checkpoint(epoch)


if __name__ == "__main__":
    config = {
        'dataset_id': "mujoco/hopper/simple-v0",
        'learning_rate': 0.001,
        'num_epochs': 10,
        'batch_size': 256,
        'seed': 42,
    }

    set_seed(config['seed'])

    train_dataloader, test_dataloader, dataset = make_dataloaders(config['dataset_id'], config['batch_size'])
    obs_dim = dataset.observation_space.shape[0]
    act_dim = dataset.action_space.shape[0]

    model = Policy(obs_dim, act_dim)
    optimizer = Optimisation(config, model, train_dataloader, test_dataloader)
    optimizer.train()
