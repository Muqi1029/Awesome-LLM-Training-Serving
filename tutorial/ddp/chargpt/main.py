import os

import hydra
import torch
from char_dataset import CharDataset, DataConfig
from model import GPT
from omegaconf import DictConfig
from optimizer import OptimizerConfig, create_optimizer
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import random_split
from trainer import Trainer, TrainerConfig
from utils import CfgNode


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def get_train_objs(gpt_cfg: CfgNode, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)

    return model, optimizer, train_set, test_set


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    # setup ddp
    ddp_setup()

    #
    gpt_cfg = GPT.get_default_config()
    gpt_cfg.merge_from_dict(cfg["gpt_config"])
    opt_cfg = OptimizerConfig(**cfg["optimizer_config"])
    data_cfg = DataConfig(**cfg["data_config"])
    trainer_cfg = TrainerConfig(**cfg["trainer_config"])

    # create train objects
    model, optimizer, train_data, test_data = get_train_objs(gpt_cfg, opt_cfg, data_cfg)
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)

    # start training
    trainer.train()

    # destroy process group
    destroy_process_group()


if __name__ == "__main__":
    main()
