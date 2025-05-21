import logging
import os

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from tutorial.accelerate.utils import save_checkpoint

from tutorial.data.dummy_dataset import DummyDataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(config: DictConfig):
    set_seed(config.seed)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    accelerator = Accelerator(
        log_with=config.log_with,
        project_dir=config.output_dir,
    )
    # init trackers
    accelerator.init_trackers(config.output_dir + "_test", config=dict(config))

    if accelerator.is_main_process:
        logger.info(accelerator.state)
        os.makedirs(config.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # load dataset
    ds = DummyDataset(config.num_examples, weights=config.weights)
    train_dl = DataLoader(
        ds,
        batch_size=config.batch_size,
        sampler=DistributedSampler(ds),
    )

    model = nn.Sequential(nn.Linear(len(config.weights), 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model, optimizer, train_dl = accelerator.prepare(model, optimizer, train_dl)

    for epoch in tqdm(
        range(config.epochs), desc="Epochs", disable=not accelerator.is_main_process
    ):
        epoch_loss = 0.0
        num_batches = 0

        for inputs, targets in train_dl:
            outputs = model(inputs)
            loss = F.mse_loss(outputs.squeeze(), targets)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.detach()
            num_batches += 1

        epoch_loss = epoch_loss / num_batches
        gathered_losses = accelerator.gather(epoch_loss)
        avg_loss = gathered_losses.mean().item()

        if accelerator.is_main_process:
            accelerator.log({"loss": avg_loss}, step=epoch + 1)
            logger.info(f"Epoch {epoch+1} loss: {avg_loss}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Training complete")
        print(model.state_dict())
        save_checkpoint(model, accelerator, config.output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
