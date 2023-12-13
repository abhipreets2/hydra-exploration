import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os

import tasks
import models
import backbones
import adapter
import heads
import torch.optim

from config_schemas.config_schema import setup_config

setup_config()

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(config: DictConfig)->None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    data_module = instantiate(config.data_module)
    task = instantiate(config.task)
    # return

    tb_logger = TensorBoardLogger("tb_logs", name="cifar")

    checkpoint_callback = ModelCheckpoint(
            monitor="validation_accuracy",
            dirpath="checkpoints",
            filename="cifar-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="max",
            )

    trainer = instantiate(config.trainer, logger=tb_logger, callbacks=[checkpoint_callback])

    trainer.fit(task, datamodule=data_module)
    trainer.test(datamodule=data_module)

if __name__=="__main__":
    train()
    # data_modules.MNISTDataModule(
    #     batch_size=64,
    #     num_workers=8,
    #     pin_memory=True,
    #     drop_last=True,
    #     data_dir="./data/mnist",
    # )

    # backbone = backbones.ResNet18(
    #     pretrained=True
    # )
    # adapter = adapter.LinearAdapter(
    #     in_features=512,
    #     out_features=10
    # )
    # head = heads.IdentityHead()
    # model = models.SimpleModel(
    #     backbone=backbone,
    #     adapter=adapter,
    #     head=head
    # )
    # print("done")
    # loss_function = 
    # tasks.MNISTClassification(
    #     model=model,
    #     optimizer=,
    #     loss_function=
    # )

