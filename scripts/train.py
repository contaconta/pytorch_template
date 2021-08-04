import os
from dataclasses import dataclass
from pathlib import Path
from omegaconf import OmegaConf, MISSING
import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from mypackage.models.plt import LitModel


@dataclass
class TrainingConfig:
    lr: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 200
    seed: int = 42


@dataclass
class Config:
    experiment_name: str = MISSING
    output_root_dir: str = MISSING
    log_dir: str = "tb_logs"
    gpus: int = 0
    num_workers: int = 0
    accelerator: str = "ddp"
    training: TrainingConfig = TrainingConfig()


def main():
    confing_cli = OmegaConf.from_cli()
    config_default = OmegaConf.structured(Config)
    config: Config = OmegaConf.to_object(OmegaConf.merge(config_default, confing_cli))

    if config.num_workers == 0:
        print(f"set num_workers by default")
        config.num_workers = os.cpu_count()

    accelerator = config.accelerator
    gpus = config.gpus

    if config.gpus < 0:
        gpus = torch.cuda.device_count()
        accelerator = None

    print("==== config ====")
    print(config)

    pl.seed_everything(seed=config.training.seed, workers=True)
    output_root_dir = Path(config.output_root_dir)

    model = torchvision.models.resnet18()
    lit_model = LitModel(base_model=model, lr=config.training.lr)
    datamodule = pl.LightningDataModule()

    mlf_logger = MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=str((output_root_dir / "mlruns")),
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_root_dir / config.experiment_name / "checkpoints"),
        filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        verbose=True,
        monitor="val_acc",
        mode="max",
    )

    trainer = pl.Trainer(
        gpus=gpus,
        accelerator=accelerator,
        max_epochs=config.training.max_epochs,
        logger=mlf_logger,
         default_root_dir=str(output_root_dir),
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
    )
    trainer.fit(lit_model, datamodule=datamodule)

    output_model_path = output_root_dir / config.experiment_name / "model.pt"
    print(f"save base model at {output_model_path}")
    torch.save(model.state_dict(), output_model_path)


if __name__ == "__main__":
    main()
