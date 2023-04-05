import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.configurations import TrainerConfig, DataLoaderConfig
from src.datamodule import DataModule
from src.don import DON
from src.keypointnet import KeypointNetwork
from src.utils import initialize_config_file


class Trainer:

    def __init__(self, yaml_config_path: str) -> None:

        # Deconstruct Configs
        read_yaml = initialize_config_file(yaml_config_path)
        trainer_config = TrainerConfig.from_dictionary(read_yaml)
        dataloader_config = DataLoaderConfig.from_dictionary(read_yaml)

        # Init. logger
        logger = TensorBoardLogger(save_dir=trainer_config.tensorboard_path,
                                   default_hp_metric=False) if trainer_config.enable_logging else None

        # Initialize models
        if "don" in read_yaml.keys():
            self.model = DON(yaml_config_path)

        elif "keypointnet" in read_yaml.keys():
            self.model = KeypointNetwork(yaml_config_path)

        else:
            raise NotImplementedError("Model not implemented")

        # Model Settings
        torch.set_float32_matmul_precision(trainer_config.precision)

        # Init. checkpoints here,
        model_checkpoints = ModelCheckpoint(monitor='val_loss',
                                            filename=trainer_config.model_checkpoint_name,
                                            dirpath=trainer_config.model_path,
                                            mode='min',
                                            save_top_k=1,
                                            verbose=True)

        lr_monitor = LearningRateMonitor(logging_interval='step')
        bar = RichProgressBar()
        checkpoints = [model_checkpoints, bar, lr_monitor]

        # Init. model and datamodule
        self.datamodule = DataModule(dataloader_config)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        self.trainer = pl.Trainer(logger=logger,
                                  enable_checkpointing=trainer_config.enable_checkpointing,
                                  check_val_every_n_epoch=trainer_config.validation_frequency,
                                  max_epochs=trainer_config.epochs,
                                  log_every_n_steps=trainer_config.logging_frequency,
                                  default_root_dir=trainer_config.training_directory,
                                  callbacks=checkpoints,
                                  enable_model_summary=True,
                                  detect_anomaly=True,
                                  accelerator='auto',
                                  devices='auto')

    def fit(self) -> None:
        self.trainer.fit(self.model, datamodule=self.datamodule)
