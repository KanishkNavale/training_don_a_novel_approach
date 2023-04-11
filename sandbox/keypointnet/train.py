from src.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(yaml_config_path="sandbox/keypointnet/keypointnet-config.yaml")
    trainer.fit()
