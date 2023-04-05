from src.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer(yaml_config_path="sandbox/don/don-config.yaml")
    trainer.fit()
