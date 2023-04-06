
from src.datamodule import DataModule
from src.don import DON, load_trained_don_model
from src.don.metrices import AUC_for_PCK
from src.utils import initialize_config_file
from src.configurations import DataLoaderConfig


if __name__ == "__main__":

    # Init. Trained DON Model
    trained_model = DON("sandbox/don/don-config.yaml")

    # Init. Dataloader
    config = initialize_config_file("sandbox/don/don-config.yaml")
    dataloader_config = DataLoaderConfig.from_dictionary(config)
    dataloader = DataModule(dataloader_config)

    PCKS, AUCS = AUC_for_PCK(trained_model,
                             dataloader,
                             iterations=3,
                             n_correspondences=100)

    print(PCKS)
    print(AUCS)
