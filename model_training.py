"""model training
"""
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

import mlflow
from omegaconf import DictConfig
from src.utils import (
    lightning_modules,
    mlflow_utils,
    train,
)
import torch.nn as nn

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="defaults",
)
def train_training(config: DictConfig) -> None:
    """Train the pytorch-geometric model
    Training is performed using torch-lighning
    Training logged in mlflow

    Args:
        config (DictConfig): Hydra config with all experiments parameters
        datapath (Path): Path to the folder with training graphs
    """


    model_type = config["training_args"]["model_type"]

    mlf_logger = mlflow_utils.create_experiment(
        mlflow_path=config["training"]["mlflow"]["tracking_uri"],
        experiment_name=config["training"]["mlflow"]["experiment_name"],
    )

    mlflow.log_param("features_list", config["training"]["training_params"]["features_list"])
    mlflow.log_params(dict(config["training"]["training_params"]["trainer_params"].items()))
    mlflow.log_params(
        dict(config["training"]["training_params"]["data_module_params"].items())
    )
    mlflow.log_params(
        dict(config["training"]["training_params"]["optimization_params"].items())
    )
    mlflow.log_params(dict(config["model_dict"][model_type].items()))

    model = hydra.utils.instantiate(config["model_dict"][model_type])
    #### PYTORCH LIGHTNING 
    # T'instantie ton model typiquement model = Module.Class_model(parametres)
    model = lightning_modules.Model(
        model,
        optimization_params=config["training"]["training_params"]["optimization_params"],
        n_classes=config["model_dict"]["transformer_encoder"]["n_classes"],
    )

    dataset_path = config["training_data"]
    datamodule = lightning_modules.CustomDataModule(
        dataset_path,
        config["training"]["training_params"]["data_module_params"]["batch_size"],
    )
    
    train.model_training(
        model,
        datamodule,
        config["training"]["training_params"]["trainer_params"],
        logger=mlf_logger,
    )
    
if __name__=='main' : 
    train_training()
else : 
    train_training()