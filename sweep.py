import argparse
import json

import pytorch_lightning as pl

import src.data_loaders as module_data
import wandb
import yaml
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from train import FinetuneEmbeddings, ToxicClassifier


def sweep_iteration(config, args, dataset, val_dataset):
    wandb.init(entity="anitavero")  # required to have access to `wandb.config`
    wandb_logger = WandbLogger()
    config_slice = wandb.config

    if "weighted_sampler" in config_slice:
        config["weighted_sampler"] = config_slice["weighted_sampler"]
    if "lr" in config_slice:
        config["optimizer"]["args"]["lr"] = config_slice["lr"]
    if "weight_decay" in config_slice:
        config["optimizer"]["args"]["weight_decay"] = config_slice["weight_decay"]
    if "batch_size" in config_slice:
        config["batch_size"] = config_slice["batch_size"]
    if "p" in config_slice:
        config["arch"]["args"]["p"] = config_slice["p"]
    if "hidden_layer_sizes" in config_slice:
        config["arch"]["args"]["hidden_layer_sizes"] = config_slice["hidden_layer_sizes"]

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )
    # model
    if config["arch"]["type"] == "finetune":
        config["arch"]["args"]["num_features"] = dataset.embeddings.shape[1]
        model = FinetuneEmbeddings(config)
    else:
        model = ToxicClassifier(config)

    # training
    trainer = pl.Trainer(
        logger=wandb_logger,
        devices=config["n_gpu"],
        accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        deterministic=True,
    )
    trainer.fit(model, data_loader, valid_data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-sc",
        "--sweep_config",
        default=None,
        type=str,
        help="Sweep config file path (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers used in the data loader (default: 10)",
    )
    parser.add_argument("-e", "--n_epochs", default=100, type=int, help="if given, override the num")
    args = parser.parse_args()

    with open(args.sweep_config) as stream:
        sweep_config = yaml.safe_load(stream)
    with open(args.config) as f:
        config = json.load(f)

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    sweep_id = wandb.sweep(sweep_config)
    # Workaround: wandb.agent can't receive functions with parameters
    seep_iter_func = lambda: sweep_iteration(config, args, dataset, val_dataset)
    wandb.agent(sweep_id, function=seep_iter_func)
