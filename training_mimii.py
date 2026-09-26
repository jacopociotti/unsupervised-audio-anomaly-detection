import torch
import numpy as np

from optparse import OptionParser
from sklearn import metrics

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data import MIMIIDataModule
from model import Wavegram_AttentionMap


def train(configs):

    seed_everything(configs.seed, workers=True)

    datamodule = MIMIIDataModule(
        path_data=configs.path_data,
        sample_rate=configs.sr,
        duration=configs.duration,
        batch_size=configs.batch_size,
        num_channels=configs.num_channels,
        spatial_augmentation=configs.spatial_augmentation
    )

    model = Wavegram_AttentionMap(
        h=128,
        lr=configs.lr,
        num_classes=configs.num_classes,
        num_channels=configs.num_channels
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="best_models/",
        filename=(f"mimii-{configs.num_channels}ch-"f"aug{int(configs.spatial_augmentation)}-""{epoch:02d}-{val/loss_class:.4f}"),
        monitor="val/loss_class",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=configs.epochs,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)

    print("Best checkpoint:", checkpoint_callback.best_model_path)

    return model, trainer, datamodule, checkpoint_callback

def evaluate(model, trainer, datamodule):

    # Puliamo eventuali risultati precedenti
    model.errors_list = []
    model.clean_errors = []
    model.anomaly_errors = []
    model.labels = []
    model.classes = []

    # Usa automaticamente il miglior checkpoint
    # selezionato sulla validation
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # Concatena tutti i batch del test
    errors = torch.cat([x.detach().cpu() for x in model.errors_list]).numpy()

    labels = torch.cat([x.detach().cpu() for x in model.labels]).numpy()

    classes = torch.cat([x.detach().cpu() for x in model.classes]).numpy()

    ############
    unique_labels = np.unique(labels)

    print("\n----- TEST INFO -----")
    print(f"Samples: {len(labels)}")
    print(f"Normal: {(labels == 0).sum()}")
    print(f"Abnormal: {(labels == 1).sum()}")
    print(f"Labels presenti: {unique_labels}")

    if len(unique_labels) < 2:
        print(
            "AUC/pAUC non calcolabili: il test limitato "
            "contiene una sola classe."
        )
        return None, None
    #########################
    # AUC
    auc = metrics.roc_auc_score(labels, errors)

    # partial AUC fino a FPR = 0.1
    pauc = metrics.roc_auc_score(labels, errors, max_fpr=0.1)

    print("\n----- TEST RESULTS -----")
    print(f"Samples: {len(labels)}")
    print(f"Normal: {(labels == 0).sum()}")
    print(f"Abnormal: {(labels == 1).sum()}")
    print(f"AUC:  {auc:.4f}")
    print(f"pAUC: {pauc:.4f}")

    machine_ranges = {
    "Fan": (0, 4),
    "Pump": (4, 8),
    "Slider": (8, 12),
    "Valve": (12, 16),
    }

    print("\n----- RESULTS BY MACHINE -----")

    for machine, (start, end) in machine_ranges.items():

        mask = (classes >= start) & (classes < end)

        machine_labels = labels[mask]
        machine_errors = errors[mask]

        if len(np.unique(machine_labels)) < 2:
            print(f"{machine}: AUC/pAUC non calcolabili")
            continue

        machine_auc = metrics.roc_auc_score(machine_labels, machine_errors)

        machine_pauc = metrics.roc_auc_score(machine_labels, machine_errors, max_fpr=0.1)

        print(f"{machine}: "f"AUC={machine_auc:.4f}, "f"pAUC={machine_pauc:.4f}, "f"N={len(machine_labels)}")

        return auc, pauc

if __name__ == "__main__":

    num_classes = 16
    path_data = r"D:\jacopo\dataset_0dB"

    parser = OptionParser()
    parser.add_option("--seed", type="int", default=42)
    parser.add_option("--path_data", type="string", default=path_data)
    parser.add_option("--num_channels", type="int", default=1)
    parser.add_option("--spatial_augmentation", action="store_true", default=False)
    parser.add_option("--num_classes", type="int", default=num_classes)
    parser.add_option("--batch_size", type="int", default=2)
    parser.add_option("--epochs", type="int", default=1)
    parser.add_option("--lr", type="float", default=0.0001)
    parser.add_option( "--sr", type="int", default=16000)
    parser.add_option("--duration", type="int", default=10)
    configs, _ = parser.parse_args()
    print("----- MIMII experiment -----")
    print("Seed:", configs.seed)
    print("Dataset:", configs.path_data)
    print("Channels:", configs.num_channels)
    print("Spatial augmentation:", configs.spatial_augmentation)
    print("Classes:", configs.num_classes)
    print("Batch size:", configs.batch_size)
    print("Epochs:", configs.epochs)
    print("Learning rate:", configs.lr)

 
    model, trainer, datamodule, checkpoint_callback = train(configs)

    evaluate(model, trainer, datamodule)
