from optparse import OptionParser
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_multi_snr import MIMIIMultiSNRDataModule
from model import Wavegram_AttentionMap


MACHINE_RANGES = {
    "Fan": (0, 4),
    "Pump": (4, 8),
    "Slider": (8, 12),
    "Valve": (12, 16),
}


def experiment_subdir(configs):
    if configs.num_channels == 1:
        return "1ch"
    spatial = "si_spatial" if configs.spatial_augmentation else "no_spatial"
    return f"{configs.num_channels}ch_{spatial}"


def create_datamodule(configs):
    return MIMIIMultiSNRDataModule(
        paths_by_snr={
            "-6dB": configs.path_minus6,
            "0dB": configs.path_0,
            "6dB": configs.path_6,
        },
        sample_rate=configs.sr,
        duration=configs.duration,
        batch_size=configs.batch_size,
        num_channels=configs.num_channels,
        spatial_augmentation=configs.spatial_augmentation,
    )


def train(configs):
    seed_everything(configs.seed, workers=True)
    output_dir = Path(configs.output_dir) / experiment_subdir(configs)
    checkpoint_dir = output_dir / "best_models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    datamodule = create_datamodule(configs)
    model = Wavegram_AttentionMap(h=128, lr=configs.lr, num_classes=configs.num_classes, num_channels=configs.num_channels)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=(f"mimii-mixedSNR-{configs.num_channels}ch-"f"aug{int(configs.spatial_augmentation)}-""{epoch:02d}-{val/loss_class:.4f}"),
        monitor="val/loss_class",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]
    if configs.early_stopping:
        callbacks.append(EarlyStopping(monitor="val/loss_class", patience=configs.patience,mode="min", verbose=True))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=configs.epochs,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )
    trainer.fit(model, datamodule=datamodule)
    best_path = checkpoint_callback.best_model_path
    if not best_path or not Path(best_path).is_file():
        raise RuntimeError("Nessun best checkpoint disponibile: impossibile effettuare il test.")
    print(f"\nBest checkpoint: {best_path}")
    print(f"Best val/loss_class: {checkpoint_callback.best_model_score}")
    return trainer, datamodule, checkpoint_callback, output_dir


def evaluate_one_snr(trainer, datamodule, configs, checkpoint_callback, output_dir, snr):
    """Ricarica lo STESSO checkpoint per ogni SNR, con accumulatori puliti."""
    datamodule.select_test_snr(snr)
    checkpoint_path = checkpoint_callback.best_model_path
    model = Wavegram_AttentionMap.load_from_checkpoint(
        checkpoint_path,
        h=128,
        lr=configs.lr,
        num_classes=configs.num_classes,
        num_channels=configs.num_channels,
        map_location="cpu",
    )
    # I campi seguenti sono liste utilizzate da test_step()
    model.errors_list = []
    model.clean_errors = []
    model.anomaly_errors = []
    model.labels = []
    model.classes = []

    print(f"\n===== TEST SNR {snr} =====")
    # ckpt_path=None: il modello è già stato caricato dal checkpoint migliore.
    test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=None)

    if not model.errors_list:
        raise RuntimeError(f"Nessuno score generato nel test {snr}.")
    errors = torch.cat([v.detach().cpu() for v in model.errors_list]).numpy()
    labels = torch.cat([v.detach().cpu() for v in model.labels]).numpy()
    classes = torch.cat([v.detach().cpu() for v in model.classes]).numpy()
    if not (len(errors) == len(labels) == len(classes)):
        raise RuntimeError("Dimensioni di score, label e classi non coerenti.")

    def auc_pauc(y, score):
        if len(np.unique(y)) < 2:
            return None, None
        return (float(metrics.roc_auc_score(y, score)),float(metrics.roc_auc_score(y, score, max_fpr=0.1)))

    auc, pauc = auc_pauc(labels, errors)
    per_machine = {}
    for machine, (start, end) in MACHINE_RANGES.items():
        mask = (classes >= start) & (classes < end)
        m_auc, m_pauc = auc_pauc(labels[mask], errors[mask])
        per_machine[machine] = {"auc": m_auc, "pauc": m_pauc,"n": int(mask.sum())}

    test_dir = output_dir / f"test_{snr}"
    test_dir.mkdir(parents=True, exist_ok=True)
    results_path = test_dir / "results.txt"

    def fmt(value):
        return "non calcolabile" if value is None else f"{value:.4f}"

    with results_path.open("w", encoding="utf-8") as f:
        f.write("===== MIMII MULTI-SNR EXPERIMENT =====\n\n")
        f.write("----- CONFIGURATION -----\n")
        f.write(f"Training SNR: -6dB, 0dB, 6dB\n")
        f.write(f"Dataset -6dB: {configs.path_minus6}\n")
        f.write(f"Dataset 0dB: {configs.path_0}\n")
        f.write(f"Dataset 6dB: {configs.path_6}\n")
        f.write(f"Test SNR: {snr}\n")
        f.write(f"Channels: {configs.num_channels}\n")
        f.write(f"Spatial augmentation: {configs.spatial_augmentation}\n")
        f.write(f"Classes: {configs.num_classes}\n")
        f.write(f"Batch size: {configs.batch_size}\n")
        f.write(f"Epochs (maximum): {configs.epochs}\n")
        f.write(f"Learning rate: {configs.lr}\n")
        f.write(f"Seed: {configs.seed}\n")
        f.write(f"Early stopping: {configs.early_stopping}\n")
        if configs.early_stopping:
            f.write(f"Patience: {configs.patience}\n")
        f.write("\n----- CHECKPOINT -----\n")
        f.write(f"Best checkpoint: {checkpoint_path}\n")
        f.write(f"Best val/loss_class: {checkpoint_callback.best_model_score}\n")
        f.write("\n----- TEST METRICS -----\n")
        if test_results:
            for key, value in test_results[0].items():
                f.write(f"{key}: {value}\n")
        f.write("\n----- ANOMALY DETECTION -----\n")
        f.write(f"Samples: {len(labels)}\n")
        f.write(f"Normal: {int((labels == 0).sum())}\n")
        f.write(f"Abnormal: {int((labels == 1).sum())}\n")
        f.write(f"AUC: {fmt(auc)}\n")
        f.write(f"pAUC: {fmt(pauc)}\n")
        f.write("\n----- RESULTS BY MACHINE -----\n")
        for machine, result in per_machine.items():
            f.write(f"{machine}: AUC={fmt(result['auc'])}, "f"pAUC={fmt(result['pauc'])}, N={result['n']}\n")

    print(f"SNR {snr}: AUC={fmt(auc)}, pAUC={fmt(pauc)}")
    print(f"Results saved in: {results_path}")
    return auc, pauc


def main():
    parser = OptionParser()
    parser.add_option("--seed", type="int", default=42)
    parser.add_option("--path_minus6", type="string", default=None)
    parser.add_option("--path_0", type="string", default=None)
    parser.add_option("--path_6", type="string", default=None)
    parser.add_option("--output_dir", type="string", default="experiments_multi_snr")
    parser.add_option("--num_channels", type="int", default=1)
    parser.add_option("--spatial_augmentation", action="store_true", default=False)
    parser.add_option("--num_classes", type="int", default=16)
    parser.add_option("--batch_size", type="int", default=2)
    parser.add_option("--epochs", type="int", default=10)
    parser.add_option("--lr", type="float", default=0.0001)
    parser.add_option("--sr", type="int", default=16000)
    parser.add_option("--duration", type="int", default=10)
    parser.add_option("--early_stopping", action="store_true", default=False)
    parser.add_option("--patience", type="int", default=10)
    configs, _ = parser.parse_args()

    for key in ("path_minus6", "path_0", "path_6"):
        if not getattr(configs, key):
            parser.error(f"Specificare --{key}.")
    if configs.num_channels not in (1, 4, 8):
        parser.error("--num_channels deve essere 1, 4 o 8.")
    if configs.num_classes != 16:
        parser.error("Il dataset MIMII corrente usa --num_classes 16.")
    if configs.batch_size < 1 or configs.epochs < 1:
        parser.error("--batch_size e --epochs devono essere positivi.")

    print("===== MIMII TRAINING MULTI-SNR =====")
    print("SNR training: -6dB, 0dB, 6dB")
    print("Seed:", configs.seed)
    print(f"Canali: {configs.num_channels}")
    print(f"Spatial augmentation: {configs.spatial_augmentation}")
    print("Classes:", configs.num_classes)
    print("Batch size:", configs.batch_size)
    print("Epochs:", configs.epochs)
    print("Learning rate:", configs.lr)
    print(f"Early stopping: {configs.early_stopping}")
    if configs.early_stopping:
        print(f"Patience: {configs.patience}")



    trainer, datamodule, checkpoint_callback, output_dir = train(configs)
    results = {}
    for snr in ("-6dB", "0dB", "6dB"):
        results[snr] = evaluate_one_snr(trainer, datamodule, configs, checkpoint_callback, output_dir, snr)
    print("\n===== RIEPILOGO =====")
    for snr, (auc, pauc) in results.items():
        print(f"{snr}: AUC={auc}, pAUC={pauc}")


if __name__ == "__main__":
    main()
