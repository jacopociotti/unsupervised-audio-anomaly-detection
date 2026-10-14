from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from data import MIMIIDataset, MIMIIDataModule


class MIMIIMultiSNRDataModule(LightningDataModule):
    def __init__(self, paths_by_snr, sample_rate=16000, duration=10, batch_size=4, num_channels=1, spatial_augmentation=False):
        super().__init__()
        if not isinstance(paths_by_snr, dict) or len(paths_by_snr) < 2:
            raise ValueError("paths_by_snr deve essere un dizionario con almeno due SNR.")

        self.paths_by_snr = {str(snr): Path(path) for snr, path in paths_by_snr.items()}
        self.sample_rate = sample_rate
        self.duration = duration
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.spatial_augmentation = spatial_augmentation

        self.train_list = []
        self.val_list = []
        self.test_lists = {}
        self.test_snr = next(iter(self.paths_by_snr))

        modules = {}
        for snr, path in self.paths_by_snr.items():
            if not path.is_dir():
                raise FileNotFoundError(f"Dataset {snr} non trovato: {path}")
            modules[snr] = MIMIIDataModule(path_data=str(path), sample_rate=sample_rate, duration=duration, batch_size=batch_size, num_channels=num_channels, spatial_augmentation=spatial_augmentation)

        # Confronta percorsi relativi, includendo macchina, ID, condizione
        # e nome WAV. Anche gli split (train/val/test) devono coincidere.
        reference_snr = next(iter(modules))
        reference_module = modules[reference_snr]
        for split in ("train_list", "val_list", "test_list"):
            reference = self._relative_names(getattr(reference_module, split), self.paths_by_snr[reference_snr])
            if not reference:
                raise ValueError(f"Split {split} vuoto per SNR {reference_snr}.")
            for snr, module in modules.items():
                current = self._relative_names(getattr(module, split), self.paths_by_snr[snr])
                if current != reference:
                    missing = sorted(reference - current)[:3]
                    extra = sorted(current - reference)[:3]
                    raise ValueError(f"Split {split} differente fra {reference_snr} e {snr}. "f"Assenti (esempi): {missing}; aggiuntivi (esempi): {extra}. ""Verificare i nomi delle registrazioni prima di procedere.")

        # Mantiene esattamente gli split definiti da MIMIIDataModule.
        for snr, module in modules.items():
            self.train_list.extend(module.train_list)
            self.val_list.extend(module.val_list)
            self.test_lists[snr] = list(module.test_list)

        self._print_summary()

    @staticmethod
    def _relative_names(files, root):
        names = [Path(f).relative_to(root).as_posix() for f in files]
        if len(names) != len(set(names)):
            raise ValueError(f"File duplicati nel dataset {root}.")
        return set(names)

    def _dataset(self, files, augment=False):
        return MIMIIDataset(files, sample_rate=self.sample_rate, duration=self.duration, num_channels=self.num_channels, spatial_augmentation=augment)

    def _print_summary(self):
        print("\\n----- MULTI-SNR DATASET -----")
        for snr in self.paths_by_snr:
            root = self.paths_by_snr[snr]
            n_train = sum(Path(p).is_relative_to(root) for p in self.train_list)
            n_val = sum(Path(p).is_relative_to(root) for p in self.val_list)
            test_files = self.test_lists[snr]
            n_normal = sum(Path(p).parent.name == "normal" for p in test_files)
            n_abnormal = len(test_files) - n_normal
            print(f"{snr}: train={n_train}, val={n_val}, "f"test={len(test_files)} "f"(normal={n_normal}, abnormal={n_abnormal})")
        print(f"Training totale: {len(self.train_list)}")
        print(f"Validation totale: {len(self.val_list)}")

    def select_test_snr(self, snr):
        snr = str(snr)
        if snr not in self.test_lists:
            raise ValueError(f"SNR {snr} non disponibile; opzioni: {list(self.test_lists)}")
        self.test_snr = snr

    def train_dataloader(self):
        return DataLoader(self._dataset(self.train_list, augment=self.spatial_augmentation),batch_size=self.batch_size,shuffle=True,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self._dataset(self.val_list),batch_size=self.batch_size,shuffle=False)

    def test_dataloader(self):
        return DataLoader(self._dataset(self.test_lists[self.test_snr]),batch_size=self.batch_size,shuffle=False)
