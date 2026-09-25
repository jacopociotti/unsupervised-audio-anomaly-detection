import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import librosa as lb
import numpy as np
import os
import random
from pathlib import Path

TUT_LABELS = ["fan", "pump", "slider", "ToyCar", "ToyConveyor", "valve"]
TUT_LABELS_INT = [0, 7, 13, 20, 27, 34]

MIMII_MACHINE_TYPES = {
    "fan": 0,
    "pump": 4,
    "slider": 8,
    "valve": 12,
}

MIMII_IDS = {
    "00": 0,
    "02": 1,
    "04": 2,
    "06": 3,
}

MIMII_CHANNEL_SELECTIONS = {
    1: [0],
    4: [0, 2, 4, 6],
    8: [0, 1, 2, 3, 4, 5, 6, 7],
}

class TUTDataset(Dataset):

    def __init__(self, path_data, list_files, sample_rate, duration):
        # Initialization for path, fs, list of files etc. 
        self.sample_rate = sample_rate
        self.path_data = path_data
        self.list_files = list_files
        self.duration = duration

    def __getitem__(self, index):
        # select an audio from the list and return metadata and label
        file_name = self.list_files[index]
        if ".npy" in file_name:
            audio_data = np.load(file_name)
        else:
            audio_data , _ = lb.load(file_name, sr = self.sample_rate, res_type = "polyphase")
        if len(audio_data) > int(self.duration * self.sample_rate):
            audio_data = audio_data[:int(self.duration * self.sample_rate)]

        metadata = 0
        numerical_label = 0
        for i, name_class in enumerate(TUT_LABELS):
            if name_class in file_name:
                metadata = TUT_LABELS_INT[i]
                numerical_label = TUT_LABELS_INT[i]
                break

        metadata += int(file_name.split('_')[2])  # in this way I have "name_id", for example fan_01 where 01 is the id of the machine
        label = 0 if "normal" in file_name else 1
        return audio_data , metadata, label, numerical_label
    
    def __len__(self):
        # return length of the list
        return len(self.list_files)

# Nuova classe per il dataset MIMII, che gestisce i file audio con più canali e assegna etichette basate sulla presenza di "normal" nel nome del file.
class MIMIIDataset(Dataset):

    def __init__(self, list_files, sample_rate=16000, duration=10, num_channels=1, spatial_augmentation=False):
        self.list_files = list_files
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_channels = num_channels
        self.spatial_augmentation = spatial_augmentation
        if self.num_channels not in MIMII_CHANNEL_SELECTIONS:
            raise ValueError(f"num_channels deve essere 1, 4 oppure 8, ricevuto: {self.num_channels}")

        self.channel_indices = MIMII_CHANNEL_SELECTIONS[self.num_channels]

    def __getitem__(self, index):
        file_name = self.list_files[index]

        # Carica tutti gli 8 canali del file MIMII
        audio_data, _ = lb.load(file_name, sr=self.sample_rate, mono=False, res_type="polyphase")

        # Limita la durata
        max_samples = int(self.duration * self.sample_rate)
        audio_data = audio_data[:, :max_samples]

        # Seleziona i canali desiderati
        audio_data = audio_data[self.channel_indices, :]

        # Spatial augmentation: random cyclic rotation dei microfoni
        # Solo durante il training e solo per configurazioni multicanale.
        if self.spatial_augmentation and self.num_channels > 1:
            shift = np.random.randint(0, self.num_channels)
            audio_data = np.roll(audio_data, shift=shift, axis=0)

        if self.num_channels == 1:
            audio_data = audio_data[0]

        # Normal = 0, abnormal = 1
        condition = os.path.basename(os.path.dirname(file_name))

        if condition == "normal":
            label = 0
        elif condition == "abnormal":
            label = 1
        else:
            raise ValueError(f"Condizione non riconosciuta nel percorso: {file_name}")
        
        path = Path(file_name)

        machine_type = path.parent.parent.parent.name
        machine_id_folder = path.parent.parent.name

        if not machine_id_folder.startswith("id_"):
            raise ValueError(f"Machine ID non riconosciuto: {machine_id_folder}")

        machine_id = machine_id_folder.replace("id_", "")

        if machine_type not in MIMII_MACHINE_TYPES:
            raise ValueError(f"Machine type non riconosciuto: {machine_type}")

        if machine_id not in MIMII_IDS:
            raise ValueError(f"Machine ID non riconosciuto: {machine_id}")

        numerical_label = MIMII_MACHINE_TYPES[machine_type]
        metadata = numerical_label + MIMII_IDS[machine_id]

        return audio_data, metadata, label, numerical_label

    def __len__(self):
        return len(self.list_files)

class MIMIIDataModule(LightningDataModule):

    def __init__(self, path_data, sample_rate=16000, duration=10, batch_size=64, num_channels=1, spatial_augmentation=False):
        super().__init__()

        self.path_data = Path(path_data)
        self.sample_rate = sample_rate
        self.duration = duration
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.spatial_augmentation = spatial_augmentation

        self.train_list = []
        self.val_list = []
        self.test_list = []

        self._create_splits()

    def _create_splits(self):

        # Cerca tutte le cartelle id_XX presenti nel dataset
        machine_id_dirs = sorted(self.path_data.glob("*/*"))

        for machine_id_dir in machine_id_dirs:

            if not machine_id_dir.is_dir():
                continue

            if not machine_id_dir.name.startswith("id_"):
                continue

            normal_dir = machine_id_dir / "normal"
            abnormal_dir = machine_id_dir / "abnormal"

            normal_files = sorted(normal_dir.glob("*.wav"))
            abnormal_files = sorted(abnormal_dir.glob("*.wav"))

            n_normal = len(normal_files)

            n_train = int(0.70 * n_normal)
            n_val = int(0.10 * n_normal)

            # 70% normal -> training
            train_files = normal_files[:n_train]

            # 10% normal -> validation
            val_files = normal_files[n_train:n_train + n_val]

            # restante 20% normal -> test
            test_normal_files = normal_files[n_train + n_val:]

            # tutti gli abnormal -> test
            test_files = test_normal_files + abnormal_files

            self.train_list.extend(str(f) for f in train_files)
            self.val_list.extend(str(f) for f in val_files)
            self.test_list.extend(str(f) for f in test_files)

    def setup(self, stage=None):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self):
        dataset = MIMIIDataset(
            self.train_list,
            sample_rate=self.sample_rate,
            duration=self.duration,
            num_channels=self.num_channels,
            spatial_augmentation=self.spatial_augmentation
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        dataset = MIMIIDataset(
            self.val_list,
            sample_rate=self.sample_rate,
            duration=self.duration,
            num_channels=self.num_channels,
            spatial_augmentation=False
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        dataset = MIMIIDataset(
            self.test_list,
            sample_rate=self.sample_rate,
            duration=self.duration,
            num_channels=self.num_channels,
            spatial_augmentation=False
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

class TUTDatamodule(LightningDataModule):

    def __init__(self, path_train, path_test, sample_rate, duration, percentage_val, batch_size):
        super().__init__()
        self.path_train = path_train
        self.path_test = path_test
        self.sample_rate = sample_rate
        self.duration = duration
        self.percentage_val = percentage_val
        self.batch_size = batch_size
        # split of the dataset
        self.train_list = self.scan_all_dir(self.path_train)
        self.val_list = random.sample(self.train_list, int(len(self.train_list) * percentage_val))
        self.train_list = set(self.train_list)
        self.val_list = set(self.val_list)
        self.train_list -= self.val_list
        self.train_list = list(self.train_list)
        self.val_list = list(self.val_list)

        self.test_list = self.scan_all_dir(self.path_test)


    def scan_all_dir(self, path):
        list_all_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                list_all_files.append(str(root + "\\" + file))
        return list_all_files

    def setup(self, stage = None):
        # Nothing to do
        pass

    def prepare_data(self):
        # Nothing to do
        pass
    
    def train_dataloader(self):
        # return the dataloader containing training data
        train_split = TUTDataset(path_data = self.path_train, list_files = self.train_list,  sample_rate = self.sample_rate, duration = self.duration)
        return DataLoader(train_split, batch_size = self.batch_size, shuffle = True)
    
    def val_dataloader(self):
        # return the dataloader containing validation data
        val_split = TUTDataset(path_data = self.path_train, list_files =  self.val_list, sample_rate = self.sample_rate, duration = self.duration)
        return DataLoader(val_split, batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        # return the dataloader containing testing data
        test_split = TUTDataset(path_data = self.path_test, list_files = self.test_list, sample_rate = self.sample_rate, duration = self.duration)
        return DataLoader(test_split, batch_size = self.batch_size, shuffle = True)
    
## TEST FUNCTION ##
if __name__ == "__main__":
    path_train = "TUT Anomaly detection/train" # path for training audio
    path_test = "TUT Anomaly detection/test" # path for test audio
    percentage_val = 0.2
    sample_rate = 16000
    batch_size = 64
    duration = 10
    # create lightning datamodule
    datamodule = TUTDatamodule(path_train = path_train, path_test = path_test, sample_rate = sample_rate, duration = duration,
                                percentage_val = percentage_val, batch_size = batch_size)
    dataloader_train = datamodule.train_dataloader()
    print(len(dataloader_train))
    dataloader_val = datamodule.val_dataloader()
    print(len(dataloader_val))
    dataloader_test = datamodule.test_dataloader()
    print(len(dataloader_test))
    print(next(iter(dataloader_test)))