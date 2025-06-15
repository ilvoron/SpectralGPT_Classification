import os
import pandas as pd
import numpy as np
import warnings
import random
from glob import glob
import json
from typing import Any, Optional, List
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from rasterio import logging
from pathlib import Path
import skimage.io as io

log = logging.getLogger()
log.setLevel(logging.ERROR)

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

CATEGORIES = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
              "aquaculture", "archaeological_site", "barn", "border_checkpoint",
              "burial_site", "car_dealership", "construction_site", "crop_field",
              "dam", "debris_or_rubble", "educational_institution", "electric_substation",
              "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
              "gas_station", "golf_course", "ground_transportation_station", "helipad",
              "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
              "lighthouse", "military_facility", "multi-unit_residential",
              "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
              "parking_lot_or_garage", "place_of_worship", "police_station", "port",
              "prison", "race_track", "railway_bridge", "recreational_facility",
              "road_bridge", "runway", "shipyard", "shopping_mall",
              "single-unit_residential", "smokestack", "solar_farm", "space_facility",
              "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
              "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
              "wind_farm", "zoo"]

NEW_LABELS = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland/shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters'
]

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}


class SatelliteDataset(Dataset):
    """
    Абстрактный класс.
    """

    def __init__(self, in_c):
        self.in_c = in_c

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        """
        Создаёт преобразования данных для обучения/валидации для данного класса датасета.
        :param is_train: Использовать ли преобразования для обучения или для валидации.
        :param input_size: Размер входного изображения (предполагается квадратное изображение).
        :param mean: Среднее значение по каналам, форма (c,) для c каналов.
        :param std: Стандартное отклонение по каналам, форма (c,).
        :return: Torch-преобразование для входного изображения перед подачей в модель.
        """

        # преобразования для обучения
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 — бикубическая интерполяция
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # преобразования для валидации
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # чтобы сохранить то же соотношение, что и для изображений 224
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class CustomDatasetFromImages(SatelliteDataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Создаёт датасет для обычной классификации RGB-изображений (обычно используется для fMoW-RGB).
        :param csv_path: путь к csv-файлу.
        :param transform: pytorch-преобразования для обработки и преобразования в тензор.
        """
        super().__init__(in_c=3)
        # Преобразования
        self.transforms = transform
        # Чтение csv-файла
        self.data_info = pd.read_csv(csv_path, header=0)
        # Первый столбец содержит пути к изображениям
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Второй столбец — метки классов
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Длина датасета
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Получить имя изображения из pandas DataFrame
        single_image_name = self.image_arr[index]
        # Открыть изображение
        img_as_img = Image.open(single_image_name)
        # Преобразовать изображение
        img_as_tensor = self.transforms(img_as_img)
        # Получить метку класса изображения
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


class CustomDatasetFromImagesTemporal(SatelliteDataset):
    def __init__(self, csv_path: str):
        """
        Создаёт временной датасет для fMoW RGB
        :param csv_path: Путь к csv-файлу, содержащему пути к изображениям
        :param meta_csv_path: Путь к csv-файлу с метаданными для каждого изображения
        """
        super().__init__(in_c=3)

        # Преобразования
        self.transforms = transforms.Compose([transforms.RandomCrop(224)])
        # Чтение csv-файла
        self.data_info = pd.read_csv(csv_path, header=0)
        # Первый столбец содержит пути к изображениям
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Второй столбец — метки классов
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Длина датасета
        self.data_len = len(self.data_info)

        self.dataset_root_path = os.path.dirname(csv_path)

        self.timestamp_arr = np.asarray(self.data_info.iloc[:, 2])
        self.name2index = dict(zip(
            [os.path.join(self.dataset_root_path, x) for x in self.image_arr],
            np.arange(self.data_len)
        ))

        self.min_year = 2002  # захардкожено для fMoW

        mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
        std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]
        self.normalization = transforms.Normalize(mean, std)
        self.totensor = transforms.ToTensor()
        self.scale = transforms.Scale(224)

    def __getitem__(self, index):
        # Получить имя изображения из pandas DataFrame
        single_image_name_1 = self.image_arr[index]

        suffix = single_image_name_1[-15:]
        prefix = single_image_name_1[:-15].rsplit('_', 1)
        regexp = '{}_*{}'.format(prefix[0], suffix)
        regexp = os.path.join(self.dataset_root_path, regexp)
        single_image_name_1 = os.path.join(self.dataset_root_path, single_image_name_1)
        temporal_files = glob(regexp)

        temporal_files.remove(single_image_name_1)
        if temporal_files == []:
            single_image_name_2 = single_image_name_1
            single_image_name_3 = single_image_name_1
        elif len(temporal_files) == 1:
            single_image_name_2 = temporal_files[0]
            single_image_name_3 = temporal_files[0]
        else:
            single_image_name_2 = random.choice(temporal_files)
            while True:
                single_image_name_3 = random.choice(temporal_files)
                if single_image_name_3 != single_image_name_2:
                    break

        img_as_img_1 = Image.open(single_image_name_1)
        img_as_img_2 = Image.open(single_image_name_2)
        img_as_img_3 = Image.open(single_image_name_3)
        img_as_tensor_1 = self.totensor(img_as_img_1)
        img_as_tensor_2 = self.totensor(img_as_img_2)
        img_as_tensor_3 = self.totensor(img_as_img_3)
        del img_as_img_1
        del img_as_img_2
        del img_as_img_3
        img_as_tensor_1 = self.scale(img_as_tensor_1)
        img_as_tensor_2 = self.scale(img_as_tensor_2)
        img_as_tensor_3 = self.scale(img_as_tensor_3)
        try:
            if img_as_tensor_1.shape[2] > 224 and img_as_tensor_2.shape[2] > 224 and img_as_tensor_3.shape[2] > 224:
                min_w = min(img_as_tensor_1.shape[2], min(img_as_tensor_2.shape[2], img_as_tensor_3.shape[2]))
                img_as_tensor = torch.cat([img_as_tensor_1[..., :min_w], img_as_tensor_2[..., :min_w], img_as_tensor_3[..., :min_w]], dim=-3)
            elif img_as_tensor_1.shape[1] > 224 and img_as_tensor_2.shape[1] > 224 and img_as_tensor_3.shape[1] > 224:
                min_w = min(img_as_tensor_1.shape[1], min(img_as_tensor_2.shape[1], img_as_tensor_3.shape[1]))
                img_as_tensor = torch.cat([img_as_tensor_1[..., :min_w, :], img_as_tensor_2[..., :min_w, :], img_as_tensor_3[..., :min_w, :]], dim=-3)
            else:
                img_as_img_1 = Image.open(single_image_name_1)
                img_as_tensor_1 = self.totensor(img_as_img_1)
                img_as_tensor_1 = self.scale(img_as_tensor_1)
                img_as_tensor = torch.cat([img_as_tensor_1, img_as_tensor_1, img_as_tensor_1], dim=-3)
        except:
            print(img_as_tensor_1.shape, img_as_tensor_2.shape, img_as_tensor_3.shape)
            assert False

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        img_as_tensor = self.transforms(img_as_tensor)
        img_as_tensor_1, img_as_tensor_2, img_as_tensor_3 = torch.chunk(img_as_tensor, 3, dim=-3)
        del img_as_tensor
        img_as_tensor_1 = self.normalization(img_as_tensor_1)
        img_as_tensor_2 = self.normalization(img_as_tensor_2)
        img_as_tensor_3 = self.normalization(img_as_tensor_3)

        ts1 = self.parse_timestamp(single_image_name_1)
        ts2 = self.parse_timestamp(single_image_name_2)
        ts3 = self.parse_timestamp(single_image_name_3)

        ts = np.stack([ts1, ts2, ts3], axis=0)

        # Получить метку класса изображения по обрезанному столбцу pandas
        single_image_label = self.label_arr[index]

        imgs = torch.stack([img_as_tensor_1, img_as_tensor_2, img_as_tensor_3], dim=0)

        del img_as_tensor_1
        del img_as_tensor_2
        del img_as_tensor_3

        return (imgs, ts, single_image_label)

    def parse_timestamp(self, name):
        timestamp = self.timestamp_arr[self.name2index[name]]
        year = int(timestamp[:4])
        month = int(timestamp[5:7])
        hour = int(timestamp[11:13])
        return np.array([year - self.min_year, month - 1, hour])

    def __len__(self):
        return self.data_len


class SentinelNormalize:
    """
    Нормализация для изображений Sentinel-2
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x, *args, **kwargs):
        min_value = self.mean - 2 * self.std
        max_value = self.mean + 2 * self.std
        img = (x - min_value) / (max_value - min_value) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class SentinelIndividualImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 years: Optional[List[int]] = [*range(2000, 2021)],
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Создаёт датасет для мультиспектральной классификации одиночных изображений.
        Обычно используется для fMoW-Sentinel.
        :param csv_path: путь к csv-файлу.
        :param transform: pytorch-преобразование для обработки и преобразования в тензор
        :param years: список лет, из которых брать изображения, None — не фильтровать
        :param categories: список категорий, из которых брать изображения, None — не фильтровать
        :param label_type: 'values' — одиночная метка, 'one-hot' — one-hot метки
        :param masked_bands: список индексов каналов, которые нужно замаскировать
        :param dropped_bands: список индексов каналов, которые нужно удалить из входного тензора изображения
        """
        super().__init__(in_c=13)
        self.df = pd.read_csv(csv_path).sort_values(['category', 'location_id', 'timestamp'])

        # Фильтрация по категориям
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        # Фильтрация по годам
        if years is not None:
            self.df['year'] = [int(timestamp.split('-')[0]) for timestamp in self.df['timestamp']]
            self.df = self.df[self.df['year'].isin(years)]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(f'FMOWDataset label_type {label_type} не поддерживается. label_type должен быть одним из:',
                             ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        img = io.imread(img_path)
        kid = (img - img.min(axis=(0, 1), keepdims=True))
        mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
        img = kid / (mom + 1e-10)

        return img.astype(np.float32)

    def __getitem__(self, idx):
        """
        Получает пару (изображение, метка) по индексу в датасете.
        :param idx: индекс пары (изображение, метка) в датафрейме. (c, h, w)
        :return: тензор изображения Torch и целочисленная метка в кортеже.
        """
        selection = self.df.iloc[idx]

        images = self.open_image('/home/ps/Documents/data/' + selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
            'image_ids': selection['image_id'],
            'timestamps': selection['timestamp']
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # Преобразования для обучения
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 — бикубическая интерполяция
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # Преобразования для валидации
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # чтобы сохранить то же соотношение, что и для изображений 224
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


def get_multihot_new(labels):
    target = np.zeros((len(NEW_LABELS),), dtype=np.float32)
    for label in labels:
        if label in GROUP_LABELS:
            target[NEW_LABELS.index(GROUP_LABELS[label])] = 1
        elif label not in set(NEW_LABELS):
            continue
        else:
            target[NEW_LABELS.index(label)] = 1
    return target


class BigEarthNetFinetuneDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Создаёт датасет для мультиспектральной классификации одиночных изображений.
        Обычно используется для fMoW-Sentinel.
        :csv_path: txt-файл с основной информацией о датасете
        :param transform: pytorch-преобразование для обработки и преобразования в тензор
        :param masked_bands: список индексов каналов, которые нужно замаскировать
        :param dropped_bands: список индексов каналов, которые нужно удалить из входного тензора изображения
        """
        super().__init__(in_c=12)

        self.transform = transform
        self.get_multihot_new = get_multihot_new
        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)
        self.image_root = Path('data/BE_cor')

        self.samples = []
        with open(csv_path) as f:
            for patch_id in f.read().splitlines():
                self.samples.append(self.image_root / patch_id)

    def __len__(self):
        return len(self.samples)

    def open_image(self, img_path):
        img = io.imread(img_path)
        return img.astype(np.float32)

    def __getitem__(self, index):
        """
        Получает пару (изображение, метка) по индексу в датасете.
        :param idx: Индекс пары (изображение, метка) в датасете. (c, h, w)
        :return: Тензор изображения Torch и метка (multi-hot) в кортеже.
        """
        path = self.samples[index]
        patch_id = path.name  # S2A_MSIL2A_20180527T093041_25_79

        images = self.open_image(path / f'{patch_id}.tif')

        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        with open(path / f'{patch_id}_labels_metadata.json', 'r') as f:
            labels = json.load(f)['labels']
        targets = self.get_multihot_new(labels)

        img_as_tensor = self.transform(images)  # (c, h, w)

        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': targets,
        }
        return img_as_tensor, targets

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # Преобразования для обучения
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(SentinelNormalize(mean, std))  # используем специфичную нормализацию Sentinel, чтобы избежать NaN
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 — бикубическая интерполяция
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # Преобразования для валидации
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(SentinelNormalize(mean, std))
        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # чтобы сохранить то же соотношение, что и для изображений 224
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class BigEarthNetImageDataset(SatelliteDataset):
    label_types = ['value', 'one-hot']
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 1310.36996126, 1087.6020813]

    def __init__(self,
                 csv_path: str,
                 transform: Any,
                 categories: Optional[List[str]] = None,
                 label_type: str = 'value',
                 masked_bands: Optional[List[int]] = None,
                 dropped_bands: Optional[List[int]] = None):
        """
        Создаёт датасет для мультиспектральной классификации одиночных изображений.
        Обычно используется для датасета fMoW-Sentinel.
        :param csv_path: путь к csv-файлу.
        :param transform: pytorch-преобразование для обработки и преобразования в тензор
        :param years: список лет, из которых брать изображения, None — не фильтровать
        :param categories: список категорий, из которых брать изображения, None — не фильтровать
        :param label_type: 'values' — одиночная метка, 'one-hot' — one-hot метки
        :param masked_bands: список индексов каналов, которые нужно замаскировать
        :param dropped_bands: список индексов каналов, которые нужно удалить из входного тензора изображения
        """
        super().__init__(in_c=12)
        self.df = pd.read_csv(csv_path).sort_values(['category'])

        # Фильтрация по категориям
        self.categories = CATEGORIES
        if categories is not None:
            self.categories = categories
            self.df = self.df.loc[categories]

        self.indices = self.df.index.unique().to_numpy()

        self.transform = transform

        if label_type not in self.label_types:
            raise ValueError(f'BigEarthNet label_type {label_type} не поддерживается. label_type должен быть одним из:',
                             ', '.join(self.label_types))
        self.label_type = label_type

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.df)

    def open_image(self, img_path):
        img = io.imread(img_path)
        kid = (img - img.min(axis=(0, 1), keepdims=True))
        mom = (img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True))
        img = kid / (mom + 1e-12)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        """
        Получает пару (изображение, метка) по индексу в датасете.
        :param idx: индекс пары (изображение, метка) в датафрейме. (c, h, w)
        :return: тензор изображения Torch и целочисленная метка в кортеже.
        """
        selection = self.df.iloc[idx]

        images = self.open_image('/home/ps/Documents/data/bigearthnet_cor/' + selection['image_path'])  # (h, w, c)
        if self.masked_bands is not None:
            images[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        labels = self.categories.index(selection['category'])

        img_as_tensor = self.transform(images)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        sample = {
            'images': images,
            'labels': labels,
        }
        return img_as_tensor, labels

    @staticmethod
    def build_transform(is_train, input_size, mean, std):
        # Преобразования для обучения
        interpol_mode = transforms.InterpolationMode.BICUBIC

        t = []
        if is_train:
            t.append(transforms.ToTensor())
            t.append(
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=interpol_mode),  # 3 — бикубическая интерполяция
            )
            t.append(transforms.RandomHorizontalFlip())
            return transforms.Compose(t)

        # Преобразования для валидации
        if input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(input_size / crop_pct)

        t.append(transforms.ToTensor())
        t.append(
            transforms.Resize(size, interpolation=interpol_mode),  # чтобы сохранить то же соотношение, что и для изображений 224
        )
        t.append(transforms.CenterCrop(input_size))

        return transforms.Compose(t)


class EuroSat(SatelliteDataset):
    mean = [752.40087073, 884.29673756, 1144.16202635, 1297.47289228, 1624.90992062, 2194.6423161, 2422.21248945,
            2517.76053101, 2581.64687018, 2645.51888987, 582.72633433, 2368.51236873, 1805.06846033]

    std = [1108.02887453, 1155.15170768, 1183.6292542, 1368.11351514, 1370.265037, 1355.55390699, 1416.51487101,
           1474.78900051, 1439.3086061, 1582.28010962, 472.37967789, 1455.52084939, 1343.48379601]

    def __init__(self, file_path, transform, masked_bands=None, dropped_bands=None):
        """
        Создаёт датасет для мультиспектральной классификации одиночных изображений EuroSAT.
        :param file_path: путь к txt-файлу, содержащему пути к изображениям EuroSAT.
        :param transform: pytorch-преобразование для обработки и преобразования в тензор
        :param masked_bands: список индексов каналов, которые нужно замаскировать
        :param dropped_bands: список индексов каналов, которые нужно удалить из входного тензора изображения
        """
        super().__init__(13)
        with open(file_path, 'r') as f:
            data = f.read().splitlines()
        self.img_paths = [row.split()[0] for row in data]
        self.labels = [int(row.split()[1]) for row in data]

        self.transform = transform

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        if self.dropped_bands is not None:
            self.in_c = self.in_c - len(dropped_bands)

    def __len__(self):
        return len(self.img_paths)

    def open_image(self, img_path):
        img = io.imread(img_path)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        img = self.open_image(img_path)  # (h, w, c)
        if self.masked_bands is not None:
            img[:, :, self.masked_bands] = np.array(self.mean)[self.masked_bands]

        img_as_tensor = self.transform(img)  # (c, h, w)
        if self.dropped_bands is not None:
            keep_idxs = [i for i in range(img_as_tensor.shape[0]) if i not in self.dropped_bands]
            img_as_tensor = img_as_tensor[keep_idxs, :, :]

        return img_as_tensor, label


def build_fmow_dataset(is_train: bool, args) -> SatelliteDataset:
    """
    Инициализирует объект SatelliteDataset на основе переданных аргументов.
    :param is_train: Использовать ли датасет для обучения или для валидации
    :param args: Объект аргументов Argparser с необходимыми параметрами
    :return: Объект SatelliteDataset.
    """
    csv_path = os.path.join(args.train_path if is_train else args.test_path)

    if args.dataset_type == 'rgb':
        mean = CustomDatasetFromImages.mean
        std = CustomDatasetFromImages.std
        transform = CustomDatasetFromImages.build_transform(is_train, args.input_size, mean, std)
        dataset = CustomDatasetFromImages(csv_path, transform)

    elif args.dataset_type == 'sentinel':
        mean = SentinelIndividualImageDataset.mean
        std = SentinelIndividualImageDataset.std
        transform = SentinelIndividualImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = SentinelIndividualImageDataset(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)

    elif args.dataset_type == 'bigearthnet':
        mean = BigEarthNetImageDataset.mean
        std = BigEarthNetImageDataset.std
        transform = BigEarthNetImageDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = BigEarthNetImageDataset(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)
    elif args.dataset_type == 'bigearthnet_finetune':
        mean = BigEarthNetFinetuneDataset.mean
        std = BigEarthNetFinetuneDataset.std
        transform = BigEarthNetFinetuneDataset.build_transform(is_train, args.input_size, mean, std)
        dataset = BigEarthNetFinetuneDataset(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)

    elif args.dataset_type == 'euro_sat':
        mean, std = EuroSat.mean, EuroSat.std
        transform = EuroSat.build_transform(is_train, args.input_size, mean, std)
        dataset = EuroSat(csv_path, transform, masked_bands=args.masked_bands, dropped_bands=args.dropped_bands)
    else:
        raise ValueError(f"Некорректный тип датасета: {args.dataset_type}")
    print(dataset)

    return dataset
