import os
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 10 channels : B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12
class BurnSeverityDataset(Dataset):
    def __init__(self, data_dir, dataset_ids, transform=None):
        self.image_dir = os.path.join(data_dir, 'image')
        self.label_dir = os.path.join(data_dir, 'label')
        self.dataset_ids = dataset_ids
        self.transform = transform

        self.before_files = []
        self.after_files = []
        self.label_files = []

        for dataset_id in self.dataset_ids:
            before_file = os.path.join(self.image_dir, 'before', dataset_id)
            after_file = os.path.join(self.image_dir, 'after', dataset_id)
            label_file = os.path.join(self.label_dir, dataset_id)

            if os.path.exists(before_file) and os.path.exists(after_file) and os.path.exists(label_file):
                self.before_files.append(before_file)
                self.after_files.append(after_file)
                self.label_files.append(label_file)
            else:
                print(f"Warning: Missing files for dataset ID {dataset_id}")
                if not os.path.exists(before_file):
                    print(f"  Before file not found: {before_file}")
                if not os.path.exists(after_file):
                    print(f"  After file not found: {after_file}")
                if not os.path.exists(label_file):
                    print(f"  Label file not found: {label_file}")

        if len(self.before_files) == 0:
            raise ValueError(f"No image files found for the given dataset IDs in {self.image_dir}")

    def __len__(self):
        return len(self.before_files)

    def __getitem__(self, idx):
        before_path = self.before_files[idx]
        after_path = self.after_files[idx]
        label_path = self.label_files[idx]

        before_img = self.load_image(before_path)
        after_img = self.load_image(after_path)
        label = self.load_label(label_path)

        if self.transform:
            augmented = self.transform(image=before_img, image2=after_img, mask=label)
            before_img = augmented['image']
            after_img = augmented['image2']
            label = augmented['mask']

        filename = os.path.basename(before_path)
        return before_img, after_img, label, filename

    def load_image(self, path):
        with rasterio.open(path) as src:
            # ===== for 10 bands (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12) =====
            # img = src.read().astype(np.float32)
            # ===== for 7 bands (B8, B5, B6, B7, B8A, B11, B12) =====
            # img = src.read([4,5,6,7,8,9,10]).astype(np.float32)
            # ===== for 6 bands (B2, B3, B4, B8, B11, B12) =====
            # img = src.read([1,2,3,4,9,10]).astype(np.float32)
            # ===== for 3 bands(B8, B11, B12) =====
            img = src.read([4,9,10]).astype(np.float32)
            img = np.transpose(img, (1, 2, 0))  # CxHxW -> HxWxC
        return img

    def load_label(self, path):
        with rasterio.open(path) as src:
            label = src.read(1).astype(np.int64)
        return label


def get_datasets(data_path, train_ids, val_ids, test_ids):
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05)),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
        ], p=0.1),
        # ===== for 10 bands (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12) =====
        # A.Normalize(mean=[947.5751907285983, 1195.8134586828469, 1324.2515813207974, 2645.19185232246, 1685.124434004735,
        #                   2309.4578441981853, 2546.6942364630036, 2752.675718307495, 2744.001138373883, 2119.2794981281254],
        #             std=[603.082063160469, 630.1817554199854, 767.4005822709588, 934.995081474556, 745.7027775136748,
        #                  813.1179193014578, 893.4360453565056, 936.9873530024242, 989.8159677271037, 924.8957662027137]),
        # ===== for 7 bands (B8, B5, B6, B7, B8A, B11, B12) =====
        # A.Normalize(mean=[2645.19185232246, 1685.124434004735, 2309.4578441981853, 2546.6942364630036,
        #                   2752.675718307495, 2744.001138373883, 2119.2794981281254],
        #             std=[934.995081474556, 745.7027775136748, 813.1179193014578, 893.4360453565056,
        #                  936.9873530024242, 989.8159677271037, 924.8957662027137]),
        # ===== for 6 bands (B2, B3, B4, B8, B11, B12) =====
        # A.Normalize(mean=[947.5751907285983, 1195.8134586828469, 1324.2515813207974, 2645.19185232246,
        #                   2744.001138373883, 2119.2794981281254],
        #             std=[603.082063160469, 630.1817554199854, 767.4005822709588, 934.995081474556,
        #                  989.8159677271037, 924.8957662027137]),
        # ===== for 3 bands(B8, B11, B12) =====
        A.Normalize(mean=[2645.19185232246, 2744.001138373883, 2119.2794981281254],
                    std=[934.995081474556, 989.8159677271037, 924.8957662027137]),
        ToTensorV2()
    ], additional_targets={'image2': 'image'})

    val_transform = A.Compose([
        A.Resize(256, 256),
        # ===== for 10 bands (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12) =====
        # A.Normalize(mean=[947.5751907285983, 1195.8134586828469, 1324.2515813207974, 2645.19185232246, 1685.124434004735,
        #                   2309.4578441981853, 2546.6942364630036, 2752.675718307495, 2744.001138373883, 2119.2794981281254],
        #             std=[603.082063160469, 630.1817554199854, 767.4005822709588, 934.995081474556, 745.7027775136748,
        #                  813.1179193014578, 893.4360453565056, 936.9873530024242, 989.8159677271037, 924.8957662027137]),
        # ===== for 7 bands (B8, B5, B6, B7, B8A, B11, B12) =====
        # A.Normalize(mean=[2645.19185232246, 1685.124434004735, 2309.4578441981853, 2546.6942364630036,
        #                   2752.675718307495, 2744.001138373883, 2119.2794981281254],
        #             std=[934.995081474556, 745.7027775136748, 813.1179193014578, 893.4360453565056,
        #                  936.9873530024242, 989.8159677271037, 924.8957662027137]),
        # ===== for 6 bands (B2, B3, B4, B8, B11, B12) =====
        # A.Normalize(mean=[947.5751907285983, 1195.8134586828469, 1324.2515813207974, 2645.19185232246,
        #                   2744.001138373883, 2119.2794981281254],
        #             std=[603.082063160469, 630.1817554199854, 767.4005822709588, 934.995081474556,
        #                  989.8159677271037, 924.8957662027137]),
        # ===== for 3 bands(B8, B11, B12) =====
        A.Normalize(mean=[2645.19185232246, 2744.001138373883, 2119.2794981281254],
                    std=[934.995081474556, 989.8159677271037, 924.8957662027137]),
        ToTensorV2()
    ], additional_targets={'image2': 'image'})

    test_transform = A.Compose([
        A.Resize(256, 256),
        # ===== for 10 bands (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12) =====
        # A.Normalize(mean=[947.5751907285983, 1195.8134586828469, 1324.2515813207974, 2645.19185232246, 1685.124434004735,
        #                   2309.4578441981853, 2546.6942364630036, 2752.675718307495, 2744.001138373883, 2119.2794981281254],
        #             std=[603.082063160469, 630.1817554199854, 767.4005822709588, 934.995081474556, 745.7027775136748,
        #                  813.1179193014578, 893.4360453565056, 936.9873530024242, 989.8159677271037, 924.8957662027137]),
        # ===== for 7 bands (B8, B5, B6, B7, B8A, B11, B12) =====
        # A.Normalize(mean=[2645.19185232246, 1685.124434004735, 2309.4578441981853, 2546.6942364630036,
        #                   2752.675718307495, 2744.001138373883, 2119.2794981281254],
        #             std=[934.995081474556, 745.7027775136748, 813.1179193014578, 893.4360453565056,
        #                  936.9873530024242, 989.8159677271037, 924.8957662027137]),
        # ===== for 6 bands (B2, B3, B4, B8, B11, B12) =====
        # A.Normalize(mean=[947.5751907285983, 1195.8134586828469, 1324.2515813207974, 2645.19185232246,
        #                   2744.001138373883, 2119.2794981281254],
        #             std=[603.082063160469, 630.1817554199854, 767.4005822709588, 934.995081474556,
        #                  989.8159677271037, 924.8957662027137]),
        # ===== for 3 bands(B8, B11, B12) =====
        A.Normalize(mean=[2645.19185232246, 2744.001138373883, 2119.2794981281254],
                    std=[934.995081474556, 989.8159677271037, 924.8957662027137]),
        ToTensorV2()
    ], additional_targets={'image2': 'image'})

    train_dataset = BurnSeverityDataset(data_path, train_ids, transform=train_transform)
    val_dataset = BurnSeverityDataset(data_path, val_ids, transform=val_transform)
    test_dataset = BurnSeverityDataset(data_path, test_ids, transform=test_transform)

    return train_dataset, val_dataset, test_dataset