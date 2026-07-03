import os
import shutil
import random
from pathlib import Path
import tensorflow as tf

RAW_DATA_DIR = Path('metal_nut/metal_nut')
PROCESSED_DATA_DIR = Path('data_processed')

def reorganize_dataset(split_ratio=0.8):

    if PROCESSED_DATA_DIR.exists():
        print(f"Directory {PROCESSED_DATA_DIR} already exists. Skipping reorganization.")
        return

    print("Reorganizing dataset for supervised binary classification...")
    

    for split in ['train', 'val']:
        for class_name in ['good', 'defect']:
            (PROCESSED_DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    raw_train_good = list((RAW_DATA_DIR / 'train' / 'good').glob('*.png'))
    raw_test_good = list((RAW_DATA_DIR / 'test' / 'good').glob('*.png'))
    all_good = raw_train_good + raw_test_good
    
    random.shuffle(all_good)
    split_idx = int(len(all_good) * split_ratio)
    train_good = all_good[:split_idx]
    val_good = all_good[split_idx:]

    for img_path in train_good:
        new_name = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.name}"
        shutil.copy(img_path, PROCESSED_DATA_DIR / 'train' / 'good' / new_name)
    for img_path in val_good:
        new_name = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.name}"
        shutil.copy(img_path, PROCESSED_DATA_DIR / 'val' / 'good' / new_name)

    all_defect = []
    test_dir = RAW_DATA_DIR / 'test'
    if test_dir.exists():
        for defect_type_dir in test_dir.iterdir():
            if defect_type_dir.is_dir() and defect_type_dir.name != 'good':
                all_defect.extend(list(defect_type_dir.glob('*.png')))
                
    random.shuffle(all_defect)
    split_idx = int(len(all_defect) * split_ratio)
    train_defect = all_defect[:split_idx]
    val_defect = all_defect[split_idx:]

    for img_path in train_defect:

        new_name = f"{img_path.parent.name}_{img_path.name}"
        shutil.copy(img_path, PROCESSED_DATA_DIR / 'train' / 'defect' / new_name)
    for img_path in val_defect:
        new_name = f"{img_path.parent.name}_{img_path.name}"
        shutil.copy(img_path, PROCESSED_DATA_DIR / 'val' / 'defect' / new_name)

    print(f"Dataset reorganization complete.")
    print(f"Train Good: {len(train_good)}, Train Defect: {len(train_defect)}")
    print(f"Val Good: {len(val_good)}, Val Defect: {len(val_defect)}")

def get_dataloaders(batch_size=32, img_size=224):

    if not PROCESSED_DATA_DIR.exists():
        reorganize_dataset()

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR / 'train',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=True
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DATA_DIR / 'val',
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=False
    )

    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")

    num_defect = len(list((PROCESSED_DATA_DIR / 'train' / 'defect').glob('*.png')))
    num_good = len(list((PROCESSED_DATA_DIR / 'train' / 'good').glob('*.png')))

    train_dataset = train_dataset.map(lambda x, y: (x, 1.0 - y))
    val_dataset = val_dataset.map(lambda x, y: (x, 1.0 - y))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, class_names, num_defect, num_good

if __name__ == '__main__':
    reorganize_dataset()
