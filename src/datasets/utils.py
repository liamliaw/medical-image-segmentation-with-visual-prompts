from monai.data import Dataset, DataLoader
import math
from pathlib import Path
import random
from .transforms import get_fit_transform_pretrain, get_fit_transform_downstream, \
    get_test_transform_downstream


def get_fit_loader_self_supervised_learning(conf):
    data = []
    # Select data from directories.
    for pth in Path(conf.image_dir_pretrain).iterdir():
        data.append({
            conf.image_dict_key: str(pth),
            'name': str(pth.stem),
        })
    # Select target numbers of data.
    if conf.num_selected_data_pretrain != -1:
        data = random.choices(data, k=conf.num_selected_data_pretrain)
        print(f'Selected {len(data)} samples.')
    if True:
        random.shuffle(data)

    split_index = math.floor(len(data) * conf.split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # Get transforms.
    ts = get_fit_transform_pretrain(conf)
    train_ds = Dataset(data=train_data, transform=ts)
    val_ds = Dataset(data=val_data, transform=ts)
    bs = get_bs(conf)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=True,
    )
    return train_loader, val_loader


def get_fit_loader_supervised_learning(conf):
    image_paths, mask_paths, data = [], [], []
    for pth in Path(conf.image_dir_supervised).iterdir():
        image_paths.append(pth)
    for pth in Path(conf.mask_dir_supervised).iterdir():
        mask_paths.append(pth)
    image_paths.sort(), mask_paths.sort()
    for img_pth, msk_pth in zip(image_paths, mask_paths):
        data.append({
            conf.image_dict_key: img_pth,
            conf.mask_dict_key: msk_pth,
            'name': str(img_pth.stem),
        })
    if conf.num_selected_data_supervised != -1:
        data = random.choices(data, k=conf.num_selected_data_supervised)
        print(f'Selected {len(data)} samples.')
    if True:
        random.shuffle(data)

    split_index = math.floor(len(data) * conf.split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]

    ts = get_fit_transform_pretrain(conf)
    train_ds = Dataset(data=train_data, transform=ts)
    val_ds = Dataset(data=val_data, transform=ts)
    bs = get_bs(conf)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=True,
    )
    return train_loader, val_loader


def get_fit_loader_downstream(conf):
    image_paths, mask_paths, data = [], [], []
    for pth in Path(conf.image_dir_downstream).iterdir():
        image_paths.append(pth)
    for pth in Path(conf.mask_dir_downstream).iterdir():
        mask_paths.append(pth)
    image_paths.sort(), mask_paths.sort()
    for img_pth, msk_pth in zip(image_paths, mask_paths):
        data.append({
            conf.image_dict_key: img_pth,
            conf.mask_dict_key: msk_pth,
            'name': str(img_pth.stem),
        })
    if conf.num_selected_data_downstream != -1:
        data = random.choices(data, k=conf.num_selected_data_downstream)
        print(f'Selected {len(data)} samples.')

    if True:
        random.shuffle(data)

    if len(data) < 2:
        raise ValueError(f'Not enough samples for downstream task.')
    if len(data) == 2:
        split_index = 1
    else:
        split_index = math.floor(len(data) * conf.split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    ts = get_fit_transform_downstream(conf)
    train_ds = Dataset(data=train_data, transform=ts)
    val_ds = Dataset(data=val_data, transform=ts)
    bs = get_bs(conf)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def get_test_loader_downstream(conf):
    image_paths, mask_paths, data = [], [], []
    for pth in Path(conf.image_dir_test).iterdir():
        image_paths.append(pth)
    for pth in Path(conf.mask_dir_test).iterdir():
        mask_paths.append(pth)
    image_paths.sort(), mask_paths.sort()
    for img_pth, msk_pth in zip(image_paths, mask_paths):
        data.append({
            conf.image_dict_key: img_pth,
            conf.mask_dict_key: msk_pth,
            'name': str(img_pth.stem),
        })
    ts = get_test_transform_downstream(conf)
    ds = Dataset(data=data, transform=ts)
    # bs = get_bs(conf)

    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

# The batch size is set in configuration files, depends on specific hardware setting.
def get_bs(conf):
    if conf.training_mode == 'downstream':
        bs = conf.batch_size_downstream
    elif conf.training_mode == 'self_supervised_learning_encoder':
        bs = conf.batch_size_multi_view
    elif (conf.training_mode == 'self_supervised_learning_decoder'
          or conf.training_mode == 'self_supervised_learning_all'
          or conf.training_mode == 'supervised_learning_decoder'
          or conf.training_mode == 'supervised_learning_all'):
        bs = conf.batch_size_students_teacher
    else:
        raise ValueError()
    return bs