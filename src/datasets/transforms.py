from itertools import chain
import nibabel as nib
from typing import Optional, Tuple, List
import monai.transforms as transforms
import torch
import numpy as np


# Test transforms.
def get_test_transform_downstream(conf):
    content_keys = [conf.image_dict_key, conf.mask_dict_key]
    test_ts = transforms.Compose([
        transforms.LoadImaged(keys=content_keys),
        transforms.EnsureChannelFirstd(keys=content_keys),
        transforms.ScaleIntensityRanged(
            keys=[conf.image_dict_key],
            a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
        ),
    ])
    test_ts = transforms.Compose([
        test_ts,
        transforms.Orientationd(
            keys=content_keys,
            axcodes='RAS',
        )
    ])
    test_ts = transforms.Compose([
        test_ts,
        transforms.ToTensord(keys=content_keys)
    ])
    return test_ts


# Downstream transform.
def get_fit_transform_downstream(conf):
    content_keys = [conf.image_dict_key, conf.mask_dict_key]
    num_samples = get_ns(conf)
    fit_ts = transforms.Compose([
        transforms.LoadImaged(keys=content_keys),
        transforms.EnsureChannelFirstd(keys=content_keys),
        transforms.ScaleIntensityRanged(
            keys=[conf.image_dict_key],
            a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
        ),
    ])
    # Orientation.
    if conf.random_orientation:
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.OneOf([
                transforms.Rotate90d(
                    keys=content_keys, k=1, spatial_axes=(0, 1)),
                transforms.Rotate90d(
                    keys=content_keys, k=1, spatial_axes=(0, 2)),
                transforms.Rotate90d(
                    keys=content_keys, k=1, spatial_axes=(1, 2)),
            ]),
        ])
    # Set axial view.
    elif conf.orientation == 'axial':
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.Orientationd(
                keys=content_keys,
                axcodes='RAS',
            )
        ])
    # Resize.
    if conf.resize_content:
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.Resized(
                keys=[conf.image_dict_key],
                spatial_size=conf.resize_spatial_size,
                size_mode='all', mode='trilinear',
            ),
            transforms.Resized(
                keys=[conf.mask_dict_key],
                spatial_size=conf.resize_spatial_size,
                size_mode='all', mode='nearest',
            ),
        ])
    # Crop and pad.
    fit_ts = transforms.Compose([
        fit_ts,
        transforms.RandSpatialCropSamplesd(
            keys=content_keys,
            roi_size=conf.seg_input_size,
            num_samples=num_samples,
            random_center=True,
            random_size=False,
        ),
        transforms.SpatialPadd(
            keys=content_keys,
            spatial_size=conf.seg_input_size,
        ),
        transforms.ToTensord(keys=content_keys)
    ])
    return fit_ts


def get_fit_transform_pretrain(conf):
    image_keys = [conf.image_dict_key]
    mask_keys = [conf.mask_dict_key]
    coord_keys = [conf.coord_dict_key]
    basic_ks = []
    basic_ks += image_keys
    num_samples = get_ns(conf)
    if (conf.training_mode == 'supervised_learning_decoder'
        or conf.training_mode == 'supervised_learning_all'):
        load_mask = True
    else:
        load_mask = False
    if (conf.training_mode == 'self_supervised_learning_decoder'
        or conf.training_mode == 'self_supervised_learning_all'
        or conf.training_mode == 'supervised_learning_decoder'
        or conf.training_mode == 'supervised_learning_all') \
            and conf.use_prototype_assignment:
        load_coord = True
        students_teacher_view = True
    else:
        load_coord = False
        students_teacher_view = False
    if load_mask:
        basic_ks += mask_keys
    if load_coord:
        basic_ks += coord_keys
    students_ks = []
    if (conf.training_mode == 'self_supervised_learning_decoder'
        or conf.training_mode == 'self_supervised_learning_all'
        or conf.training_mode == 'supervised_learning_decoder'
        or conf.training_mode == 'supervised_learning_all') \
            and conf.use_prototype_assignment:
        num_students = len(conf.crop_size_students)
        for i in range(num_students):
            for k in basic_ks:
                students_ks.append(f'{k}_st_{i}')
    # Load image.
    fit_ts = transforms.Compose([
        transforms.LoadImaged(keys=image_keys),
        transforms.EnsureChannelFirstd(keys=image_keys),
        transforms.ScaleIntensityRanged(
            keys=image_keys,
            a_min=-1000, a_max=1000,
            b_min=0, b_max=1,
            clip=True,
        ),
    ])
    # Load extra contents.
    if load_mask:
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.LoadImaged(keys=mask_keys),
            transforms.EnsureChannelFirstd(keys=mask_keys),
        ])
    # Set axial view.
    if conf.orientation == 'axial':
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.Orientationd(
                keys=image_keys if not load_mask else image_keys + mask_keys,
                axcodes='RAS',
            )
        ])
    if conf.resize_content:
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.Resized(
                keys=image_keys, spatial_size=conf.resize_spatial_size,
                size_mode='all', mode='trilinear',
            ),
        ])
        if load_mask:
            fit_ts = transforms.Compose([
                fit_ts,
                transforms.Resized(
                    keys=mask_keys, spatial_size=conf.resize_spatial_size,
                    size_mode='all', mode='nearest',
                ),
            ])

    if load_coord:
        fit_ts = transforms.Compose([
            fit_ts, LoadCoordGridd(keys=image_keys, name=coord_keys[0])])
    # Orientation.
    if conf.random_orientation:
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.OneOf([
                transforms.Rotate90d(
                    keys=basic_ks, k=1, spatial_axes=(0, 1)),
                transforms.Rotate90d(
                    keys=basic_ks, k=1, spatial_axes=(0, 2)),
                transforms.Rotate90d(
                    keys=basic_ks, k=1, spatial_axes=(1, 2)),
            ]),
        ])
    # Crop samples.
    fit_ts = transforms.Compose([
        fit_ts,
        transforms.RandSpatialCropSamplesd(
            keys=basic_ks,
            roi_size=conf.roi_size,
            num_samples=num_samples,
            random_center=True,
            random_size=False,
        ),
        transforms.SpatialPadd(
            keys=basic_ks,
            spatial_size=conf.roi_size,
        ),
    ])
    # Copy samples.
    if students_teacher_view:
        num_students = len(conf.crop_size_students)
        fit_ts = transforms.Compose([
            fit_ts,
            transforms.CopyItemsd(
                keys=basic_ks, names=students_ks, times=num_students)
        ])
    # Random image transforms.
    if conf.random_transforms:
        basic_rand_ts = transforms.Compose([
            transforms.RandBiasFieldd(
                keys=[k for k in basic_ks if conf.image_dict_key in str(k)],
                prob=0.05),
            transforms.RandStdShiftIntensityd(
                keys=[k for k in basic_ks if conf.image_dict_key in str(k)],
                prob=0.05, factors=(0.0, 0.1)),
            transforms.RandAdjustContrastd(
                keys=[k for k in basic_ks if conf.image_dict_key in str(k)],
                prob=0.05),
            transforms.RandScaleIntensityd(
                keys=[k for k in basic_ks if conf.image_dict_key in str(k)],
                prob=0.05, factors=-2.),
            transforms.RandHistogramShiftd(
                keys=[k for k in basic_ks if conf.image_dict_key in str(k)],
                prob=0.05, num_control_points=(8, 12)),
        ])
        fit_ts = transforms.Compose([
            fit_ts,
            basic_rand_ts,
        ])
        if students_teacher_view:
            # Random transformation for students.
            students_rand_ts = transforms.Compose([])
            num_students = len(conf.crop_size_students)
            for i in range(num_students):
                students_rand_ts = transforms.Compose([
                    students_rand_ts,
                    transforms.OneOf([
                        transforms.Compose([]),
                        transforms.RandCoarseDropoutd(
                            keys=[k for k in students_ks
                                  if conf.image_dict_key in str(k)
                                  and f'st_{i}' in str(k)],
                            prob=1.0, dropout_holes=True, holes=1,
                            max_holes=3, spatial_size=4, max_spatial_size=16),
                        transforms.RandCoarseDropoutd(
                            keys=[k for k in students_ks
                                  if conf.image_dict_key in str(k)
                                  and f'st_{i}' in str(k)],
                            prob=1.0, dropout_holes=False, holes=5,
                            spatial_size=32, max_spatial_size=48),
                        transforms.RandCoarseShuffled(
                            keys=[k for k in students_ks
                                  if conf.image_dict_key in str(k)
                                  and f'st_{i}' in str(k)],
                            prob=1.0, holes=1, max_holes=3, spatial_size=4,
                            max_spatial_size=16),
                    ], weights=(0.7, 0.1, 0.1, 0.1)),
                    transforms.RandBiasFieldd(
                        keys=[k for k in students_ks
                              if conf.image_dict_key in str(k)
                              and f'st_{i}' in str(k)],
                        prob=0.1),
                    transforms.RandStdShiftIntensityd(
                        keys=[k for k in students_ks
                              if conf.image_dict_key in str(k)
                              and f'st_{i}' in str(k)],
                        prob=0.1, factors=(0.0, 0.2)),
                    transforms.RandAdjustContrastd(
                        keys=[k for k in students_ks
                              if conf.image_dict_key in str(k)
                              and f'st_{i}' in str(k)],
                        prob=0.1),
                    transforms.RandScaleIntensityd(
                        keys=[k for k in students_ks
                              if conf.image_dict_key in str(k)
                              and f'st_{i}' in str(k)],
                        prob=0.1, factors=-2.),
                    transforms.RandHistogramShiftd(
                        keys=[k for k in students_ks
                              if conf.image_dict_key in str(k)
                              and f'st_{i}' in str(k)],
                        prob=0.1, num_control_points=(8, 12)),
                ])

    # Crop students.
    if students_teacher_view:
        num_students = len(conf.crop_size_students)
        for i in range(num_students):
            fit_ts = transforms.Compose([
                fit_ts,
                transforms.RandSpatialCropd(
                    keys=[k for k in students_ks if f'st_{i}' in str(k)],
                    roi_size=conf.crop_size_students[i],
                    random_center=True, random_size=False),
                transforms.SpatialPadd(
                    keys=[k for k in students_ks if f'st_{i}' in str(k)],
                    spatial_size=conf.crop_size_students[i],
                ),
            ])
    # To torch tensor.
    fit_ts = transforms.Compose([
        fit_ts,
        transforms.ToTensord(keys=basic_ks + students_ks),
    ])

    return fit_ts


class LoadCoordGridd(transforms.MapTransform):
    def __init__(self, keys, name):
        super().__init__(keys)
        self.name = name

    def __call__(self, data):
        for key in self.keys:
            if key in data and 'image' in key:
                data[str(key).replace('image', self.name)] = \
                    get_coord_grid(data[key].shape)
        return data


# Generate coordination grids.
def get_coord_grid(image_size: Tuple[int, int, int, int]):
    coord_grid = torch.stack(torch.meshgrid(
        torch.arange(image_size[1]), torch.arange(image_size[2]),
        torch.arange(image_size[3]), indexing='ij'), dim=0).float()
    coord_grid -= torch.tensor(
        ((image_size[1] - 1) / 2., (image_size[2] - 1) / 2.,
         (image_size[3] - 1) / 2.)).reshape(3, 1, 1, 1)
    return np.array(coord_grid)


# Set in configuration files, depends on specific hardware setting.
def get_ns(conf):
    if conf.training_mode == 'downstream':
        ns = conf.num_samples_downstream
    elif conf.training_mode == 'self_supervised_learning_encoder':
        ns = conf.num_samples_multi_view
    elif (conf.training_mode == 'self_supervised_learning_decoder'
          or conf.training_mode == 'self_supervised_learning_all'
          or conf.training_mode == 'supervised_learning_decoder'
          or conf.training_mode == 'supervised_learning_all'):
        ns = conf.num_samples_students_teacher
    else:
        raise ValueError()
    return ns


class LoadPseudoBgMaskd(transforms.MapTransform):
    def __init__(self, keys, name):
        super().__init__(keys)
        self.name = name

    def __call__(self, data):
        for key in self.keys:
            if key in data and 'image' in key:
                data[key.replace('image', self.name)] = \
                    data['image'].gt(0.0025)
        return data

if __name__ == '__main__':
    pass
