import cv2
from einops import rearrange
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from typing import Sequence


class MeanIoU:
    def __init__(self, num_classes, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, preds, target):
        pred = torch.argmax(preds, dim=1, keepdim=True)
        for cls in range(self.num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection

            self.intersection[cls] += intersection.item()
            self.union[cls] += union.item()

    def compute(self):
        return torch.mean(self.intersection / (self.union + 1e-6))


class DiceCoefficient:
    def __init__(self, num_classes, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, preds, target):
        pred = torch.argmax(preds, dim=1, keepdim=True)
        for cls in range(self.num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()
            intersection = (pred_cls * target_cls).sum()
            pred_sum = pred_cls.sum()
            target_sum = target_cls.sum()

            self.intersection[cls] += intersection.item()
            self.union[cls] += pred_sum.item() + target_sum.item()

    def compute(self):
        return torch.mean(2 * self.intersection / (self.union + 1e-6))


class WarmupCosineSchedule(LambdaLR):
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 t_total: int,
                 cycles: float = 0.5,
                 last_epoch: int = -1
                 ) -> None:
        """
        taken from https://github.com/UCSC-VLAA/SwinMM
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) \
            / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(
            math.pi * float(self.cycles) * 2.0 * progress)))


def generate_pseudo_mask_from_multi_view(
        out, n_tokens, src_rot=None, dst_rot=None,
        permutation=None,
):
    h, w, d = out['latent_output'].shape[-3:]
    sim = similarity_aggregation(
        latent=rearrange(
            out['latent_output'], 'b c h w d -> b (h w d) c'),
        prompt=rearrange(
            out['final_instruction'], 'b (i n) c -> b i n c', n=n_tokens),
        mean_aggregation=False,
    )
    pse_msk = rearrange(
        sim, 'b i (h w d) -> b i h w d',
        h=h, w=w, d=d,
    )
    if permutation is not None:
        pse_msk = permutation(pse_msk)
    if (src_rot is not None) and (dst_rot is not None):
        pse_msk = align_rotation(pse_msk, src_rot=src_rot, dst_rot=dst_rot)
    return pse_msk


def similarity_aggregation(latent,
                           prompt,
                           temp: float = 0.1,
                           mean_aggregation: bool = False,
                           ):
    sim = (torch.einsum(
        'b m c, b i n c -> b i n m',
        F.normalize(latent, p=2, dim=-1),
        F.normalize(prompt, p=2, dim=-1)
    ) + 1) / 2
    if not mean_aggregation:
        sim = torch.softmax(sim.detach() / temp, dim=2) * sim
        sim = torch.sum(sim, dim=2)
    else:
        sim = torch.mean(sim, dim=2)
    return sim


def view_reconstruction(
        name, n_slices, epoch, step, ori_img=None, rec_img=None):
    st = ori_img.size(4) // n_slices
    img_slices, rec_slices = [], []
    for i in range(n_slices):
        img_slice = ori_img[0, 0, :, :, i * st]
        img_slice = img_slice.detach().cpu().numpy() * 255
        img_slice = cv2.resize(
            img_slice, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        img_slices.append(img_slice)
        rec_slice = rec_img[0, 0, :, :, i * st]
        rec_slice = rec_slice.detach().cpu().numpy() * 255
        rec_slice = cv2.resize(
            rec_slice, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        rec_slices.append(rec_slice)
    img_cat = np.concatenate([img for img in img_slices], axis=1)
    rec_cat = np.concatenate([rec for rec in rec_slices], axis=1)
    cat_total = np.concatenate(
        [img_cat, rec_cat], axis=0)
    save_dir = Path(f'~/image_outputs/rec').expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = Path(
        f'~/image_outputs/rec/epoch{epoch}_step{step}_{name[0]}_rec.png'
    ).expanduser()
    cv2.imwrite(str(save_path), cat_total)


def view_prototype(
        name, n_slices, epoch, step, c, prt1=None, prt2=None):
    # prt1 = torch.argmax(prt1, dim=1, keepdim=True)
    # prt2 = torch.argmax(prt2, dim=1, keepdim=True)
    st = prt1.size(2) // n_slices
    slices1, slices2 = [], []
    for i in range(n_slices):
        slice1 = prt1[0, 0, i * st, :, :]
        slice1 = slice1.detach().cpu().numpy() * (255 // c)
        slice1 = cv2.resize(
            slice1, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        slices1.append(slice1)
        slice2 = prt2[0, 0, i * st, :, :]
        slice2 = slice2.detach().cpu().numpy() * (255 // c)
        slice2 = cv2.resize(
            slice2, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        slices2.append(slice2)
    cat1 = np.concatenate([img for img in slices1], axis=1)
    cat2 = np.concatenate([img for img in slices2], axis=1)
    cat_total = np.concatenate(
        [cat1, cat2], axis=0)
    save_dir = Path(f'~/image_outputs/prt').expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = Path(
        f'~/image_outputs/prt/epoch{epoch}_step{step}_{name}_prt.png'
    ).expanduser()
    cv2.imwrite(str(save_path), cat_total)


def view_prototype_students_teacher(
        name, n_slices, epoch, step, chs,
        prt_tch=None, img_tch=None,
        prt_sts=None, img_sts=None,
):
    def _get_slices(img, prt, stride):
        slices_img, slices_prt = [], []
        for i in range(n_slices):
            sl_img = img[0, 0, :, :, i * stride]
            sl_img = sl_img.detach().cpu().numpy() * 255
            sl_img = cv2.resize(
                sl_img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            slices_img.append(sl_img)
            sl_prt = prt[0, 0, :, :, i * stride]
            sl_prt = sl_prt.detach().cpu().numpy() * (255 // chs)
            sl_prt = cv2.resize(
                sl_prt, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
            slices_prt.append(sl_prt)
        cat_img = np.concatenate([img for img in slices_img], axis=1)
        cat_prt = np.concatenate([prt for prt in slices_prt], axis=1)
        cat_total = np.concatenate(
            [cat_img, cat_prt], axis=0)
        return cat_total

    total_out = []
    st_tch = prt_tch.size(4) // n_slices
    prt_tch = torch.argmax(prt_tch, dim=1, keepdim=True)
    total_out.append(_get_slices(img_tch, prt_tch, st_tch))
    for i in range(len(prt_sts)):
        st_sts = prt_sts[i].size(4) // n_slices
        prt_sts[i] = torch.argmax(prt_sts[i], dim=1, keepdim=True)
        total_out.append(_get_slices(img_sts[i], prt_sts[i], st_sts))
    cat_total = np.concatenate([out for out in total_out], axis=0)

    save_dir = Path(f'~/image_outputs/prt').expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = Path(
        f'~/image_outputs/prt/epoch{epoch}_step{step}_{name[0]}_prt.png'
    ).expanduser()
    cv2.imwrite(str(save_path), cat_total)

def view_segmentation(name, n_slices, epoch, step,
                      seg_pred=None, seg_target=None, img=None, n_classes=None,):
    # print(seg_target.cpu().numpy().max())
    st = seg_pred.size(4) // n_slices
    pred_slices, target_slices, img_slices = [], [], []
    for i in range(n_slices):
        pred_slice = torch.argmax(
            seg_pred, dim=1, keepdim=True
        )[1, 0, :, :, i * st]
        pred_slice = pred_slice.detach().cpu().numpy() * (255 // n_classes)
        pred_slice = cv2.resize(
            pred_slice, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        pred_slices.append(pred_slice)
        target_slice = seg_target[1, 0, :, :, i * st]
        target_slice = target_slice.detach().cpu().numpy() * (255 // n_classes)
        target_slice = cv2.resize(
            target_slice, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        target_slices.append(target_slice)

        img_slice = img[1, 0, :, :, i * st]
        img_slice = img_slice.detach().cpu().numpy() * (255 // n_classes)
        img_slice = cv2.resize(
            img_slice, dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
        img_slices.append(img_slice)

    pred_cat = np.concatenate([pred for pred in pred_slices], axis=1)
    target_cat = np.concatenate([tgt for tgt in target_slices], axis=1)
    img_cat = np.concatenate([img for img in img_slices], axis=1)
    cat_total = np.concatenate([pred_cat, target_cat, img_cat], axis=0)
    save_dir = Path(f'~/image_outputs/seg_new').expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = Path(
        f'~/image_outputs/seg_new/epoch{epoch}_step{step}_{name}_seg.png'
    ).expanduser()
    cv2.imwrite(str(save_path), cat_total)


def random_mask(x: torch.Tensor,
                input_size: Sequence[int],
                patch_size: Sequence[int],
                masking_ratio: float,
                ):
    if any([s0 % s1 != 0 for s0, s1 in zip(input_size, patch_size)]):
        raise ValueError(
            f'Input size {input_size} and patch size {patch_size} '
            f'is not compatible!')

    mask_shape = [s0 // s1 for s0, s1 in zip(input_size, patch_size)]
    n_patches = np.prod(mask_shape).item()
    mask = np.ones(n_patches, dtype=bool)
    indices = np.random.choice(
        n_patches,
        round(n_patches * (1 - masking_ratio)),
        replace=False,
    )
    mask[indices] = False
    mask = mask.reshape(mask_shape)
    h, w, d = patch_size
    mask = np.logical_or(
        mask[:, None, :, None, :, None],
        np.zeros([1, h, 1, w, 1, d], dtype=bool),
    ).reshape(input_size)
    mask = torch.from_numpy(mask).to(x.device)
    x_masked = x.detach().clone()
    x_masked[:, :, mask] = 0
    return x_masked, ~mask


def random_permute(x: torch.Tensor):
    permutation_transforms = {
        0: lambda x: x.permute(0, 1, 3, 2, 4),
        1: lambda x: x.permute(0, 1, 4, 3, 2),
        2: lambda x: x.permute(0, 1, 2, 4, 3),
    }
    permutation = permutation_transforms[np.random.choice(
        len(permutation_transforms))]
    x_permuted = permutation(x).contiguous()
    return x_permuted, permutation


def align_rotation(x, src_rot=None, dst_rot=None):
    assert src_rot is not None or dst_rot is not None, \
        f'src_rot or dst_rot is None!'
    n_img = x.size(0)
    x_new = x.clone()
    # Just rotate back.
    if src_rot is not None:
        for i, d in zip(range(n_img), src_rot):
            if d == 1:
                x_new[i] = x[i].rot90(3, (2, 3)).contiguous()
            elif d == 2:
                x_new[i] = x[i].rot90(2, (2, 3)).contiguous()
            elif d == 3:
                x_new[i] = x[i].rot90(1, (2, 3)).contiguous()
    if dst_rot is not None:
        for i, d in zip(range(n_img), dst_rot):
            if d == 1:
                x_new[i] = x_new[i].rot90(1, (2, 3)).contiguous()
            elif d == 2:
                x_new[i] = x_new[i].rot90(2, (2, 3)).contiguous()
            elif d == 3:
                x_new[i] = x_new[i].rot90(3, (2, 3)).contiguous()
    return x_new


def random_rotate(x: torch.Tensor):
    n_img = x.size(0)
    x_rot = x.detach().clone()
    y_rot = torch.zeros(n_img, dtype=torch.int64, device=x.device)
    for i in range(n_img):
        orientation = np.random.randint(0, 4)
        if orientation == 1:
            x_rot[i] = x[i].unsqueeze(0).rot90(1, (2, 3))
        elif orientation == 2:
            x_rot[i] = x[i].unsqueeze(0).rot90(2, (2, 3))
        elif orientation == 3:
            x_rot[i] = x[i].unsqueeze(0).rot90(3, (2, 3))
        y_rot[i] = orientation
    return x_rot, y_rot


@torch.no_grad()
def sinkhorn_knopp(q, num_iters=3):
    b, c, h, w, d = q.shape
    q = rearrange(q, 'b c h w d -> b c (h w d)')
    q = torch.exp(F.normalize(q, p=2, dim=1))
    q = q / torch.sum(q)

    for _ in range(num_iters):
        sum_of_rows = torch.sum(q, dim=0, keepdim=True)
        q /= sum_of_rows
        q /= c
        sum_of_cols = torch.sum(q, dim=1, keepdim=True)
        q /= sum_of_cols
        q /= b

    q *= b
    q = rearrange(q, 'b c (h w d) -> b c h w d', h=h, w=w, d=d)
    return q


# Map label indices to actual output channels.
def map_label_indices(masks: torch.Tensor, active_labels):
    active_labels.sort()
    bool_mask = torch.zeros_like(masks, dtype=torch.bool)
    for label in active_labels:
        bool_mask |= (masks == float(label))
    masks[~bool_mask] = 0

    mapping = {
        lbl: new_lbl for lbl, new_lbl in zip(active_labels,
                                             range(len(active_labels)))
    }

    for lbl, new_lbl in mapping.items():
        masks[masks == lbl] = float(new_lbl)

    # print(masks.numpy().max())
    return masks

