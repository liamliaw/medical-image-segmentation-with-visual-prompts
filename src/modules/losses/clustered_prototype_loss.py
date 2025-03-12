import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
from einops import rearrange


"""
Based on https://github.com/marcdcfischer/PUNet/blob/main/src/modules/losses/contrastive_protos_teacher.py.
"""

class ClusteredPrototypeLoss(nn.Module):
    def __init__(self,
                 reduction_factor: float = 8.0,
                 k_means_iterations: int = 3,
                 fwhm: float = 128.0,
                 ):
        super().__init__()
        self.reduction_factor = reduction_factor
        self.k_means_iterations = k_means_iterations
        self.fwhm = fwhm

    def forward(self,
                emb_s: List[torch.Tensor],
                emb_t: torch.Tensor,
                coord_s: List[torch.Tensor],
                coord_t: torch.Tensor,
                temp_s: float = 0.066,
                temp_t: float = 0.033
                ):
        n_students = len(emb_s)
        # Sample initial prototypes.
        emb_p_sampled, coord_p_sampled = sample_embedding(
            emb_t, coord_t, self.reduction_factor * 2)
        # Sample teacher and students embeddings to reduce computing cost.
        emb_t_sampled, coord_t_sampled = sample_embedding(
            emb_t, coord_t, self.reduction_factor)
        emb_s_sampled, coord_s_sampled = [], []
        for i in range(n_students):
            e, c = sample_embedding(
                emb_s[i], coord_s[i], self.reduction_factor, with_jitter=True)
            emb_s_sampled.append(e), coord_s_sampled.append(c)
        # Prototype clustering.
        emb_p_sampled, coord_p_sampled, sim_t_p = cluster_prototype(
            emb_p_sampled, coord_p_sampled,
            emb_t_sampled, coord_t_sampled,
            self.k_means_iterations, temp_t, self.fwhm)

        total_loss = torch.tensor(0., device=emb_s[0].device)
        for i in range(n_students):
            # Assignments.
            assignment_loss = assign_prototype(
                emb_s_sampled[i], coord_s_sampled[i],
                emb_t_sampled, coord_t_sampled,
                emb_p_sampled, coord_p_sampled,
                sim_t_p, temp_s, self.fwhm,
            )
            total_loss += assignment_loss.mean()
        return total_loss


def assign_prototype(emb_z, coord_z, emb_t, coord_t, emb_p, coord_p,
                     sim_t_p, temp, fwhm):
    emb_z_n = F.normalize(emb_z, p=2, dim=-1)
    emb_p_n = F.normalize(emb_p, p=2, dim=-1)
    loss = []
    indices_closest, mask_max_dist = get_pos_idx(
        coord_x=coord_z,
        coord_y=coord_t,
        fwhm=fwhm,
    )
    sim_soft = torch.softmax(
        torch.einsum('b n c, b p c -> b n p', emb_z_n, emb_p_n) / temp,
        dim=-1
    )
    for j in range(emb_z.shape[0]):
        sim_slice = sim_soft[j][mask_max_dist[j]]
        assignment = sim_t_p[j, ...][
            indices_closest[j]][mask_max_dist[j]]
        ce_clustered = -(
            assignment
            * torch.clamp(torch.log(sim_slice + 1e-16), min=-1e3, max=-0.)
        ).sum(dim=1).mean(dim=0)
        loss.append(ce_clustered.reshape(-1))
    return torch.cat(loss)


def cluster_prototype(emb_p, coord_p, emb_t, coord_t, n_iter, temp, fwhm):
    emb_p_n = F.normalize(emb_p, p=2, dim=-1)
    emb_t_n = F.normalize(emb_t, p=2, dim=-1)
    # K-means iteration.
    for _ in range(n_iter):
        sim_soft = torch.softmax(
            torch.einsum('b n c, b p c -> b n p', emb_t_n, emb_p_n) / temp,
            dim=-1,
            )
        # Get position weights.
        pos_weight = get_pos_idx(
            coord_x=coord_t,
            coord_y=coord_p,
            fwhm=fwhm,
            return_pos_weight=True,
        )
        sim_weighted = sim_soft * pos_weight
        # Aggregate new prototypes and coordination.
        emb_p = torch.einsum(
            'b n p, b n c -> b p c',
            sim_weighted, emb_t,
        ) / torch.sum(
            sim_weighted, dim=1,
        ).unsqueeze(-1)
        emb_p_n = F.normalize(emb_p, p=2, dim=-1)
        h, w, d = coord_p.shape[2:]
        coord_p = torch.einsum(
            'b n p, b n c -> b p c',
            sim_weighted,
            rearrange(coord_t, 'b c h w d -> b (h w d) c')
        ) / torch.sum(
            sim_weighted, dim=1
        ).unsqueeze(-1)
        coord_p = rearrange(
            coord_p, 'b (h w d) c -> b c h w d', h=h, w=w, d=d,
        )
    # Recalculate teacher proxy alignment.
    sim_soft = torch.softmax(
        torch.einsum('b n c, b p c -> b n p', emb_t_n, emb_p_n) / temp,
        dim=-1,
    )
    pos_weight = get_pos_idx(
        coord_x=coord_t,
        coord_y=coord_p,
        fwhm=fwhm,
        return_pos_weight=True,
    )
    sim_weighted = sim_soft * pos_weight
    return emb_p, coord_p, sim_weighted


def get_pos_idx(
        coord_x: torch.Tensor,
        coord_y: torch.Tensor,
        fwhm: float = 256.,
        return_pos_weight: bool = False,
        max_dist: float = 4.0,
):
    diff_xyz = (
            rearrange(coord_x, 'b c h w d -> c b (h w d) ()')
            - rearrange(coord_y, 'b c h w d -> c b () (h w d)')
    )  # [3, B, N1, N2].
    # diff_xyz[2, ...] *= scale_z
    diff_all = torch.linalg.norm(diff_xyz, ord=2, dim=0)  # [B, N1, N2]
    if return_pos_weight:
        sigma_squared = (fwhm / 2.355) ** 2  # FWHM ~= 2.355 * sigma
        pos_weights = torch.exp(-(diff_all ** 2 / (2 * sigma_squared)))
        return pos_weights
    else:
        pos_min, idx_closest = torch.min(diff_all, dim=-1)
        mask_max_dist = (pos_min <= max_dist)
        return idx_closest, mask_max_dist


def sample_embedding(emb, coord, reduction_factor, with_jitter=False):
    with torch.no_grad():
        theta = torch.tensor(
            [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            device=emb.device,
        ).unsqueeze(0)
        reduced_size = [
            max(int(sh // reduction_factor), 1)
            for sh in emb.shape[2:]
        ]
        aff = F.affine_grid(
            theta=theta,
            size=[1, 1, *reduced_size],
            align_corners=False
        ).expand(emb.shape[0], -1, -1, -1, -1)  # [B, H', W', D', 3]
    if with_jitter:
        with torch.no_grad():
            spatial_jitter = torch.randint(
                low=0, high=int(math.ceil(reduction_factor)),
                size=(6,)
            )
        emb = emb[
              :, :,
              spatial_jitter[0]: emb.shape[2] - spatial_jitter[1],
              spatial_jitter[2]: emb.shape[3] - spatial_jitter[3],
              spatial_jitter[4]: emb.shape[4] - spatial_jitter[5],
              ]
        coord = coord[
                :, :,
                spatial_jitter[0]: coord.shape[2] - spatial_jitter[1],
                spatial_jitter[2]: coord.shape[3] - spatial_jitter[3],
                spatial_jitter[4]: coord.shape[4] - spatial_jitter[5],
                ]
    emb_sampled = F.grid_sample(
        emb, grid=aff, mode='bilinear',
        padding_mode='reflection', align_corners=False
    )  # [B, C, H', W', D']
    emb_sampled = rearrange(emb_sampled, 'b c h w d -> b (h w d) c')
    coord_sampled = F.grid_sample(
        coord, grid=aff, mode='bilinear',
        padding_mode='reflection', align_corners=False
    )  # [B, 3, H', W', D']

    return emb_sampled, coord_sampled

# def generate_prototypes(emb_t, coord_t, reduction_factor):
#     with torch.no_grad():
#         theta = torch.tensor(
#             [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
#             device=emb_t.device,
#         ).unsqueeze(0)
#         reduced_size = [
#             max(int(sh // reduction_factor), 1)
#             for sh in emb_t.shape[2:]
#         ]
#         aff = F.affine_grid(
#             theta=theta,
#             size=[1, 1, *reduced_size],
#             align_corners=False
#         ).expand(emb_t.shape[0], -1, -1, -1, -1)  # [B, H', W', D', 3]
#         emb_t_sampled = F.grid_sample(
#             emb_t, grid=aff, mode='bilinear',
#             padding_mode='reflection', align_corners=False
#         )  # [B, C, H', W', D']
#         coord_t_sampled = F.grid_sample(
#             coord_t, grid=aff, mode='bilinear',
#             padding_mode='reflection', align_corners=False
#         )  # [B, 3, H', W', D']
#
#     return emb_t_sampled, coord_t_sampled
# def reduce_students_teacher(emb_s, coord_s, emb_t, coord_t, reduction_factor):
#     # For computational efficiency.
#
#     n_students = len(emb_s)
#     emb_s_reduced, coord_s_reduced = \
#         [None for _ in range(n_students)], \
#         [None for _ in range(n_students)]
#     emb_t_reduced, coord_t_reduced = None, None
#     for i, (emb, coord) in enumerate(
#             zip(emb_s + [emb_t], coord_s + [coord_t])):
#         with torch.no_grad():
#             theta = torch.tensor(
#                 [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
#                 device=emb_s[0].device
#             ).unsqueeze(0)
#             reduced_size = [
#                 max(int(sh // reduction_factor), 1)
#                 for sh in emb.shape[2:]]
#             aff = F.affine_grid(
#                 theta=theta,
#                 size=[1, 1, *reduced_size],
#                 align_corners=False
#             ).expand(emb.shape[0], -1, -1, -1, -1)
#             spatial_jitter = torch.randint(
#                 low=0, high=int(math.ceil(reduction_factor)),
#                 size=(4,)
#             )
#         if i < n_students:
#             emb = emb[
#                 :, :,
#                 spatial_jitter[0]: emb.shape[2] - spatial_jitter[1],
#                 spatial_jitter[2]: emb.shape[3] - spatial_jitter[3], :
#             ]
#             coord = coord[
#                 :, :,
#                 spatial_jitter[0]: coord.shape[2] - spatial_jitter[1],
#                 spatial_jitter[2]: coord.shape[3] - spatial_jitter[3], :
#             ]
#         emb = F.grid_sample(
#             emb, grid=aff, mode='bilinear',
#             padding_mode='reflection', align_corners=False
#         )  # [B, C, H', W', D']
#         coord = F.grid_sample(
#             coord, grid=aff, mode='bilinear',
#             padding_mode='reflection', align_corners=False
#         )
#         if i < n_students:
#             emb_s_reduced[i] = emb
#             coord_s_reduced[i] = coord
#         else:
#             emb_t_reduced = emb
#             coord_t_reduced = coord
#
#     return emb_s_reduced, coord_s_reduced, emb_t_reduced, coord_t_reduced
# def generate_masks_student_teacher(
#         coord_s: list[torch.tensor],
#         coord_t: torch.tensor,
#         fwhm: float = 256.,
#         max_sim_dist: float = 4.0,
#         scale_z: float = 2.0,
#         thresh: float = 0.5):
#
#     pos_masks_student_teacher = []
#     indices_closest = []
#     masks_max_sim_dist = []
#     for i, coord in enumerate(coord_s):
#         diff_xyz = (
#                 rearrange(coord, 'b c h w d -> c b (h w d) ()')
#                 - rearrange(coord_t, 'b c h w d -> c b () (h w d)')
#         )  # [3, b, n1, n2].
#         diff_xyz[2, ...] *= scale_z
#         diff_all = torch.linalg.norm(diff_xyz, ord=2, dim=0)  # [b, n1, n2]
#         sigma_squared = (fwhm / 2.355) ** 2  # fwhm ~= 2.355 * sigma
#         pos_masks_student_teacher.append(
#             torch.exp(-(diff_all ** 2 / (2 * sigma_squared))) >= thresh)
#         # weights are compared to the threshold to produce binary mask.
#         pos_minimum, idx_closest = torch.min(diff_all, dim=-1)
#         # [b, n1], [b, n1].
#         masks_max_sim_dist.append(pos_minimum <= max_sim_dist)  # [b, n1].
#         indices_closest.append(idx_closest)
#
#     return pos_masks_student_teacher, indices_closest, masks_max_sim_dist
#
#
# def get_pos_weights(
#         coord_t: torch.Tensor,
#         coord_p: torch.Tensor,
#         fwhm: float = 256.,
#         scale_z: float = 2.0):
#     diff_xyz = (
#             rearrange(coord_t, 'b c h w d -> c b (h w d) ()')
#             - rearrange(coord_p, 'b n c -> c b () n')
#     )  # [3, B, N1, N2].
#     diff_xyz[2, ...] *= scale_z
#     diff_all = torch.linalg.norm(diff_xyz, ord=2, dim=0)  # [B, N1, N2]
#     sigma_squared = (fwhm / 2.355) ** 2  # FWHM ~= 2.355 * sigma
#     pos_weights = torch.exp(-(diff_all ** 2 / (2 * sigma_squared)))
#
#     return pos_weights
#
# def forward(self,
#             emb_s: List[torch.Tensor],
#             emb_t: torch.Tensor,
#             coord_s: List[torch.Tensor],
#             coord_t: torch.Tensor,
#             temp_s: float = 0.066,
#             temp_t: float = 0.033):

#     n_students = len(emb_s)

#     emb_p, coord_p = generate_prototypes(
#         emb_t, coord_t, self.reduction_factor_protos)

#     emb_s_reduced, coord_s_reduced, emb_t_reduced, coord_t_reduced = \
#         reduce_students_teacher(emb_s, coord_s, emb_t, coord_t,
#                                 self.reduction_factor_students)

#     loss_sim_clustered = [
#         torch.zeros((0,), device=emb_s[0].device)
#         for _ in range(n_students)]

#     with torch.no_grad():
#         pos_weights_s_t, indices_closest, masks_max_sim_dist = \
#             generate_masks_student_teacher(
#                 coord_s=coord_s_reduced,
#                 coord_t=coord_t_reduced,
#                 fwhm=self.fwhm_student_teacher,
#             )
#         # Normalize embeddings.
#         emb_t_normed = F.normalize(
#             rearrange(emb_t_reduced, 'b c h w d -> b (h w d) c'),
#             p=2, dim=-1
#         )
#         emb_p_normed = F.normalize(
#             rearrange(emb_p, 'b c h w d -> b (h w d) c'),
#             p=2, dim=-1
#         )
#         coord_p = rearrange(
#             coord_p,
#             'b c h w d -> b (h w d) c'
#         )
#         # K-means iteration.
#         for _ in range(self.k_means_iterations):
#             sim_t_p_soft = torch.softmax(
#                 torch.einsum('b n c, b p c -> b n p',
#                              emb_t_normed, emb_p_normed) / temp_t,
#                 dim=-1,
#             )
#             # Get position weights.
#             pos_weights_t_p = get_pos_weights(
#                 coord_t=coord_t_reduced,
#                 coord_p=coord_p,
#                 fwhm=self.fwhm_teacher_protos,
#             )
#             if self.use_weight_protos:
#                 sim_t_p_soft_weighted = sim_t_p_soft \
#                     * pos_weights_t_p
#             else:
#                 sim_t_p_soft_weighted = sim_t_p_soft

#             # Aggregate new protos and coords.
#             emb_p = torch.einsum(
#                 'b n p, b n c -> b p c',
#                 sim_t_p_soft_weighted,
#                 rearrange(emb_t_reduced, 'b c h w d -> b (h w d) c'),
#             ) / torch.sum(
#                 sim_t_p_soft_weighted,
#                 dim=1
#             ).unsqueeze(-1)

#             emb_p_normed = F.normalize(emb_p, p=2, dim=-1)
#             coord_p = torch.einsum(
#                 'b n p, b n c -> b p c',
#                 sim_t_p_soft_weighted,
#                 rearrange(coord_t_reduced, 'b c h w d -> b (h w d) c')
#             ) / torch.sum(
#                 sim_t_p_soft_weighted,
#                 dim=1
#             ).unsqueeze(-1)
#         # Recalculate teacher proxy alignment.
#         sim_t_p_soft = torch.softmax(
#             torch.einsum('b n c, b p c -> b n p',
#                          emb_t_normed, emb_p_normed) / temp_t,
#             dim=-1
#         )
#         pos_weights_t_p = get_pos_weights(
#             coord_t=coord_t_reduced,
#             coord_p=coord_p,
#             fwhm=self.fwhm_teacher_protos,
#         )
#         if self.use_weight_teacher:
#             sim_t_p_soft_weighted = sim_t_p_soft \
#                 * pos_weights_t_p
#         else:
#             sim_t_p_soft_weighted = sim_t_p_soft

#     for idx_s in range(n_students):
#         emb_s_normed = F.normalize(
#             rearrange(emb_s_reduced[idx_s],
#                       'b c h w d -> b (h w d) c'),
#             p=2, dim=-1,
#         )  # [B, N, C]
#         sim_s_p_soft = torch.softmax(
#             torch.einsum('b n c, b p c -> b n p',
#                          emb_s_normed, emb_p_normed) / temp_s,
#             dim=-1
#         )
#         cluster_assignments = sim_t_p_soft_weighted[
#             :, indices_closest[idx_s]]
#         ce_clustered = -(
#             cluster_assignments
#             * torch.clamp(torch.log(
#                 sim_s_p_soft.unsqueeze(1) + 1e-16),
#                 min=-1e3, max=-0.)
#         ).sum(dim=1).mean(dim=0)
#         loss_sim_clustered[idx_s] = torch.concat(
#             [loss_sim_clustered[idx_s], ce_clustered.reshape(-1)])

#     total_losses = 0
#     for idx_s in range(n_students):
#         total_losses += self.loss_weight \
#             * loss_sim_clustered[idx_s].mean() \
#             if loss_sim_clustered[idx_s].shape[0] > 0 \
#             else torch.tensor(0., device=emb_s[0].device)
#     return total_losses
