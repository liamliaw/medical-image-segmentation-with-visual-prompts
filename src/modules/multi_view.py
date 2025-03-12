import torch
import torch.nn as nn
from torch.optim import AdamW
from torchinfo import summary
from .losses import ContrastivePairLoss
from .utils import (
    WarmupCosineSchedule,
    random_mask, random_permute, random_rotate,
    view_reconstruction,
)


class MultiViewTrainer:
    def __init__(self, hparams, backbone, train_loader, val_loader, logger, writer):
        super().__init__()
        self.hparams = hparams
        self.model = backbone(hparams)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.writer = writer
        if torch.cuda.is_available() and hparams.gpu != 0:
            self.device = torch.device('cuda')

    def configure_losses(self):
        # Losses
        loss_fn, avg_losses, avg_losses_val, best_val = {}, {}, {}, {}
        if self.hparams.use_reconstruction:
            loss_fn['rec'] = nn.MSELoss()
            avg_losses['rec'] = []
            avg_losses_val['rec'] = []
            best_val['rec'] = float('inf')
        if self.hparams.use_rotation_prediction:
            loss_fn['rot'] = nn.CrossEntropyLoss()
            avg_losses['rot'] = []
            avg_losses_val['rot'] = []
            best_val['rot'] = float('inf')
        if self.hparams.use_contrastive_learning:
            loss_fn['con'] = ContrastivePairLoss(
                self.hparams.batch_size_multi_view
                * self.hparams.num_samples_multi_view)
            avg_losses['con'] = []
            avg_losses_val['con'] = []
            best_val['con'] = float('inf')
        if self.hparams.use_mutual_learning:
            loss_fn['mut'] = nn.MSELoss()
            avg_losses['mut'] = []
            avg_losses_val['mut'] = []
            best_val['mut'] = float('inf')
        if not loss_fn:
            raise ValueError('No loss defined!')
        avg_losses['tot'] = []
        avg_losses_val['tot'] = []
        best_val['tot'] = float('inf')
        return loss_fn, avg_losses, avg_losses_val, best_val

    def configure_optimizers(self):
        param_groups = [{
            'params': [p for _, p in [*self.model.named_parameters_encoder()]],
            'lr': float(self.hparams.lr_multi_view),
            'weight_decay': float(self.hparams.weight_decay_multi_view),
        }]
        n_trainable = [p.numel() for p in param_groups[0]['params']]
        if self.hparams.use_encoder_prompting:
            param_groups.append({
                'params': [p for _, p in [
                    *self.model.named_parameters_prompt_tokens_encoder()]],
                'lr': float(self.hparams.lr_prompt_tokens),
                'weight_decay': float(self.hparams.weight_decay_prompt_tokens),
            })
            n_trainable += [p.numel() for p in param_groups[1]['params']]
        self.logger.info(f'{sum(n_trainable)} parameters trainable.')
        optimizer = AdamW(
            params=param_groups,
            lr=float(self.hparams.lr_multi_view),
            weight_decay=float(self.hparams.weight_decay_multi_view),
        )
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            # warmup_steps=int(len(self.train_loader.dataset)
            #                  / self.hparams.batch_size_multi_view * 0.2),
            # t_total=len(self.train_loader.dataset) // self.hparams.batch_size_multi_view,
            warmup_steps=self.hparams.warmup_steps_multi_view,
            t_total=self.hparams.t_total_multi_view,
        )
        self.logger.info(f'Size training set: {len(self.train_loader.dataset)}.')
        self.logger.info(f'Size validation set: {len(self.val_loader.dataset)}.')
        return optimizer, scheduler

    def train(self):
        self.self_supervised_learning()

    def self_supervised_learning(self):
        optimizer, scheduler = self.configure_optimizers()
        loss_fn, avg_losses, avg_losses_val, best_val = self.configure_losses()
        # Load checkpoint.
        start_epoch = 0
        if self.hparams.load_ckpt_backbone is True:
            ckpt = torch.load(self.hparams.load_ckpt_backbone_path)
            start_epoch = ckpt['current_epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.to(self.device)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        for _, loss in loss_fn.items():
            loss.to(self.device)
        self.model.to(self.device)
        self.logger.info(summary(self.model))
        self.logger.info(f'Tensorboard: {self.hparams.summary_dir}')
        self.logger.info(f'Using device: {self.device}.')
        self.logger.info(f'Start training from epoch {start_epoch:04d}.')
        # Start training.
        for epoch in range(start_epoch, self.hparams.max_epochs_multi_view + 1):
            # Training.
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                total_loss = torch.tensor(0.0, device=self.device)
                x = batch['image'].to(self.device)
                # Two views.
                x_i, y_rot_i = random_rotate(x)
                x_j, y_rot_j = random_rotate(x)
                x_i, mask_i = random_mask(
                    x_i, self.hparams.roi_size,
                    self.hparams.masking_shape,
                    self.hparams.masking_ratio,
                )
                x_j, mask_j = random_mask(
                    x_j, self.hparams.roi_size,
                    self.hparams.masking_shape,
                    self.hparams.masking_ratio,
                )
                # Two outputs.
                out_i = self.model(x_i)
                out_j = self.model(x_j)

                if self.hparams.use_reconstruction:
                    img = torch.cat([
                        x_i * mask_i,
                        x_j * mask_j,
                    ], dim=0)
                    rec = torch.cat([
                        out_i['reconstruction'] * mask_i,
                        out_j['reconstruction'] * mask_j,
                    ], dim=0)
                    rec_loss = loss_fn['rec'](
                        rec, img,
                    ) / (1 - self.hparams.masking_ratio)
                    total_loss += self.hparams.weight_rec * rec_loss
                    avg_losses['rec'].append(rec_loss)
                    if self.hparams.view and step % 5 == 0:
                        view_reconstruction(
                            name=batch['name'], n_slices=4,
                            ori_img=x_i * mask_i,
                            rec_img=out_i['reconstruction'] * mask_i,
                            epoch=epoch, step=step,
                        )
                if self.hparams.use_rotation_prediction:
                    rot_pred = torch.cat([
                        out_i['rotation_prediction'],
                        out_j['rotation_prediction'],
                    ], dim=0)
                    rot_target = torch.cat([y_rot_i, y_rot_j], dim=0)
                    rot_loss = loss_fn['rot'](rot_pred, rot_target)
                    total_loss += self.hparams.weight_rot * rot_loss
                    avg_losses['rot'].append(rot_loss)
                if self.hparams.use_contrastive_learning:
                    z1 = out_i['contrastive_coding']
                    z2 = out_j['contrastive_coding']
                    contrastive_loss = loss_fn['con'](z1, z2)
                    total_loss += self.hparams.weight_con * contrastive_loss
                    avg_losses['con'].append(contrastive_loss)
                if self.hparams.use_mutual_learning:
                    x_k, perm_k = random_permute(x_i)
                    out_k = self.model(x_k)
                    rec_no_perm = out_i['reconstruction']
                    rec_permuted = perm_k(out_k['reconstruction']).contiguous()
                    mutual_loss = loss_fn['mut'](
                        rec_permuted * mask_i,
                        rec_no_perm * mask_i,
                    ) / (1 - self.hparams.masking_ratio)
                    total_loss += mutual_loss
                    avg_losses['mut'].append(mutual_loss)

                avg_losses['tot'].append(total_loss)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 5 == 0:
                    self.logger.info(f'Learning rate in step {step} epoch {epoch}: '
                                     f'{scheduler.get_last_lr()[0]:.5f}.')
                    if self.hparams.use_reconstruction:
                        latest = avg_losses['rec'][-1]
                        self.logger.info(
                            f'Reconstruction training loss in step {step} epoch {epoch}: '
                            f'{latest:.5f}.')
                    if self.hparams.use_rotation_prediction:
                        latest = avg_losses['rot'][-1]
                        self.logger.info(
                            f'Rotation prediction training loss in step {step} epoch {epoch}: '
                            f'{latest:.5f}.')
                    if self.hparams.use_contrastive_learning:
                        latest = avg_losses['con'][-1]
                        self.logger.info(
                            f'Contrastive training loss in step {step} epoch {epoch}: '
                            f'{latest:.5f}.')
                    if self.hparams.use_mutual_learning:
                        latest = avg_losses['mut'][-1]
                        self.logger.info(
                            f'Mutual learning training loss in step {step} epoch {epoch}: '
                            f'{latest:.5f}.')

            for name, avg_loss in avg_losses.items():
                avg = torch.mean(torch.stack(avg_loss))
                self.writer.add_scalar(f'train_loss/{name}', avg, epoch)
                avg_loss.clear()
            if self.hparams.save_ckpt_backbone and epoch % 10 == 0:
                save_pth = self.hparams.save_ckpt_backbone_path
                save_pth.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'current_epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_pth / f'{epoch:04d}.pt')
                self.logger.info(f'Saved checkpoint for epoch {epoch:04d}.')

            # Validation.
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.val_loader):
                    total_loss_val = torch.tensor(0.0, device=self.device)
                    x = batch['image'].to(self.device)

                    x_i, y_rot_i = random_rotate(x)
                    x_j, y_rot_j = random_rotate(x)
                    x_i, mask_i = random_mask(
                        x_i, self.hparams.roi_size,
                        self.hparams.masking_shape,
                        self.hparams.masking_ratio,
                    )
                    x_j, mask_j = random_mask(
                        x_j, self.hparams.roi_size,
                        self.hparams.masking_shape,
                        self.hparams.masking_ratio,
                    )

                    out_i = self.model(x_i)
                    out_j = self.model(x_j)

                    if self.hparams.use_reconstruction:
                        img = torch.cat([
                            x_i * mask_i,
                            x_j * mask_j,
                        ], dim=0)
                        rec = torch.cat([
                            out_i['reconstruction'] * mask_i,
                            out_j['reconstruction'] * mask_j,
                        ], dim=0)
                        rec_loss = loss_fn['rec'](
                            rec, img,
                        ) / (1 - self.hparams.masking_ratio)
                        total_loss_val += self.hparams.weight_rec * rec_loss
                        avg_losses_val['rec'].append(rec_loss)
                    if self.hparams.use_rotation_prediction:
                        rot_pred = torch.cat([
                            out_i['rotation_prediction'],
                            out_j['rotation_prediction'],
                        ], dim=0)
                        rot_target = torch.cat([y_rot_i, y_rot_j], dim=0)
                        rot_loss = loss_fn['rot'](rot_pred, rot_target)
                        total_loss_val += self.hparams.weight_rot * rot_loss
                        avg_losses_val['rot'].append(rot_loss)
                    if self.hparams.use_contrastive_learning:
                        z1 = out_i['contrastive_coding']
                        z2 = out_j['contrastive_coding']
                        contrastive_loss = loss_fn['con'](z1, z2)
                        total_loss_val += self.hparams.weight_con * contrastive_loss
                        avg_losses_val['con'].append(contrastive_loss)
                    if self.hparams.use_mutual_learning:
                        x_k, perm_k = random_permute(x_i)
                        out_k = self.model(x_k)
                        rec_no_perm = out_i['reconstruction']
                        rec_permuted = perm_k(out_k['reconstruction']).contiguous()
                        mutual_loss = loss_fn['mut'](
                            rec_permuted * mask_i,
                            rec_no_perm * mask_i,
                        ) / (1 - self.hparams.masking_ratio)
                        total_loss_val += mutual_loss
                        avg_losses_val['mut'].append(mutual_loss)

                    avg_losses_val['tot'].append(total_loss_val)

                    if step % 5 == 0:
                        if self.hparams.use_reconstruction:
                            latest = avg_losses_val['rec'][-1]
                            self.logger.info(
                                f'Reconstruction validation loss in step {step} epoch {epoch}: '
                                f'{latest:.5f}.')
                        if self.hparams.use_rotation_prediction:
                            latest = avg_losses_val['rot'][-1]
                            self.logger.info(
                                f'Rotation prediction validation loss in step {step} epoch {epoch}: '
                                f'{latest:.5f}.')
                        if self.hparams.use_contrastive_learning:
                            latest = avg_losses_val['con'][-1]
                            self.logger.info(
                                f'Contrastive validation loss in step {step} epoch {epoch}: '
                                f'{latest:.5f}.')
                        if self.hparams.use_mutual_learning:
                            latest = avg_losses_val['mut'][-1]
                            self.logger.info(
                                f'Mutual learning validation loss in step {step} epoch {epoch}: '
                                f'{latest:.5f}.')

                for name, avg_loss in avg_losses_val.items():
                    avg = torch.mean(torch.stack(avg_loss))
                    if avg.item() < best_val[name]:
                        best_val[name] = avg.item()
                        self.logger.info(f'Best {name}_loss_val improved in epoch {epoch}.')
                    self.writer.add_scalar(f'val_loss/{name}', avg, epoch)
                    avg_loss.clear()