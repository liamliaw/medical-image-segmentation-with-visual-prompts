from monai.losses import FocalLoss, DiceLoss
import torch
from torch.optim import AdamW
from torchinfo import summary
from .utils import WarmupCosineSchedule, view_prototype_students_teacher, \
    MeanIoU, view_segmentation, map_label_indices
from .momentum_model import MomentumModel
from .losses import ClusteredPrototypeLoss


class StudentsTeacherTrainer:
    def __init__(self, hparams, backbone, train_loader, val_loader, logger, writer):
        super().__init__()
        self.hparams = hparams
        self.model = MomentumModel(hparams, backbone)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.writer = writer
        self.n_students = len(hparams.crop_size_students)
        if torch.cuda.is_available() and hparams.gpu != 0:
            self.device = torch.device('cuda')

    def configure_optimizers(self):
        if (self.hparams.training_mode == 'self_supervised_learning_all'
                or self.hparams.training_mode == 'supervised_learning_all'):
            param_groups = [{
                'params': [*[p for _, p in [*self.model.net_student.named_parameters_decoder()]],
                           *[p for _, p in [*self.model.net_student.named_parameters_encoder()]]],
                'lr': float(self.hparams.lr_students_teacher),
                'weight_decay': float(self.hparams.weight_decay_students_teacher),
            }]
            n_trainable = [p.numel() for p in param_groups[-1]['params']]
            if self.hparams.use_encoder_prompting:
                param_groups.append({
                    'params': [p for _, p in [
                        *self.model.net_student.named_parameters_prompt_tokens_encoder()]],
                    'lr': float(self.hparams.lr_prompt_tokens),
                    'weight_decay': float(self.hparams.weight_decay_prompt_tokens),
                })
                n_trainable += [p.numel() for p in param_groups[-1]['params']]
        elif (self.hparams.training_mode == 'self_supervised_learning_decoder'
              or self.hparams.training_mode == 'supervised_learning_decoder'):
            param_groups = [{
                'params': [p for _, p in [*self.model.net_student.named_parameters_decoder()]],
                'lr': float(self.hparams.lr_students_teacher),
                'weight_decay': float(self.hparams.weight_decay_students_teacher),
            }]
            n_trainable = [p.numel() for p in param_groups[-1]['params']]
        else:
            param_groups = [{}]
            n_trainable = 0

        if self.hparams.use_decoder_prompting:
            param_groups.append({
                'params': [p for _, p in [
                    *self.model.net_student.named_parameters_prompt_tokens_decoder()]],
                'lr': float(self.hparams.lr_prompt_tokens),
                'weight_decay': float(self.hparams.weight_decay_prompt_tokens),
            })
            n_trainable += [p.numel() for p in param_groups[-1]['params']]

        self.logger.info(f'{sum(n_trainable)} parameters trainable.')
        optimizer = AdamW(
            params=param_groups,
            lr=float(self.hparams.lr_students_teacher),
            weight_decay=float(self.hparams.weight_decay_students_teacher),
        )
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            # warmup_steps=int(len(self.train_loader.dataset)
            #                  / self.hparams.batch_size_students_teacher * 0.2),
            # t_total=len(self.train_loader.dataset) // self.hparams.batch_size_students_teacher,
            warmup_steps=self.hparams.warmup_steps_students_teacher,
            t_total=self.hparams.t_total_students_teacher,
        )
        self.logger.info(f'Size training set: {len(self.train_loader.dataset)}.')
        self.logger.info(f'Size validation set: {len(self.val_loader.dataset)}.')
        return optimizer, scheduler

    def configure_losses(self):
        # Losses
        loss_fn, avg_losses, avg_losses_val, best_val, metrics = {}, {}, {}, {}, {}
        if self.hparams.use_prototype_assignment:
            loss_fn['prt'] = ClusteredPrototypeLoss(
                reduction_factor=self.hparams.reduction_factor,
                fwhm=self.hparams.fwhm,
                k_means_iterations=self.hparams.k_means_iterations,
            )
            avg_losses['prt'] = []
            avg_losses_val['prt'] = []
            best_val['prt'] = float('inf')
        if (self.hparams.training_mode == 'supervised_learning_decoder'
            or self.hparams.training_mode == 'supervised_learning_all') \
                and self.hparams.use_real_label:
            loss_fn['seg'] = DiceLoss(
                include_background=self.hparams.include_background,
                to_onehot_y=True,
                softmax=True,
            )
            avg_losses['seg'] = []
            avg_losses_val['seg'] = []
            best_val['seg'] = float('inf')
            metrics['seg'] = MeanIoU(
                num_classes=self.hparams.output_channels_pretrain,
            )
        if not loss_fn:
            raise ValueError('No loss defined!')
        avg_losses['tot'] = []
        avg_losses_val['tot'] = []
        best_val['tot'] = float('inf')
        return loss_fn, avg_losses, avg_losses_val, best_val, metrics

    def train(self):
        optimizer, scheduler = self.configure_optimizers()
        loss_fn, avg_losses, avg_losses_val, best_val, metrics = self.configure_losses()
        # Load checkpoint.
        start_epoch = 0
        if self.hparams.load_ckpt_backbone is True:
            backbone_ckpt = torch.load(self.hparams.load_ckpt_backbone_path)
            if 'teacher_state_dict' in backbone_ckpt.keys():
                start_epoch = backbone_ckpt['current_epoch']
                backbone_state_dict = backbone_ckpt['model_state_dict']
                self.model.net_student.load_state_dict(backbone_state_dict)
                teacher_state_dict = backbone_ckpt['teacher_state_dict']
                self.model.net_teacher.load_state_dict(teacher_state_dict)
                self.model.to(self.device)
                optimizer.load_state_dict(backbone_ckpt['optimizer_state_dict'])
                scheduler.load_state_dict(backbone_ckpt['scheduler_state_dict'])
            else:
                backbone_state_dict = backbone_ckpt['model_state_dict']
                current_state_dict = self.model.net_student.state_dict()
                for (name, param) in backbone_state_dict.items():
                    if name in current_state_dict.keys():
                        current_state_dict[name] = param
                self.model.copy_state_dict()

        for _, loss in loss_fn.items():
            loss.to(self.device)
        self.model.to(self.device)
        self.logger.info(summary(self.model))
        self.logger.info(f'Tensorboard: {self.hparams.summary_dir}')
        self.logger.info(f'Using device: {self.device}.')
        self.logger.info(f'Start training from epoch {start_epoch:04d}.')
        for epoch in range(start_epoch, self.hparams.max_epochs_students_teacher + 1):
            # Training.
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                total_loss = torch.tensor(0.0, device=self.device)
                x_tch = batch['image'].to(self.device)
                coord_tch = batch['coord'].to(self.device)
                x_sts, coord_sts = [], []
                for i in range(self.n_students):
                    x_sts.append(batch[f'image_st_{i}'].to(self.device))
                    coord_sts.append(batch[f'coord_st_{i}'].to(self.device))
                # Update teacher first.
                self.model.update_teacher()
                # Update student.
                out_sts, out_tch = self.model(x_sts, x_tch)

                if self.hparams.use_prototype_assignment:
                    contrastive_loss = loss_fn['prt'](
                        emb_s=[out_sts[i]['latent_outputs'] for i in range(len(out_sts))],
                        emb_t=out_tch['latent_outputs'],
                        coord_s=coord_sts,
                        coord_t=coord_tch,
                    )
                    avg_losses['prt'].append(contrastive_loss)
                    total_loss += contrastive_loss
                    if self.hparams.view and epoch % 1 == 0 and step % 5 == 0:
                        view_prototype_students_teacher(
                            name=batch['name'], n_slices=4,
                            chs=self.hparams.hidden_channels[0],
                            prt_tch=out_tch['latent_outputs'], img_tch=x_tch,
                            prt_sts=[out['latent_outputs'] for out in out_sts],
                            img_sts=x_sts,
                            epoch=epoch, step=step,
                        )
                if (self.hparams.training_mode == 'supervised_learning_decoder'
                    or self.hparams.training_mode == 'supervised_learning_all') \
                        and self.hparams.use_real_label:
                    seg_true = map_label_indices(
                        batch['mask_st_0'], self.hparams.active_labels_pretrain,
                    ).to(self.device)
                    # seg_true = batch['mask_st_0'].to(self.device)
                    seg_loss = loss_fn['seg'](out_sts[0]['seg_pred'], seg_true)
                    avg_losses['seg'].append(seg_loss)
                    total_loss += seg_loss
                    metrics['seg'].update(
                        preds=out_sts[0]['seg_pred'],
                        target=seg_true,
                    )
                    if epoch % 1 == 0 and step % 5 == 0 and self.hparams.view:
                        view_segmentation(
                            name=batch['name'], n_slices=4,
                            seg_pred=out_sts[0]['seg_pred'],
                            seg_target=seg_true,
                            n_classes=self.hparams.output_channels_pretrain,
                            epoch=epoch, step=step,
                        )
                avg_losses['tot'].append(total_loss)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 5 == 0:
                    self.logger.info(f'Learning rate in step {step} epoch {epoch}: '
                                     f'{scheduler.get_last_lr()[0]:.5f}.')
                    if self.hparams.use_prototype_assignment:
                        latest = avg_losses['prt'][-1]
                        self.logger.info(
                            f'Contrastive training loss in step {step} epoch {epoch}: '
                            f'{latest:.4f}.')
                    if (self.hparams.training_mode == 'supervised_learning_decoder'
                        or self.hparams.training_mode == 'supervised_learning_all') \
                            and self.hparams.use_real_label:
                        latest = avg_losses['seg'][-1]
                        self.logger.info(
                            f'Segmentation training loss in step {step} epoch {epoch}: '
                            f'{latest:.4f}.')

            for name, avg_loss in avg_losses.items():
                avg = torch.mean(torch.stack(avg_loss))
                self.writer.add_scalar(f'train_loss/{name}', avg, epoch)
                avg_loss.clear()
            if bool(metrics):
                for name, metric in metrics.items():
                    value = metric.compute()
                    self.writer.add_scalar(f'train_metric/{name}', value, epoch)
                    metric.reset()
            if self.hparams.save_ckpt_backbone and epoch % 10 == 0:
                save_pth = self.hparams.save_ckpt_backbone_path
                save_pth.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'current_epoch': epoch + 1,
                    'model_state_dict': self.model.net_student.state_dict(),
                    'teacher_state_dict': self.model.net_teacher.state_dict(),
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
                    x_tch = batch['image'].to(self.device)
                    coord_tch = batch['coord'].to(self.device)
                    x_sts, coord_sts = [], []
                    for i in range(self.n_students):
                        x_sts.append(batch[f'image_st_{i}'].to(self.device))
                        coord_sts.append(batch[f'coord_st_{i}'].to(self.device))

                    out_sts, out_tch = self.model(x_sts, x_tch)

                    if self.hparams.use_prototype_assignment:
                        contrastive_loss = loss_fn['prt'](
                            emb_s=[out_sts[i]['latent_outputs'] for i in range(len(out_sts))],
                            emb_t=out_tch['latent_outputs'],
                            coord_s=coord_sts,
                            coord_t=coord_tch,
                        )
                        avg_losses_val['prt'].append(contrastive_loss)
                        total_loss_val += contrastive_loss
                    if (self.hparams.training_mode == 'supervised_learning_decoder'
                        or self.hparams.training_mode == 'supervised_learning_all') \
                            and self.hparams.use_real_label:
                        seg_true = map_label_indices(
                            batch['mask_st_0'], self.hparams.active_labels_pretrain,
                        ).to(self.device)
                        seg_loss = loss_fn['seg'](out_sts[0]['seg_pred'], seg_true)
                        avg_losses_val['seg'].append(seg_loss)
                        total_loss_val += seg_loss
                        metrics['seg'].update(
                            preds=out_sts[0]['seg_pred'],
                            target=seg_true,
                        )
                    avg_losses_val['tot'].append(total_loss_val)

                    if step % 5 == 0:
                        if self.hparams.use_prototype_assignment:
                            latest = avg_losses_val['prt'][-1]
                            self.logger.info(
                                f'Contrastive validation loss in step {step} epoch {epoch}: '
                                f'{latest:.4f}.')
                        if (self.hparams.training_mode == 'supervised_learning_decoder'
                            or self.hparams.training_mode == 'supervised_learning_all') \
                                and self.hparams.use_real_label:
                            latest = avg_losses_val['seg'][-1]
                            self.logger.info(
                                f'Segmentation validation loss in step {step} epoch {epoch}: '
                                f'{latest:.4f}.')

                for name, avg_loss in avg_losses_val.items():
                    avg = torch.mean(torch.stack(avg_loss))
                    if avg.item() < best_val[name]:
                        best_val[name] = avg.item()
                        self.logger.info(f'Best {name}_loss_val improved in epoch {epoch}.')
                    self.writer.add_scalar(f'val_loss/{name}', avg, epoch)
                    avg_loss.clear()
                if bool(metrics):
                    for name, metric in metrics.items():
                        value = metric.compute()
                        self.writer.add_scalar(f'val_metric/{name}', value, epoch)
                        metric.reset()
