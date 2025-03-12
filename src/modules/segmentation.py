from monai.losses import DiceLoss, DiceFocalLoss
import torch
from torch.optim import Adam, AdamW, lr_scheduler
from torchinfo import summary
from .utils import (
    WarmupCosineSchedule,
    view_segmentation,
    MeanIoU, DiceCoefficient,
    map_label_indices,
)


class SegmentationTrainer:
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

    def configure_optimizers(self):
        params = [p for _, p in [
            *self.model.named_parameters_downstream(),
        ]]
        n_trainable = [p.numel() for p in params]
        self.logger.info(f'{sum(n_trainable)} parameters trainable.')
        optimizer = AdamW(
            params=params,
            lr=float(self.hparams.lr_downstream),
            weight_decay=float(self.hparams.weight_decay_downstream),
        )
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
        self.logger.info(f'Size training set: {len(self.train_loader.dataset)}.')
        self.logger.info(f'Size validation set: {len(self.val_loader.dataset)}.')
        return optimizer, scheduler

    def configure_losses(self):
        loss_fn, avg_losses, metrics, avg_losses_val, best_val = {}, {}, {}, {}, {}
        if True:
            loss_fn['seg'] = DiceFocalLoss(
                include_background=self.hparams.include_background,
                to_onehot_y=True,
                softmax=True,
                gamma=4.0,
                # alpha=0.05,
            )
            metrics['seg'] = MeanIoU(
                num_classes=self.hparams.output_channels_downstream,
            )
            avg_losses['seg'] = []
            avg_losses_val['seg'] = []
            best_val['seg'] = float('inf')
        if not loss_fn:
            raise ValueError('No loss defined!')
        avg_losses['tot'] = []
        avg_losses_val['tot'] = []
        best_val['tot'] = float('inf')
        return loss_fn, avg_losses, metrics, avg_losses_val, best_val

    def train(self):
        optimizer, scheduler = self.configure_optimizers()
        loss_fn, avg_losses, metrics, avg_losses_val, best_val = self.configure_losses()
        # Load checkpoint.
        start_epoch = 0
        if self.hparams.load_ckpt_backbone is True:
            backbone_ckpt = torch.load(self.hparams.load_ckpt_backbone_path)
            backbone_state_dict = backbone_ckpt['model_state_dict']
            current_state_dict = self.model.state_dict()
            for (name, param) in backbone_state_dict.items():
                # print(f'Load {name}.')
                current_state_dict[name] = param
        if self.hparams.load_ckpt_prompt_tokens is True:
            ckpt = torch.load(self.hparams.load_ckpt_instruction_path)
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
        for epoch in range(start_epoch, self.hparams.max_epochs_downstream + 1):
            # Training.
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            self.model.train()
            for step, batch in enumerate(self.train_loader):
                total_loss = torch.tensor(0.0, device=self.device)
                x = batch['image'].to(self.device)
                seg_true = map_label_indices(
                    batch['mask'], self.hparams.active_labels_downstream,
                ).to(self.device)
                if True:
                    out = self.model(x)
                    seg_loss = loss_fn['seg'](out['downstream'], seg_true)
                    total_loss += seg_loss
                    avg_losses['seg'].append(seg_loss)
                    metrics['seg'].update(
                        preds=out['downstream'],
                        target=seg_true,
                    )
                    if epoch % 1 == 0 and step % 5 == 0 and self.hparams.view:
                        view_segmentation(
                            name=batch['name'], n_slices=4,
                            seg_pred=out['downstream'],
                            seg_target=seg_true,
                            n_classes=self.hparams.output_channels_downstream,
                            epoch=epoch, step=step,
                        )
                avg_losses['tot'].append(total_loss)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if step % 5 == 0:
                    self.logger.info(f'Learning rate in step {step} epoch {epoch}: '
                                     f'{scheduler.get_last_lr()[0]:.5f}.')
                    if True:
                        latest = avg_losses['seg'][-1]
                        score = metrics['seg'].compute()
                        self.logger.info(
                            f'Segmentation training loss in step {step} epoch {epoch}: '
                            f'{latest:.5f}.')
                        self.logger.info(
                            f'Segmentation training score in step {step} epoch {epoch}: '
                            f'{score:.5f}.')

            for name, avg_loss in avg_losses.items():
                avg = torch.mean(torch.stack(avg_loss))
                self.writer.add_scalar(f'train_loss/{name}', avg, epoch)
                avg_loss.clear()
            for name, metric in metrics.items():
                value = metric.compute()
                self.writer.add_scalar(f'train_metric/{name}', value, epoch)
                metric.reset()
            if self.hparams.save_ckpt_prompt_tokens and epoch % 20 == 0:
                save_pth = self.hparams.save_ckpt_prompt_tokens_path
                save_pth.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'current_epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_pth / f'{epoch:04d}.pt')
                self.logger.info(f'Saved checkpoint for epoch {epoch:04d}.')
            scheduler.step()

            # Validation.
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.val_loader):
                    total_loss_val = torch.tensor(0.0, device=self.device)
                    x = batch['image'].to(self.device)
                    seg_true = map_label_indices(
                        batch['mask'], self.hparams.active_labels_downstream,
                    ).to(self.device)
                    if True:
                        out = self.model(x)
                        seg_loss = loss_fn['seg'](out['downstream'], seg_true)
                        total_loss_val += seg_loss
                        avg_losses_val['seg'].append(seg_loss)
                        metrics['seg'].update(
                            preds=out['downstream'],
                            target=seg_true,
                        )
                    avg_losses_val['tot'].append(total_loss_val)

                    if step % 5 == 0:
                        if True:
                            latest = avg_losses_val['seg'][-1]
                            score = metrics['seg'].compute()
                            self.logger.info(
                                f'Segmentation validation loss in step {step} epoch {epoch}: '
                                f'{latest:.5f}.')
                            self.logger.info(
                                f'Segmentation validation score in step {step} epoch {epoch}: '
                                f'{score:.5f}.')

                for name, avg_loss in avg_losses_val.items():
                    avg = torch.mean(torch.stack(avg_loss))
                    if avg.item() < best_val[name]:
                        best_val[name] = avg.item()
                        self.logger.info(f'Best {name}_loss_val improved in epoch {epoch}.')
                    self.writer.add_scalar(f'val_loss/{name}', avg, epoch)
                    avg_loss.clear()
                for name, metric in metrics.items():
                    value = metric.compute()
                    self.writer.add_scalar(f'val_metric/{name}', value, epoch)
                    metric.reset()


    # Only for testing.
    def test(self):
        test_metrics, metric_vals = {}, {}
        test_metrics['iou'] = MeanIoU(
            num_classes=self.hparams.output_channels_downstream)
        test_metrics['dcc'] = DiceCoefficient(
            num_classes=self.hparams.output_channels_downstream)
        metric_vals['iou'] = []
        metric_vals['dcc'] = []
        # Load checkpoint.
        start_epoch = 0
        if self.hparams.load_ckpt_prompt_tokens is True:
            ckpt = torch.load(self.hparams.load_ckpt_prompt_tokens_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        for epoch in range(start_epoch, 1):
            if self.device == torch.device('cuda'):
                torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                for step, batch in enumerate(self.train_loader):
                    x = batch['image']  # .to(self.device)
                    seg_true = map_label_indices(
                        batch['mask'], self.hparams.active_labels_downstream,
                    )  # .to(self.device)

                    # Generate fixed sliding windows for testing.
                    image_size = x.size()[2:]
                    window_size = self.hparams.roi_size
                    stride = [i // 2 for i in self.hparams.roi_size]
                    adjusted_size = [0, 0, 0]
                    slc = [None, None, None]
                    for i in range(len(image_size)):
                        adjusted_size[i] = (image_size[i] - window_size[i]) \
                            // stride[i] * stride[i] + window_size[i]
                        start = (image_size[i] - adjusted_size[i]) // 2
                        slc[i] = slice(start, start + adjusted_size[i])
                    adjusted_x = x.as_tensor()[:, :, slc[0], slc[1], slc[2]]
                    adjusted_seg = seg_true.as_tensor()[:, :, slc[0], slc[1], slc[2]]
                    # Get sliding window.
                    x_slc = adjusted_x.unfold(
                        2, window_size[0], stride[0]).unfold(
                        3, window_size[1], stride[1]).unfold(
                        4, window_size[2], stride[2]).flatten(2, 4).permute(
                        2, 1, 0, 3, 4, 5).squeeze(2).contiguous()
                    seg_slc = adjusted_seg.unfold(
                        2, window_size[0], stride[0]).unfold(
                        3, window_size[1], stride[1]).unfold(
                        4, window_size[2], stride[2]).flatten(2, 4).permute(
                        2, 1, 0, 3, 4, 5).squeeze(2).contiguous()

                    # Generating sub batches
                    num_full_batches = x_slc.size(0) // 10  # hardcoded batch size
                    img_batches = []
                    seg_batches = []
                    for i in range(num_full_batches):
                        start = i * 10
                        end = start + 10
                        img_batches.append(x_slc[start:end])
                        seg_batches.append(seg_slc[start:end])
                    # Handle the last batch if it's not a full batch.
                    if x_slc.size(0) % 10 != 0:
                        img_batches.append(x_slc[num_full_batches * 10:])
                        seg_batches.append(seg_slc[num_full_batches * 10:])

                    for idx, (img_batch, seg_batch) in enumerate(zip(img_batches, seg_batches)):
                        img_batch = img_batch.to(device=self.device)
                        seg_batch = seg_batch.to(device=self.device)

                        out = self.model(img_batch)
                        test_metrics['iou'].update(
                            preds=out['downstream'],
                            target=seg_batch,
                        )
                        test_metrics['dcc'].update(
                            preds=out['downstream'],
                            target=seg_batch,
                        )
                        if idx % 10 == 0 and self.hparams.view:
                            name = '_'.join(
                                self.hparams.load_ckpt_prompt_tokens_path.parent.name.split('_')[5:]) \
                                   + '_' + batch['name'][0]
                            view_segmentation(
                                name=name, n_slices=4,
                                seg_pred=out['downstream'],
                                seg_target=seg_batch,
                                img=img_batch,
                                n_classes=self.hparams.output_channels_downstream,
                                epoch=epoch, step=step,
                            )
                    for name, metric in test_metrics.items():
                        value = metric.compute()
                        metric_vals[name].append(float(value))
                        metric.reset()
        for name, values in metric_vals.items():
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            self.logger.info(f'{self.hparams.run_name}: {name}: {mean:.4f} +/- {std:.4f}.')