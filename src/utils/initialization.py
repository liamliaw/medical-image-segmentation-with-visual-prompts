from datasets import *
from modules import *
from .loggers import get_logger, get_summary_writer


def setup_fitting(hparams):
    backbone = select_backbone(hparams)
    train_loader, val_loader = select_fitting_loader(hparams)
    logger = get_logger(hparams.log_dir)
    writer = get_summary_writer(hparams.summary_dir)
    trainer = select_trainer(
        hparams, backbone, train_loader, val_loader, logger, writer)
    return trainer


def setup_testing(hparams):
    backbone = select_backbone(hparams)
    loader = select_testing_loader(hparams)
    logger = get_logger(hparams.log_dir)
    writer = get_summary_writer(hparams.summary_dir)
    trainer = select_trainer(hparams, backbone, loader, None, logger, writer)
    return trainer


def select_backbone(hparams):
    if hparams.backbone == 'swin_unetr':
        return SwinUnetR
    else:
        raise NotImplementedError()


def select_trainer(hparams, backbone, train_loader, val_loader, logger, writer):
    if hparams.training_mode == 'self_supervised_learning_encoder':
        return MultiViewTrainer(
            hparams, backbone, train_loader, val_loader, logger, writer)
    elif (hparams.training_mode == 'self_supervised_learning_decoder'
          or hparams.training_mode == 'self_supervised_learning_all'
          or hparams.training_mode == 'supervised_learning_decoder'
          or hparams.training_mode == 'supervised_learning_all'):
        return StudentsTeacherTrainer(
            hparams, backbone, train_loader, val_loader, logger, writer)
    elif hparams.training_mode == 'downstream':
        return SegmentationTrainer(
            hparams, backbone, train_loader, val_loader, logger, writer)
    else:
        raise NotImplementedError()


def select_fitting_loader(hparams):
    if hparams.training_mode == 'downstream':
        return get_fit_loader_downstream(hparams)
    elif (hparams.training_mode == 'self_supervised_learning_encoder'
          or hparams.training_mode == 'self_supervised_learning_decoder'
          or hparams.training_mode == 'self_supervised_learning_all'):
        return get_fit_loader_self_supervised_learning(hparams)
    elif (hparams.training_mode == 'supervised_learning_decoder'
          or hparams.training_mode == 'supervised_learning_all'):
        return get_fit_loader_supervised_learning(hparams)
    else:
        raise NotImplementedError()


def select_testing_loader(hparams):
    return get_test_loader_downstream(hparams)
