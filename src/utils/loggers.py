import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def get_logger(log_dir: Path):
    log_dir.mkdir(exist_ok=True, parents=True)
    log_format = (
        'Time: %(asctime)s | Logger: %(name)s | '
        'Level: %(levelname)s | Massage: %(message)s'
    )
    logging.basicConfig(
        filename=log_dir / 'log.txt',
        filemode='a',
        format=log_format,
        level=logging.INFO
    )
    logger = logging.getLogger('Root')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    return logger


# Summary writer for tensorboard.
def get_summary_writer(summary_dir: Path):
    summary_dir.mkdir(exist_ok=True, parents=True)
    return SummaryWriter(str(summary_dir))