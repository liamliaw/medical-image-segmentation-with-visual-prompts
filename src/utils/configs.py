from argparse import Namespace
from datetime import datetime
from pathlib import Path
import yaml

# Default paths for configs and saving checkpoints and logs
DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / 'configurations'
DEFAULT_CKPT_DIR = Path(__file__).parent.parent.parent / 'checkpoints'
DEFAULT_ROOT_LOG_DIR = Path(__file__).parent.parent.parent / 'logs'


# Merge argparse configs and yaml configs.
def get_full_configs(args: Namespace):
    full_configs = vars(args)
    # More configs are stored in a separate yaml file.
    config_pth = Path(args.configs)
    if config_pth.is_absolute() and config_pth.exists():
        with open(config_pth, 'r') as f:
            more_configs = yaml.safe_load(f)
    elif config_pth.expanduser().exists():
        with open(config_pth.expanduser(), 'r') as f:
            more_configs = yaml.safe_load(f)
    elif (DEFAULT_CONFIG_DIR / config_pth).exists():
        with open(DEFAULT_CONFIG_DIR / config_pth, 'r') as f:
            more_configs = yaml.safe_load(f)
    else:
        raise FileNotFoundError('The configuration file does not exist!')
    full_configs.update(more_configs)

    # Path for loading a checkpoint for backbone model.
    if args.load_ckpt_backbone is True:
        if args.load_ckpt_backbone_path != '':
            load_ckpt_backbone_pth = Path(args.load_ckpt_backbone_path)
            if load_ckpt_backbone_pth.is_absolute() \
                and load_ckpt_backbone_pth.exists():
                full_configs['load_ckpt_backbone_path'] = load_ckpt_backbone_pth
            elif load_ckpt_backbone_pth.expanduser().exists():
                full_configs['load_ckpt_backbone_path'] = \
                    load_ckpt_backbone_pth.expanduser()
            elif (DEFAULT_CKPT_DIR / 'backbone' / load_ckpt_backbone_pth).exists():
                full_configs['load_ckpt_backbone_path'] = \
                    DEFAULT_CKPT_DIR / 'backbone' / load_ckpt_backbone_pth
            else:
                raise FileNotFoundError(
                    'The backbone checkpoint does not exist!')
        else:
            raise ValueError('The backbone checkpoint path is empty!')
    # Path for loading a checkpoint for instructions.
    if args.load_ckpt_prompt_tokens is True:
        if args.load_ckpt_prompt_tokens_path != '':
            load_ckpt_prompt_tokens_pth = Path(args.load_ckpt_prompt_tokens_path)
            if load_ckpt_prompt_tokens_pth.is_absolute() \
                    and load_ckpt_prompt_tokens_pth.exists():
                full_configs['load_ckpt_prompt_tokens_path'] = load_ckpt_prompt_tokens_pth
            elif load_ckpt_prompt_tokens_pth.expanduser().exists():
                full_configs['load_ckpt_prompt_tokens_path'] = \
                    load_ckpt_prompt_tokens_pth.expanduser()
            elif (DEFAULT_CKPT_DIR / 'prompt_tokens' / load_ckpt_prompt_tokens_pth).exists():
                full_configs['load_ckpt_prompt_tokens_path'] = \
                    DEFAULT_CKPT_DIR / 'prompt_tokens' / load_ckpt_prompt_tokens_pth
            else:
                raise FileNotFoundError(
                    'The prompt tokens checkpoint does not exist!')
        else:
            raise ValueError('The prompt tokens checkpoint path is empty!')
    # Path for saving checkpoint for backbone model.
    timestamp = datetime.now().strftime('%m%d%H%M%S')
    run_name = (full_configs['mode'] + '_'
                + full_configs['backbone'] + '_'
                + timestamp + '_' + args.run_name)
    if args.save_ckpt_backbone is True:
        if args.save_ckpt_backbone_path != '':
            save_ckpt_backbone_pth = Path(args.save_ckpt_backbone_path)
            if save_ckpt_backbone_pth.is_absolute():
                full_configs['save_ckpt_backbone_path'] = save_ckpt_backbone_pth
            elif '~' in str(save_ckpt_backbone_pth):
                full_configs['save_ckpt_backbone_path'] = \
                    save_ckpt_backbone_pth.expanduser()
            else:
                full_configs['save_ckpt_backbone_path'] = \
                    DEFAULT_CKPT_DIR / 'backbone' / save_ckpt_backbone_pth
        else:
            full_configs['save_ckpt_backbone_path'] = \
                DEFAULT_CKPT_DIR / 'backbone' / run_name
    # Path for saving checkpoint for instructions.
    if args.save_ckpt_prompt_tokens is True:
        if args.save_ckpt_prompt_tokens_path != '':
            save_ckpt_prompt_tokens_pth = Path(args.save_ckpt_prompt_tokens_path)
            if save_ckpt_prompt_tokens_pth.is_absolute():
                full_configs['save_ckpt_prompt_tokens_path'] = save_ckpt_prompt_tokens_pth
            elif '~' in str(save_ckpt_prompt_tokens_pth):
                full_configs['save_ckpt_prompt_tokens_path'] = \
                    save_ckpt_prompt_tokens_pth.expanduser()
            else:
                full_configs['save_ckpt_prompt_tokens_path'] = \
                    DEFAULT_CKPT_DIR / 'prompt_tokens' / save_ckpt_prompt_tokens_pth
        else:
            full_configs['save_ckpt_prompt_tokens_path'] = \
                DEFAULT_CKPT_DIR / 'prompt_tokens' / run_name

    # Paths for general logger and tensorboard summary writer.
    log_dir = Path(DEFAULT_ROOT_LOG_DIR) / run_name
    full_configs['log_dir'] = log_dir
    full_configs['summary_dir'] = log_dir / 'summary'

    return Namespace(**full_configs)