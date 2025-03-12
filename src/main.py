from argparse import ArgumentParser
from utils import get_full_configs, setup_fitting, setup_testing
from argparse import Namespace


def main(mode: str, hparams: Namespace):
    if mode == 'fit':
        trainer = setup_fitting(hparams)
        trainer.train()
    elif mode == 'test':
        trainer = setup_testing(hparams)
        trainer.test()
    else:
        raise ValueError(f'The mode is not available.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', default='fit', type=str)
    parser.add_argument('--training-mode', default='self_supervised_learning_all', type=str)
    parser.add_argument('--configs', default='example_configs.yml', type=str)

    parser.add_argument('--load-ckpt-backbone', action='store_true', default=False)
    parser.add_argument('--load-ckpt-backbone-path', default='', type=str)
    parser.add_argument('--save-ckpt-backbone', action='store_true', default=False)
    parser.add_argument('--save-ckpt-backbone-path', default='', type=str)

    parser.add_argument('--load-ckpt-prompt-tokens', action='store_true', default=False)
    parser.add_argument('--load-ckpt-prompt-tokens-path', default='', type=str)
    parser.add_argument('--save-ckpt-prompt-tokens', action='store_true', default=False)
    parser.add_argument('--save-ckpt-prompt-tokens-path', default='', type=str)

    parser.add_argument('--use-encoder-prompting', action='store_true', default=False)
    parser.add_argument('--use-decoder-prompting', action='store_true', default=False)

    parser.add_argument('--backbone', default='swin_unetr', type=str)
    parser.add_argument('--run-name', default='', type=str)
    # Extra configs are set in the configuration file!
    # See ../configurations/example_configs.yml.

    args = parser.parse_args()
    configs = get_full_configs(args)

    main(args.mode, configs)