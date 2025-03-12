$1 ~/prompt-tuning/src/main.py \
--training-mode self_supervised_learning_encoder \
--configs $2 \
--save-ckpt-backbone \
--backbone swin_unetr \
--use-encoder-prompting \
--run-name "ssl_enc_use_ep_$3"

# $1 is the python command, $2 is the config file name, $3 is the name you set
# The args can be set in command line.