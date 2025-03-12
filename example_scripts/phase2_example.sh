$1 ~/prompt-tuning/src/main.py \
--training-mode self_supervised_learning_decoder \
--configs $2 \
--load-ckpt-backbone \
--load-ckpt-backbone-path $4 \
--save-ckpt-backbone \
--use-encoder-prompting \
--use-decoder-prompting \
--backbone swin_unetr \
--run-name "ssl_dec_use_ep_use_dp_$3"

# $1 is the python command, $2 is the config file name, $3 is the name you set
# The args can be set in command line.
