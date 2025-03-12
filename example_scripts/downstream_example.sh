#!/bin/bash


# This is an example of a set of downstream training, the downstream training
# can also be run individually.
PY=~/venv/bin/python3
prompt_tokens_base_dir="/set/your/path/here" # The dir that saves prompt tokens in the configuration.
logs_base_dir="/set/your/path/here" # The dir that saves logs in the configuration.
result_file="/set/a/file/path/to/save/the/result" # A file that saves the results.


run_list1=(
  "downstream_ssl_dec_no_ep_no_dp"
  "downstream_ssl_dec_no_ep_no_dp_test_ep"
  "downstream_ssl_dec_no_ep_no_dp_test_dp"
  "downstream_ssl_dec_no_ep_no_dp_test_ap"
  "downstream_ssl_dec_no_ep_use_dp"
  "downstream_ssl_dec_no_ep_use_dp_test_ep"
  "downstream_ssl_dec_use_ep_no_dp"
  "downstream_ssl_dec_use_ep_no_dp_test_dp"
  "downstream_ssl_dec_use_ep_use_dp"
)
test_list1=(
  "test_ssl_dec_no_ep_no_dp"
  "test_ssl_dec_no_ep_no_dp_test_ep"
  "test_ssl_dec_no_ep_no_dp_test_dp"
  "test_ssl_dec_no_ep_no_dp_test_ap"
  "test_ssl_dec_no_ep_use_dp"
  "test_ssl_dec_no_ep_use_dp_test_ep"
  "test_ssl_dec_use_ep_no_dp"
  "test_ssl_dec_use_ep_no_dp_test_dp"
  "test_ssl_dec_use_ep_use_dp"
)
ckpt_list1=(
  "fit_swin_unetr_0710003006_ssl_dec_no_ep_no_dp_tcia/0400.pt"
  "fit_swin_unetr_0710003006_ssl_dec_no_ep_no_dp_tcia/0400.pt"
  "fit_swin_unetr_0710003006_ssl_dec_no_ep_no_dp_tcia/0400.pt"
  "fit_swin_unetr_0710003006_ssl_dec_no_ep_no_dp_tcia/0400.pt"
  "fit_swin_unetr_0710013054_ssl_dec_no_ep_use_dp_tcia/0400.pt"
  "fit_swin_unetr_0710013054_ssl_dec_no_ep_use_dp_tcia/0400.pt"
  "fit_swin_unetr_0710024252_ssl_dec_use_ep_no_dp_tcia/0400.pt"
  "fit_swin_unetr_0710024252_ssl_dec_use_ep_no_dp_tcia/0400.pt"
  "fit_swin_unetr_0710035108_ssl_dec_use_ep_use_dp_tcia/0400.pt"
)
use_ep_flag_list1=(
  ""
  "--use-encoder-prompting"
  ""
  "--use-encoder-prompting"
  ""
  "--use-encoder-prompting"
  "--use-encoder-prompting"
  "--use-encoder-prompting"
  "--use-encoder-prompting"
)
use_dp_flag_list1=(
  ""
  ""
  "--use-decoder-prompting"
  "--use-decoder-prompting"
  "--use-decoder-prompting"
  "--use-decoder-prompting"
  ""
  "--use-decoder-prompting"
  "--use-decoder-prompting"
)
configs1="tcia.yml"
run_name1="tcia_bi_ssl_label_6"

for ((i=0; i<${#run_list1[@]}; i++)); do
  $PY ~/prompt-tuning/src/main.py \
  --training-mode downstream \
  --configs $configs1 \
  --load-ckpt-backbone \
  --load-ckpt-backbone-path ${ckpt_list1[i]} \
  --save-ckpt-prompt-tokens \
  ${use_ep_flag_list1[i]} \
  ${use_dp_flag_list1[i]} \
  --run-name "${run_list1[i]}_$run_name1"

  prompt_token_ckpt="$(ls -t "$prompt_tokens_base_dir" | head -1)/0300.pt"
  for j in {1..5}; do
    $PY ~/prompt-tuning/src/main.py \
    --mode test \
    --training-mode downstream \
    --configs $configs1 \
    --load-ckpt-prompt-tokens \
    --load-ckpt-prompt-tokens-path $prompt_token_ckpt \
    ${use_ep_flag_list1[i]} \
    ${use_dp_flag_list1[i]} \
    --run-name "${test_list1[i]}_$run_name1"
    log_file="$(ls -t "$logs_base_dir" | head -1)/log.txt"
    cat $logs_base_dir$log_file >> $result_file
  done
done
