#!/bin/bash

PY=~/venv/bin/python3
logs_base_dir="the/dir/for/log"  # The log_dir in the configuration.
result_file="/set/a/file/path/to/save/the/result" # A file for saving the results.

use_ep_flag_list=(
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
use_dp_flag_list=(
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
configs="acdc.yml"
run_name="acdc_bi_ssl_label_3"

ckpt="0300.pt"

test_list=(
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
ckpt_list=(
  "fit_swin_unetr_0711001537_downstream_ssl_dec_no_ep_no_dp_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711002712_downstream_ssl_dec_no_ep_no_dp_test_ep_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711005932_downstream_ssl_dec_no_ep_no_dp_test_dp_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711012200_downstream_ssl_dec_no_ep_no_dp_test_ap_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711015849_downstream_ssl_dec_no_ep_use_dp_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711022118_downstream_ssl_dec_no_ep_use_dp_test_ep_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711025803_downstream_ssl_dec_use_ep_no_dp_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711033024_downstream_ssl_dec_use_ep_no_dp_test_dp_acdc_bi_ssl_label_3/$ckpt"
  "fit_swin_unetr_0711040712_downstream_ssl_dec_use_ep_use_dp_acdc_bi_ssl_label_3/$ckpt"
)

for ((i=0; i<${#test_list[@]}; i++)); do
  $PY ~/prompt-tuning/src/main.py \
  --mode test \
  --training-mode downstream \
  --configs $configs \
  --load-ckpt-prompt-tokens \
  --load-ckpt-prompt-tokens-path ${ckpt_list[i]}\
  ${use_ep_flag_list[i]} \
  ${use_dp_flag_list[i]} \
  --run-name "${test_list[i]}_$run_name"
  log_file="$(ls -t "$logs_base_dir" | head -1)/log.txt"
  cat $logs_base_dir$log_file >> $result_file
done
