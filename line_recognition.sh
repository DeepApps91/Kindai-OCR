#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python -u ./translate_line.py -k 10 ./pretrain/WAP_params.pkl \
	./pretrain/kindai_voc.txt \
	./data_kindai/val.pkl \
	./data_kindai/caption_val.txt \
	./result/test_decode_result.txt \
	./result/test.wer
