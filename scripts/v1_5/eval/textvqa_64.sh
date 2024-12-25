#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
MODEL=llava-v1.5-7b

python3 -m llava.eval.model_vqa_loader \
    --model-path /home/mi/work/liuting/SparseVLMs/liuhaotian/$MODEL \
    --question-file /home/mi/work/liuting/SparseVLMs/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /home/mi/work/liuting/SparseVLMs/playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --global_thr 0.011 \
    --individual_thr 0.01

python3 -m llava.eval.eval_textvqa \
    --annotation-file /home/mi/work/liuting/SparseVLMs/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$MODEL.jsonl
