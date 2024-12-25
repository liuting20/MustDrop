#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
MODEL=llava-v1.5-7b

python -m llava.eval.model_vqa_loader \
    --model-path /home/mi/work/liuting/SparseVLMs/liuhaotian/$MODEL \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --sparse \
    --global_thr 0.005 \
    --individual_thr 0.001

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.5-7b.json

