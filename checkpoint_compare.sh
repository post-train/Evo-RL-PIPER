# 中文简介：对多个 A2A checkpoint 做离线对比评估，观察不同训练步数的动作拟合效果差异。

python /home/szk/szk/Evo-RL/compare_a2a_checkpoints.py \
    --checkpoint-path /home/szk/szk/Evo-RL/outputs/a2a_train_20260426_121524/checkpoints/020000/pretrained_model \
    --checkpoint-path /home/szk/szk/Evo-RL/outputs/a2a_train_20260426_121524/checkpoints/030000/pretrained_model \
    --checkpoint-path /home/szk/szk/Evo-RL/outputs/a2a_train_20260426_121524/checkpoints/040000/pretrained_model \
    --checkpoint-path /home/szk/szk/Evo-RL/outputs/a2a_train_20260426_121524/checkpoints/050000/pretrained_model \
    --dataset-root /home/szk/szk/Evo-RL/datasets3_trimmed_10hz_320x240 \
    --dataset-repo-id datasets3_trimmed_10hz_320x240 \
    --device cuda \
    --batch-size 6 \
    --num-workers 2 \
    --max-batches 20 \
    --output-json /home/szk/szk/Evo-RL/outputs/checkpoint_compare.json
