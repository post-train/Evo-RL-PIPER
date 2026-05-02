# 中文简介：把现有数据集降采样到 10Hz，并同时输出目标分辨率版本，便于后续训练或分析。

python /home/szk/szk/Evo-RL/downsample_lerobot_dataset.py \
    --input-root /home/szk/szk/Evo-RL/datasets3_trimmed \
    --output-root /home/szk/szk/Evo-RL/datasets3_trimmed_10hz_320x240 \
    --repo-id datasets3_trimmed_10hz_320x240 \
    --target-fps 10 \
    --target-width 320 \
    --target-height 240 \
    --overwrite
