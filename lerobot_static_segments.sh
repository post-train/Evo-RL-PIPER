# 中文简介：记录如何裁掉数据集中长时间静止片段，主要用于离线清洗数据集。

#   python /home/szk/szk/Evo-RL/trim_lerobot_static_segments.py \
#     --input-root /home/szk/szk/Evo-RL/datasets3 \
#     --output-root /home/szk/szk/Evo-RL/datasets3_trimmed \
#     --buffer-frames 5 \
#     --motion-threshold 0.2 \
#     --sustain-window 5 \
#     --active-count 3 \
#     --min-static-frames 15 \
#     --min-episode-length 32 \
#     --overwrite

#   如果你想把前后缓冲分开调，用这两个参数：

#   - --head-buffer-frames 3
#   - --tail-buffer-frames 10
  python /home/szk/szk/Evo-RL/trim_lerobot_static_segments.py \
    --input-root /home/szk/szk/Evo-RL/datasets3 \
    --output-root /home/szk/szk/Evo-RL/datasets3_trimmed \
    --buffer-frames 5 \
    --motion-threshold 0.2 \
    --sustain-window 5 \
    --active-count 3 \
    --min-static-frames 15 \
    --min-episode-length 32 \
    --overwrite
