# 中文简介：使用 rerun 可视化指定 episode 的数据集内容，快速检查相机画面与标注质量。

rerun reset
lerobot-dataset-viz \
    --repo-id local/fold_a_towel \
    --root /home/szk/szk/Evo-RL/datasets3 \
    --episode-index 91\
    --display-compressed-images false
