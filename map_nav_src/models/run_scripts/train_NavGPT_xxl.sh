# ─── GPU settings ──────────────────────────────────────────────────────────────
# Set CUDA_VISIBLE_DEVICES and NPROC to match your hardware.
# The world_size in r2r_navgpt_ft_xxl.yaml must equal NPROC.
CUDA_VISIBLE_DEVICES='0,1,2,3'
NPROC=4

pick_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

MASTER_PORT=$(pick_free_port)
echo "[train_NavGPT_xxl] master_port=${MASTER_PORT}"

# Run from map_nav_src/models (same convention as train_NavGPT_xl.sh)
TOKENIZERS_PARALLELISM="false" \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
train.py --cfg-path lavis/projects/blip2/train/r2r_navgpt_ft_xxl.yaml
