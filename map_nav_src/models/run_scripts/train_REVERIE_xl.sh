# ─── GPU settings ──────────────────────────────────────────────────────────────
# Set CUDA_VISIBLE_DEVICES and NPROC to match your hardware.
# The world_size in reverie_navgpt_ft_xl.yaml must equal NPROC.
CUDA_VISIBLE_DEVICES='3'
NPROC=1

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
echo "[train_REVERIE_xl] master_port=${MASTER_PORT}"

# Expected annotation files:
#   datasets/REVERIE/NavGPT-Instruct/REVERIE_NavGPT_train_v1.json
#   datasets/REVERIE/NavGPT-Instruct/REVERIE_NavGPT_val_v1.json
TOKENIZERS_PARALLELISM="false" \
NCCL_P2P_DISABLE=1 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} \
train.py --cfg-path lavis/projects/blip2/train/reverie_navgpt_ft_xl.yaml
