#!/usr/bin/env bash
# Zero-shot RxR-EN eval: R2R-finetuned NavGPT2-XL, no RxR training.
#
# Prerequisites:
#   1. Official RxR guide JSONL from https://github.com/google-research-datasets/RxR
#   2. python scripts/prepare_rxr_data.py --source_dir <rxr_guide_dir>
#   3. R2R connectivity + MP3D_eva_clip_g features (same as test_r2r_xl.sh)
#
# Metrics (RxR / DUET): nDTW, sDTW, OSR, SR, SPL, TL on val_seen & val_unseen.
# Paper Table 4 reports val_unseen; use --submit for test (from rxr_test_standard).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_NAV_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MAP_NAV_SRC_DIR}/.." && pwd)"
DATA_ROOT="${PROJECT_ROOT}/datasets"
cd "${MAP_NAV_SRC_DIR}" || exit 1

train_alg=dagger

features=eva-clip-g
ft_dim=1408

CUDA_VISIBLE_DEVICES='3'
ngpus=1

seed=0
batch_size=2

# T5 prompt length cap (R2R default 200). Q-Former text is capped at 512 tokens in NavGPT_model.
max_instr_len=200

# ─── R2R checkpoint (same naming as test_r2r_xl.sh) ─────────────────────────────
R2R_NAME=NavGPT2-XL-renew-history
R2R_NAME=${R2R_NAME}-seed.${seed}
R2R_NAME=${R2R_NAME}-bs${batch_size}
R2R_OUTDIR=${DATA_ROOT}/R2R/exprs_map/finetune/${R2R_NAME}
RESUME_FILE=${R2R_OUTDIR}/ckpts/best_val_unseen

# ─── RxR-EN eval output ───────────────────────────────────────────────────────
name=NavGPT2-XL-zeroshot-r2r-to-RXR-EN
name=${name}-from-${R2R_NAME}
outdir=${DATA_ROOT}/RXR-EN/exprs_map/zeroshot/${name}

# ─── Feature flags: keep identical to test_r2r_xl.sh ─────────────────────────
USE_IPE="--use_ipe"
IPE_NUM_HEADS=8
IPE_CONTEXT_MODE=both
IPE_HIST_POOL=dist_softmax
IPE_HIST_DIST_TAU=5.0
IPE_STAGE=post_encoder
USE_HISTORY_TOKEN="--use_history_token"

OBS_POOL_MODE=obj_adaptive
OBJ_PHRASE_SOURCE=rule
OBJ_SELECT_RHO=0.8
OBJ_REL_TEMP=0.2
OBJ_VIEW_TEMP=0.07
OBJ_MAX_PHRASES=12
OBJ_SCORE_SPACE=llm
OBJ_CLIP_MODEL_NAME=ViT-g-14
OBJ_CLIP_PRETRAINED=""
OBJ_CLIP_PRETRAINED_ARG=""
if [ -n "${OBJ_CLIP_PRETRAINED}" ]; then
  OBJ_CLIP_PRETRAINED_ARG="--obj_clip_pretrained ${OBJ_CLIP_PRETRAINED}"
fi
PRINT_OBJ_PHRASES=""
PRINT_OBJ_COSINE=""

USE_SUB_INSTR=""
USE_LEARNED_PROGRESS_STOP="--use_learned_progress_stop"
PROGRESS_AUX_WEIGHT=0.1
PROGRESS_TARGET=gt_path_ratio

RXR_ANNO=${DATA_ROOT}/RXR-EN/annotations/RXR-EN_val_unseen_enc.json

pick_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

if [ ! -f "${RESUME_FILE}" ]; then
  echo "[test_rxr_zeroshot] ERROR: R2R checkpoint not found: ${RESUME_FILE}" >&2
  exit 1
fi

if [ ! -f "${RXR_ANNO}" ]; then
  echo "[test_rxr_zeroshot] ERROR: RxR annotations missing: ${RXR_ANNO}" >&2
  echo "  Run: python scripts/prepare_rxr_data.py --source_dir <dir with rxr_*_guide.jsonl>" >&2
  echo "  Data: https://github.com/google-research-datasets/RxR" >&2
  exit 1
fi

flag="--root_dir ${DATA_ROOT}
      --dataset rxr-en
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion global

      --expert_policy spl
      --train_alg ${train_alg}

      --num_x_layers 4

      --max_action_steps 15
      --max_instr_len ${max_instr_len}

      --batch_size ${batch_size}
      --lr 1e-5
      --iters 200000
      --log_every 2500
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0.

      ${USE_IPE}
      --ipe_num_heads ${IPE_NUM_HEADS}
      --ipe_context_mode ${IPE_CONTEXT_MODE}
      --ipe_hist_pool ${IPE_HIST_POOL}
      --ipe_hist_dist_tau ${IPE_HIST_DIST_TAU}
      --ipe_stage ${IPE_STAGE}
      ${USE_HISTORY_TOKEN}
      --obs_pool_mode ${OBS_POOL_MODE}
      --obj_phrase_source ${OBJ_PHRASE_SOURCE}
      --obj_select_rho ${OBJ_SELECT_RHO}
      --obj_rel_temp ${OBJ_REL_TEMP}
      --obj_view_temp ${OBJ_VIEW_TEMP}
      --obj_score_space ${OBJ_SCORE_SPACE}
      --obj_clip_model_name ${OBJ_CLIP_MODEL_NAME}
      ${OBJ_CLIP_PRETRAINED_ARG}
      --obj_max_phrases ${OBJ_MAX_PHRASES}
      ${PRINT_OBJ_PHRASES}
      ${PRINT_OBJ_COSINE}
      ${USE_LEARNED_PROGRESS_STOP}
      --progress_aux_weight ${PROGRESS_AUX_WEIGHT}
      --progress_target ${PROGRESS_TARGET}
      ${USE_SUB_INSTR}"

WANDB_PROJECT='based VLM'
WANDB_ENTITY='Vision-Language-Navigation'
WANDB_AUTO_NAME="--wandb_auto_name"

echo "[test_rxr_zeroshot] R2R ckpt: ${RESUME_FILE}"
echo "[test_rxr_zeroshot] RxR out: ${outdir}"

TEST_MASTER_PORT=$(pick_free_port)
echo "[test_rxr_zeroshot] test master_port=${TEST_MASTER_PORT}"
TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=1 \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
  torchrun --nproc_per_node=${ngpus} --master_addr=127.0.0.1 --master_port=${TEST_MASTER_PORT} \
    r2r/main_nav.py $flag \
    --test --submit \
    --freeze_qformer \
    --qformer_ckpt_path models/lavis/output/NavGPT-InstructBLIP-FlanT5xl/20260408054/checkpoint_best.pth \
    --resume_file ${RESUME_FILE} \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    ${WANDB_AUTO_NAME}
