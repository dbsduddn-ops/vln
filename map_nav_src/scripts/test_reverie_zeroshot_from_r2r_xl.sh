#!/usr/bin/env bash
# Zero-shot REVERIE eval: load R2R-finetuned NavGPT2-XL (same arch flags as test_r2r_xl.sh).
# No REVERIE training — only --test --submit on REVERIE splits.
#
# Metrics (REVERIE challenge / DUET reverie/env.py):
#   - Computed on val_seen & val_unseen: Nav-Succ (sr), Nav-OSucc (oracle_sr), Nav-SPL (spl), Nav-Length (lengths)
#   - RGS / RGSPL (primary) need pred_objid in submit JSON — NavGPT map agent is navigation-only, so RGS is N/A
#   - test split: trajectory JSON only (no public GT); email preds to reverie.challenge@gmail.com for RGSPL
# Ref: DUET https://github.com/cshizhe/VLN-DUET (reverie/main_nav_obj.py), REVERIE challenge metrics page

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

# ─── R2R checkpoint source (must match training / test_r2r_xl.sh name) ─────────
R2R_NAME=NavGPT2-XL-renew-history
R2R_NAME=${R2R_NAME}-seed.${seed}
R2R_NAME=${R2R_NAME}-bs${batch_size}
R2R_OUTDIR=${DATA_ROOT}/R2R/exprs_map/finetune/${R2R_NAME}
RESUME_FILE=${R2R_OUTDIR}/ckpts/best_val_unseen

# ─── REVERIE eval output (separate from REVERIE fine-tune runs) ────────────────
name=NavGPT2-XL-zeroshot-r2r-to-REVERIE
name=${name}-from-${R2R_NAME}
outdir=${DATA_ROOT}/REVERIE/exprs_map/zeroshot/${name}

# ─── Feature flags: keep identical to test_r2r_xl.sh / R2R training ────────────
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
  echo "[test_reverie_zeroshot] ERROR: R2R checkpoint not found: ${RESUME_FILE}" >&2
  exit 1
fi

flag="--root_dir ${DATA_ROOT}
      --dataset REVERIE
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
      --max_instr_len 200

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

echo "[test_reverie_zeroshot] R2R ckpt: ${RESUME_FILE}"
echo "[test_reverie_zeroshot] REVERIE out: ${outdir}"

TEST_MASTER_PORT=$(pick_free_port)
echo "[test_reverie_zeroshot] test master_port=${TEST_MASTER_PORT}"
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
