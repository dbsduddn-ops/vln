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
batch_size=2  # must match finetuning setup if possible
train_data_ratio=1.0

name=NavGPT2-XL-renew-history
name=${name}-seed.${seed}
name=${name}-bs${batch_size}

outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}

# ─── Feature flags ─────────────────────────────────────────────────────────────
# Idea A: Instruction Progress Estimator  (on/off)
USE_IPE="--use_ipe"           # set to "" to disable
IPE_NUM_HEADS=8
# IPE query: both | obs_only | hist_only
IPE_CONTEXT_MODE=both
# History pooling: mean | dist_softmax (graph shortest-path distance to current VP; closer = higher weight)
IPE_HIST_POOL=dist_softmax
IPE_HIST_DIST_TAU=5.0
IPE_STAGE=post_encoder
# Optional: fuse view + compressed T5 state into per-node history token
USE_HISTORY_TOKEN="--use_history_token"   # set to "--use_history_token" to enable

# Observation pooling: mean | obj_adaptive
OBS_POOL_MODE=obj_adaptive
# Object phrase extraction: rule | llm (llm-style heuristic)
OBJ_PHRASE_SOURCE=rule
# Adaptive selection threshold: keep views with score >= rho * max_score
OBJ_SELECT_RHO=0.8
OBJ_REL_TEMP=0.2
OBJ_VIEW_TEMP=0.07
OBJ_MAX_PHRASES=12
# View-object cosine score space: llm | clip
OBJ_SCORE_SPACE=llm
# CLIP text encoder settings for OBJ_SCORE_SPACE=clip.
# NOTE: Leave empty to avoid loading an external CLIP checkpoint.
# If needed, set a CLIP-specific pretrained tag/path (not NavGPT checkpoint).
OBJ_CLIP_MODEL_NAME=ViT-g-14
OBJ_CLIP_PRETRAINED=""
OBJ_CLIP_PRETRAINED_ARG=""
if [ -n "${OBJ_CLIP_PRETRAINED}" ]; then
  OBJ_CLIP_PRETRAINED_ARG="--obj_clip_pretrained ${OBJ_CLIP_PRETRAINED}"
fi
# Log 원본 instruction + rule 추출 구문 (default GPU, instruction당 1회). 실제 출력은 OBS_POOL_MODE=obj_adaptive 일 때만.
PRINT_OBJ_PHRASES=""   # set to "" to disable
# Log per-step cosine/attention + compact embedding diagnostics.
PRINT_OBJ_COSINE=""   # set to "" to disable

# Idea B: Sub-instruction step tracking   (on/off)
USE_SUB_INSTR=""  # set to "" to disable --use_sub_instr

# Idea C: Learned progress-aware stopping (on/off)
USE_LEARNED_PROGRESS_STOP="--use_learned_progress_stop"  # set to "--use_learned_progress_stop" to enable
PROGRESS_AUX_WEIGHT=0.1
PROGRESS_TARGET=gt_path_ratio

# If you want a different checkpoint, override this path.
RESUME_FILE=${outdir}/ckpts/best_val_unseen

pick_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
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
      --train_data_ratio ${train_data_ratio}
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

TEST_MASTER_PORT=$(pick_free_port)
echo "[test_r2r_xl] test master_port=${TEST_MASTER_PORT}"
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
