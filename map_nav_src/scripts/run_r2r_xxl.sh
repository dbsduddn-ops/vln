DATA_ROOT=../datasets

train_alg=dagger

features=eva-clip-g
ft_dim=1408

CUDA_VISIBLE_DEVICES='3'
ngpus=1

seed=0
batch_size=1  # per-GPU batch size; XXL may need 1 if OOM

name=NavGPT2-XXL
name=${name}-seed.${seed}
name=${name}-bs${batch_size}

outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}

# BLIP2 / Q-Former checkpoint (update to your trained XXL checkpoint)
QFORMER_CKPT="models/lavis/output/NavGPT-InstructBLIP-FlanT5xxl/20260419184/checkpoint_best.pth"

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

# Observation pooling: mean | obj_adaptive
OBS_POOL_MODE=obj_adaptive
# Object phrase extraction: rule | llm (llm-style heuristic)
OBJ_PHRASE_SOURCE=rule
# Adaptive selection threshold: keep views with score >= rho * max_score
OBJ_SELECT_RHO=0.8
OBJ_REL_TEMP=0.2
OBJ_MAX_PHRASES=12

# Idea B: Sub-instruction step tracking   (on/off)
USE_SUB_INSTR=""  # set to "" to disable --use_sub_instr

pick_free_port() {
  python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

# Debug sync mode (very slow). Keep 0 for normal training.
CUDA_SYNC_DEBUG=0

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
      --ipe_stage ${IPE_STAGE}
      --ipe_num_heads ${IPE_NUM_HEADS}
      --ipe_context_mode ${IPE_CONTEXT_MODE}
      --ipe_hist_pool ${IPE_HIST_POOL}
      --ipe_hist_dist_tau ${IPE_HIST_DIST_TAU}
      --obs_pool_mode ${OBS_POOL_MODE}
      --obj_phrase_source ${OBJ_PHRASE_SOURCE}
      --obj_select_rho ${OBJ_SELECT_RHO}
      --obj_rel_temp ${OBJ_REL_TEMP}
      --obj_max_phrases ${OBJ_MAX_PHRASES}
      ${USE_SUB_INSTR}"

# ─── W&B settings ──────────────────────────────────────────────────────────────
WANDB_PROJECT='based VLM'
WANDB_ENTITY='Vision-Language-Navigation'
WANDB_AUTO_NAME="--wandb_auto_name"

# ─── Train ─────────────────────────────────────────────────────────────────────
TRAIN_MASTER_PORT=$(pick_free_port)
echo "[run_r2r_xxl] train master_port=${TRAIN_MASTER_PORT}"
TRANSFORMERS_OFFLINE=1 \
  HF_HUB_OFFLINE=1 \
  NCCL_IB_DISABLE=1 \
  NCCL_P2P_DISABLE=1 \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
  CUDA_LAUNCH_BLOCKING=${CUDA_SYNC_DEBUG} \
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
  torchrun --nproc_per_node=${ngpus} --master_addr=127.0.0.1 --master_port=${TRAIN_MASTER_PORT} \
    r2r/main_nav.py $flag \
    --freeze_qformer \
    --step_update \
    --model_type flant5xxl \
    --llm_ctx_size 4096 \
    --aug ../datasets/R2R/annotations/prevalent_aug.json \
    --qformer_ckpt_path ${QFORMER_CKPT} \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    ${WANDB_AUTO_NAME}

# ─── Test ──────────────────────────────────────────────────────────────────────
TEST_MASTER_PORT=$(pick_free_port)
echo "[run_r2r_xxl] test master_port=${TEST_MASTER_PORT}"
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
    --model_type flant5xxl \
    --llm_ctx_size 4096 \
    --qformer_ckpt_path ${QFORMER_CKPT} \
    --resume_file ${outdir}/ckpts/best_val_unseen \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    ${WANDB_AUTO_NAME}
