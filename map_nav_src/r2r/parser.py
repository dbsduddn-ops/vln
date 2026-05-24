import argparse
import os


def build_wandb_run_name(args):
    """Derive a concise W&B run name from key experiment flags (for --wandb_auto_name)."""
    parts = [
        args.dataset,
        f"seed{args.seed}",
        f"bs{args.batch_size}",
        f"x{args.num_x_layers}",
    ]
    if args.use_ipe:
        parts.append("ipe")
        parts.append(args.ipe_stage)
        parts.append(args.ipe_context_mode)
        parts.append(args.ipe_hist_pool)
        if args.ipe_hist_pool == "dist_softmax":
            parts.append(f"tau{args.ipe_hist_dist_tau:g}")
        parts.append(f"ih{args.ipe_num_heads}")
    else:
        parts.append("noipe")
    parts.append("histtok" if args.use_history_token else "nohisttok")
    parts.append(args.obs_pool_mode)
    if args.obs_pool_mode == "obj_adaptive":
        parts.append(args.obj_phrase_source)
        parts.append(f"rho{args.obj_select_rho:g}")
    parts.append("subinstr" if args.use_sub_instr else "nosubinstr")
    ratio = float(getattr(args, "train_data_ratio", 1.0))
    if ratio < 1.0:
        parts.append(f"pct{int(round(ratio * 100))}")
    if args.test:
        parts.append("test")
    return "-".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r', 'rxr-en', 'REVERIE'])
    parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs_map/finetune/default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', choices=['global', 'local', 'dynamic'], default='global')
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=200000, help='training iterations')
    parser.add_argument('--accumulate_grad_step', type=int, default=1, help='accumulate gradient step')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=None,
                        help='validation interval in iterations; if unset and --eval_every_epoch is enabled, auto-computed from 1 epoch')
    parser.add_argument('--eval_every_epoch', action='store_true', default=False,
                        help='run validation every N epochs (estimated by train_env.size() / batch_size)')
    parser.add_argument('--eval_every_n_epochs', type=int, default=1,
                        help='number of epochs per validation when --eval_every_epoch is enabled')
    parser.add_argument('--eval_first', action='store_true', default=False)
    parser.add_argument('--step_update', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    parser.add_argument(
        '--train_data_ratio', type=float, default=1.0,
        help='Fraction of R2R train instructions to use (0, 1]. '
             '0.1 / 0.5 / 1.0 reproduce NavGPT-2 paper Table 3 (10%%, 50%%, 100%% data). '
             'Subsample is deterministic per --seed; indices saved to logs/train_subsample.json.',
    )
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='eva-clip-g')
    
    # VLM Model Config
    parser.add_argument('--arch', choices=['blip2_t5_instruct_nav', 'blip2_vicuna_instruct_nav'], default='blip2_t5_instruct_nav')
    parser.add_argument('--model_type', choices=['flant5xl', 'flant5xxl', 'vicuna7b', 'vicuna13b'], default='flant5xl')
    # parser.add_argument('--qformer_ckpt_path', type=str, default="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth")
    parser.add_argument('--qformer_ckpt_path', type=str, default=None)
    parser.add_argument('--qformer_pos_emb', action='store_true', default=False)
    parser.add_argument('--load_patch_feature', action='store_true', default=True)
    parser.add_argument('--output_thought', action='store_true', default=False)
    parser.add_argument('--freeze_qformer', action='store_true', default=True)
    # Generation config
    parser.add_argument('--use_nucleus_sampling', action='store_true', default=False)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--num_captions', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)

    # Global branch config
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--attention_probs_dropout_prob', type=int, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
    parser.add_argument('--layer_norm_eps', type=int, default=1e-12)
    parser.add_argument('--hidden_size', type=int, default=768, help='DuET hidden size')
    parser.add_argument('--llm_ctx_size', type=int, default=2048, help='LLM hidden size')
    parser.add_argument('--max_action_steps', type=int, default=15)
    
    parser.add_argument('--llm_token_merge', type=str, default='linear', choices=['linear', 'mean'], help='How to merge image token features in LLM')
    parser.add_argument('--enc_full_graph', action='store_true', default=True)
    parser.add_argument('--graph_sprels', action='store_true', default=True)
    parser.add_argument('--use_lang2visn_attn', action='store_true', default=False)
    parser.add_argument('--use_clip_img_emb', action='store_true', default=False, help='Concatenate clip image embedding for DuET image embedding.')
    parser.add_argument('--global_cross_attn', action='store_true', default=True, help='Cross attention between vision and language in DuET.')
    parser.add_argument('--single_action_head', action='store_true', default=False, help='Single action head for DuET.')

    # Idea A: Instruction Progress Estimator (IPE)
    parser.add_argument('--use_ipe', action='store_true', default=False,
                        help='Enable Instruction Progress Estimator: soft-weight instruction embeddings '
                             'with a context query from observation and/or pooled visit history (see --ipe_*).')
    parser.add_argument('--ipe_num_heads', type=int, default=8,
                        help='Number of attention heads in the IPE cross-attention module.')
    parser.add_argument(
        '--ipe_context_mode', type=str, default='both',
        choices=['both', 'obs_only', 'hist_only'],
        help='IPE query context: both = avg observation + history (sum); obs_only / hist_only use one term.',
    )
    parser.add_argument(
        '--ipe_hist_pool', type=str, default='mean',
        choices=['mean', 'dist_softmax'],
        help='How to pool visited-node embeddings for IPE history: uniform mean, or softmax weights '
             'from negative graph shortest-path distance to the current viewpoint (closer = higher weight).',
    )
    parser.add_argument(
        '--ipe_hist_dist_tau', type=float, default=5.0,
        help='Temperature for --ipe_hist_pool dist_softmax (in the same distance units as the graph, meters).',
    )
    parser.add_argument(
        '--ipe_stage', type=str, default='post_encoder',
        choices=['pre_encoder', 'post_encoder', 'both'],
        help='Where IPE runs: post_encoder = after T5 encoder (legacy); pre_encoder = on instruction token '
             'embeddings before T5 encoder (requires panorama context first); both = apply twice.',
    )
    parser.add_argument(
        '--use_history_token', action='store_true', default=False,
        help='Store per-node fused history tokens (view embedding + compressed T5 instruction state) and '
             'use them in graph/history embeddings. Keeps existing distance-based history weighting unchanged.',
    )
    parser.add_argument(
        '--obs_pool_mode', type=str, default='mean',
        choices=['mean', 'obj_adaptive'],
        help='Observation pooling for current panorama embedding: plain mean or object-aware adaptive selection pooling.',
    )
    parser.add_argument(
        '--obj_phrase_source', type=str, default='rule',
        choices=['rule', 'llm'],
        help='Object phrase extraction mode for obj_adaptive pooling: rule-based noun phrase extractor or llm-style heuristic extractor.',
    )
    parser.add_argument(
        '--obj_select_rho', type=float, default=0.8,
        help='Adaptive view selection threshold ratio rho. Keep view i if score_i >= rho * max_score.',
    )
    parser.add_argument(
        '--obj_rel_temp', type=float, default=0.2,
        help='Softmax temperature when weighting extracted object phrases by current instruction context.',
    )
    parser.add_argument(
        '--obj_view_temp', type=float, default=0.07,
        help='Softmax temperature for cosine-based view attention pooling in obj_adaptive mode.',
    )
    parser.add_argument(
        '--obj_score_space', type=str, default='llm',
        choices=['llm', 'clip'],
        help='Space used for view-object cosine scoring in obj_adaptive mode: llm (legacy) or clip (CLIP text/view space).',
    )
    parser.add_argument(
        '--obj_clip_model_name', type=str, default='ViT-g-14',
        help='CLIP model name used when --obj_score_space clip.',
    )
    parser.add_argument(
        '--obj_clip_pretrained', type=str, default='',
        help='CLIP pretrained tag/path used when --obj_score_space clip. Empty means no pretrained loading.',
    )
    parser.add_argument(
        '--obj_max_phrases', type=int, default=12,
        help='Maximum number of object phrases kept per instruction for object-aware adaptive pooling.',
    )
    parser.add_argument(
        '--print_obj_phrases', action='store_true', default=False,
        help='When --obs_pool_mode obj_adaptive and --obj_phrase_source rule, log each unique instruction '
             'and rule-extracted object phrases once (default GPU only; uses phrase cache).',
    )
    parser.add_argument(
        '--print_obj_cosine', action='store_true', default=False,
        help='Log per-step object-view cosine scores, attention weights, and compact embedding diagnostics.',
    )

    # Idea B: Sub-instruction tracking
    parser.add_argument('--use_sub_instr', action='store_true', default=False,
                        help='Enable sub-instruction tracking: splits instruction into sentences and '
                             'highlights the estimated current action step in the LLM prompt.')
    parser.add_argument(
        '--use_learned_progress_stop', action='store_true', default=False,
        help='Enable learned progress-aware stopping: predicts latent progress from policy state and '
             'uses it to calibrate stop logit with a trainable stop calibrator (no hand-tuned rule).',
    )
    parser.add_argument(
        '--progress_aux_weight', type=float, default=0.1,
        help='Weight of auxiliary progress regression loss when --use_learned_progress_stop is enabled.',
    )
    parser.add_argument(
        '--progress_target', type=str, default='gt_path_ratio',
        choices=['gt_path_ratio', 'max_step_ratio'],
        help='Target for progress auxiliary supervision: ratio over GT path length or max_action_steps.',
    )

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=1408)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0.0, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='dagger'
    )

    # Weights & Biases
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='NavGPT2')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument(
        '--wandb_auto_name', action='store_true', default=False,
        help='If set with --use_wandb, set wandb run name from dataset/seed/batch/IPE/sub-instr flags.',
    )

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'eva-clip-g': 'MP3D_eva_clip_g_can.lmdb',
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, args.dataset.upper(), 'annotations')
    args.candidate_file_dir = os.path.join(ROOTDIR, 'R2R', 'annotations', 'scanvp_candidates.json')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    if args.use_wandb and args.wandb_auto_name:
        args.wandb_name = build_wandb_run_name(args)

    return args

