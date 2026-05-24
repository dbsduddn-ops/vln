import os
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
import wandb

import sys
from pathlib import Path

# Add the parent directory to sys.path to find mymodule
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince, make_eval_table
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB
from r2r.data_utils import construct_instrs, load_obj2vps
from r2r.env import R2RNavBatch
from r2r.parser import parse_args
from r2r.agent import GMapNavAgent


def _fmt_duration(sec):
    sec = max(int(sec), 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _subsample_train_instr_data(data, ratio, seed):
    """Deterministically subsample train instructions by ratio (paper Table 3: 10%/50%/100%)."""
    n_full = len(data)
    if ratio >= 1.0:
        return data, None, n_full, n_full
    if ratio <= 0.0:
        raise ValueError(f"train_data_ratio must be in (0, 1], got {ratio}")
    keep_n = max(1, int(n_full * ratio))
    rng = np.random.RandomState(seed)
    keep_idxs = rng.choice(n_full, size=keep_n, replace=False)
    keep_idxs = np.sort(keep_idxs)
    subset = [data[i] for i in keep_idxs]
    return subset, keep_idxs.tolist(), n_full, keep_n


def _save_train_subsample_manifest(args, manifest):
    if manifest is None:
        return
    path = os.path.join(args.log_dir, 'train_subsample.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def build_dataset(args, rank=0, is_test=False):

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    if args.dataset == 'REVERIE':
        obj2vps = load_obj2vps(os.path.join(args.anno_dir, 'BBoxes.json'))

    dataset_class = R2RNavBatch

    # because we don't use distributed sampler here
    # in order to make different processes deal with different training examples
    # we need to shuffle the data with different seed in each processes
    if args.aug is not None:
        aug_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [args.aug], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        if not is_test and args.train_data_ratio < 1.0:
            aug_instr_data, _, aug_full, aug_keep = _subsample_train_instr_data(
                aug_instr_data, args.train_data_ratio, args.seed + 10007
            )
            if rank <= 0:
                print(
                    f"[INFO] Aug data subsampled: {aug_keep}/{aug_full} "
                    f"(train_data_ratio={args.train_data_ratio})"
                )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, args.candidate_file_dir,
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
        )
    else:
        aug_env = None

    if not is_test:
        train_instr_data = construct_instrs(
            args.anno_dir, args.dataset, ['train'], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        train_instr_data, keep_idxs, n_full, n_keep = _subsample_train_instr_data(
            train_instr_data, args.train_data_ratio, args.seed
        )
        if rank <= 0:
            if args.train_data_ratio < 1.0:
                print(
                    f"[INFO] R2R train subsampled: {n_keep}/{n_full} instructions "
                    f"({args.train_data_ratio * 100:.0f}%, seed={args.seed})"
                )
            manifest = None
            if keep_idxs is not None:
                manifest = {
                    'dataset': args.dataset,
                    'seed': args.seed,
                    'train_data_ratio': args.train_data_ratio,
                    'train_data_percent': int(round(args.train_data_ratio * 100)),
                    'n_full': n_full,
                    'n_keep': n_keep,
                    'keep_indices': keep_idxs,
                }
                _save_train_subsample_manifest(args, manifest)
        train_env = dataset_class(
            feat_db, train_instr_data, args.connectivity_dir, args.candidate_file_dir,
            batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None, name='train', 
        )
    else:
        train_env = None

    if args.dataset in ('REVERIE', 'rxr-en'):
        # REVERIE / RxR have no val_train_seen (RxR splits: val_seen, val_unseen, test_standard)
        val_env_names = ['val_seen', 'val_unseen']
    else:
        val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']

    if args.dataset == 'r4r' and (not args.test):
        val_env_names[-1] == 'val_unseen_sampled'
    
    if args.submit and args.dataset != 'r4r':
        val_env_names.append('test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split], 
            tokenizer=args.tokenizer, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, args.candidate_file_dir,
            batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
        )   # evaluation using all objects
        if args.dataset == 'REVERIE':
            val_env.obj2vps = obj2vps
        else:
            val_env.obj2vps = None
        val_envs[split] = val_env

    return train_env, val_envs, aug_env


def train(args, train_env, val_envs, aug_env=None, rank=-1):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                entity=args.wandb_entity,
                config=vars(args),
            )

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank)

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\n[INFO] Load the model from {}, {} iterations.".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "[INFO] Validation before training."
        val_results = []
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                # loss_str += ", %s " % env_name
                val_result = {"eval_split": env_name} # Create a new dictionary with env_name as the first key
                for metric, val in score_summary.items():
                    # loss_str += ', %s: %.2f' % (metric, val)
                    val_result[metric] = round(val, 2)
                val_results.append(val_result)
        loss_str += "\n" + make_eval_table(val_results)
        if default_gpu:
            write_to_record_file(loss_str, record_file)

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\n[INFO] Listener training starts, start iteration: %s' % str(start_iter), record_file
        )
        write_to_record_file(
            f"[INFO] Train env size: {train_env.size()} instructions "
            f"(train_data_ratio={args.train_data_ratio}).",
            record_file,
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
    if args.dataset == 'r4r':
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state":""}}

    if args.eval_every is not None:
        eval_every = max(1, int(args.eval_every))
    elif args.eval_every_epoch:
        iters_per_epoch = max(1, int(np.ceil(float(train_env.size()) / float(args.batch_size))))
        eval_every = max(1, int(iters_per_epoch * max(1, int(args.eval_every_n_epochs))))
    else:
        eval_every = max(1, int(args.log_every))

    if default_gpu:
        write_to_record_file(
            f"[INFO] Validation interval: every {eval_every} iterations.",
            record_file
        )
    
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        interval_start = time.time()
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            if default_gpu and idx == start_iter:
                print("[INFO] Entering first training interval...")
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            jdx_length = len(range(interval // 2))
            for jdx in range(interval // 2):
                # Train with GT data
                listner.env = train_env
                if default_gpu and idx == start_iter and jdx == 0:
                    print("[INFO] Entering first training interval (GT + AUG)...")
                listner.train(1, feedback=args.feedback)

                # Train with Augmented data
                listner.env = aug_env
                listner.train(1, feedback=args.feedback)

                if default_gpu:
                    print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            generation_loss = sum(listner.logs['generation_loss']) / max(len(listner.logs['generation_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("loss/generation_loss", generation_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            if args.use_wandb:
                wandb.log({
                    "train/critic_loss": critic_loss,
                    "train/policy_entropy": entropy,
                    "train/RL_loss": RL_loss,
                    "train/IL_loss": IL_loss,
                    "train/generation_loss": generation_loss,
                    "train/total_actions": total,
                    "train/max_length": length,
                }, step=idx)
            write_to_record_file(
                "\n[INFO] Training: total_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, gen_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, generation_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )
            train_elapsed = time.time() - start
            train_progress = max(float(iter) / float(args.iters), 1e-8)
            train_eta = train_elapsed / train_progress - train_elapsed
            interval_elapsed = time.time() - interval_start
            print(
                "[INFO] Train progress: iter %d/%d (%.2f%%), interval_time %s, elapsed %s, ETA %s" % (
                    iter,
                    args.iters,
                    train_progress * 100.0,
                    _fmt_duration(interval_elapsed),
                    _fmt_duration(train_elapsed),
                    _fmt_duration(train_eta),
                )
            )

        should_eval = (iter % eval_every == 0) or (iter >= start_iter + args.iters)
        if default_gpu:
            if should_eval:
                print("[INFO] Start validation at %d iterations." % iter)
            else:
                print("[INFO] Skip validation at %d (next every %d iterations)." % (iter, eval_every))
        if should_eval:
            # Run validation
            loss_str = "Iter {}".format(iter)
            val_results = []
            eval_start = time.time()
            total_eval_envs = len(val_envs)
            for env_name, env in val_envs.items():
                env_start = time.time()
                if default_gpu:
                    completed_envs = len(val_results)
                    print(
                        "[INFO] Eval progress: %d/%d, running split %s" % (
                            completed_envs + 1, total_eval_envs, env_name
                        )
                    )
                listner.env = env

                # Get validation distance from goal under test evaluation conditions
                listner.test(use_dropout=False, feedback='argmax', iters=None)
                preds = listner.get_results()
                preds = merge_dist_results(all_gather(preds))

                if default_gpu:
                    score_summary, _ = env.eval_metrics(preds)
                    val_result = {"eval_split": env_name} # Create a new dictionary with env_name as the first key
                    # loss_str += ",\n%s" % env_name
                    wandb_val_log = {}
                    for metric, val in score_summary.items():
                        # loss_str += ', %s: %.2f' % (metric, val)
                        val_result[metric] = round(val, 2)
                        writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
                        wandb_val_log['val/%s/%s' % (env_name, metric)] = score_summary[metric]
                    val_results.append(val_result)
                    if args.use_wandb:
                        wandb.log(wandb_val_log, step=idx)
                    env_elapsed = time.time() - env_start
                    done = len(val_results)
                    eval_elapsed = time.time() - eval_start
                    avg_env = eval_elapsed / max(done, 1)
                    eval_eta = avg_env * (total_eval_envs - done)
                    print(
                        "[INFO] Eval split done: %s (%d/%d), split_time %s, eval_elapsed %s, eval_ETA %s" % (
                            env_name,
                            done,
                            total_eval_envs,
                            _fmt_duration(env_elapsed),
                            _fmt_duration(eval_elapsed),
                            _fmt_duration(eval_eta),
                        )
                    )

                    # select model by spl
                    if env_name in best_val:
                        if score_summary['spl'] >= best_val[env_name]['spl']:
                            best_val[env_name]['spl'] = score_summary['spl']
                            best_val[env_name]['sr'] = score_summary['sr']
                            best_val[env_name]['state'] = 'Iter %d\n%s' % (iter, make_eval_table(val_results))
                            listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                    
            
            if default_gpu:
                listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))
                loss_table = make_eval_table(val_results)
                loss_str += "\n" + loss_table

                write_to_record_file(
                    ('[INFO] %s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                    record_file
                )
                write_to_record_file("BEST RESULT TILL NOW", record_file)
                for env_name in best_val:
                    write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)

    if default_gpu and args.use_wandb:
        wandb.finish()


def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
        if args.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                entity=args.wandb_entity,
                config=vars(args),
            )
    
    val_results = []
    eval_all_start = time.time()
    total_eval_envs = len(val_envs)
    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            continue
        env_start = time.time()
        if default_gpu:
            done = len(val_results)
            print(
                "[INFO] Eval progress: %d/%d, running split %s" % (
                    done + 1, total_eval_envs, env_name
                )
            )
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                val_result = {"eval_split": env_name} # Create a new dictionary with env_name as the first key
                # loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    # loss_str += ', %s: %.2f' % (metric, val)
                    val_result[metric] = round(val, 2)
                val_results.append(val_result)
                if args.use_wandb:
                    wandb.log(
                        {'test/%s/%s' % (env_name, metric): val
                         for metric, val in val_result.items() if metric != 'eval_split'},
                    )
                env_elapsed = time.time() - env_start
                done = len(val_results)
                eval_elapsed = time.time() - eval_all_start
                avg_env = eval_elapsed / max(done, 1)
                eval_eta = avg_env * (total_eval_envs - done)
                print(
                    "[INFO] Eval split done: %s (%d/%d), split_time %s, eval_elapsed %s, eval_ETA %s" % (
                        env_name,
                        done,
                        total_eval_envs,
                        _fmt_duration(env_elapsed),
                        _fmt_duration(eval_elapsed),
                        _fmt_duration(eval_eta),
                    )
                )


            json.dump(
                preds,
                open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                sort_keys=True, indent=4, separators=(',', ': ')
            )
    
    if default_gpu:
        loss_str = make_eval_table(val_results)
        write_to_record_file(loss_str, record_file)
        if args.use_wandb:
            wandb.finish()


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        # torchrun sets LOCAL_RANK as an env var; fall back to the CLI arg if missing
        local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        args.local_rank = local_rank
        torch.cuda.set_device(local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env = build_dataset(args, rank=rank, is_test=args.test)

    if not args.test:
        train(args, train_env, val_envs, aug_env=aug_env, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)
            

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # torchrun/elastic will still log worker shutdown; keep Python traceback clean.
        raise SystemExit(130)
