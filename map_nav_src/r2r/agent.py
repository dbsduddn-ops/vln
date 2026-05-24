import json
import os
import re
import sys
import importlib
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
try:
    import nltk
except Exception:
    nltk = None
try:
    import spacy
except Exception:
    spacy = None

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.NavGPT_model import NavGPT, Critic
from models.ops import pad_tensors_wgrad

from .prompt_template import NavGPT_PROMPT, NavGPT_SUB_INSTR_PROMPT

from transformers import PretrainedConfig


def _get_current_sub_instruction(instruction: str, step: int, max_steps: int) -> str:
    """Idea B helper: splits instruction into sub-sentences and returns the one
    estimated to be relevant at the current navigation step.

    Splitting uses sentence-ending punctuation as delimiters so that phrases like
    "Turn left. Walk past the table. Stop." become three sub-instructions.
    A linear progress estimate maps step → sub-sentence index.
    """
    subs = [s.strip() for s in re.split(r'(?<=[.!?])\s+', instruction.strip()) if s.strip()]
    if len(subs) <= 1:
        return instruction
    n = len(subs)
    idx = min(int(step * n / max(max_steps, 1)), n - 1)
    return subs[idx]


class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self, config):
        ''' Build model, either from scratch or from saved model checkpoint. '''
        self.NavGPT = NavGPT(config).to(self.device)
        if config.bert_ckpt_file is not None:
            ckpt_weights = torch.load(config.bert_ckpt_file)['NavGPT']['state_dict']
            self.NavGPT.load_state_dict(ckpt_weights, strict=False)
        
        if config.freeze_qformer:
            print("[INFO] Freezing the Q-Former.")
            for name, param in self.NavGPT.llm.Blip2InstructNav.named_parameters():
                param.requires_grad = False
        
        self.critic = Critic(self.args).to(self.device)

        # Idea B: Sub-instruction tracking — selects a focused prompt template
        self.use_sub_instr = getattr(config, 'use_sub_instr', False)
        self.prompt = NavGPT_SUB_INSTR_PROMPT if self.use_sub_instr else NavGPT_PROMPT

        # Idea A: IPE flag (module lives inside self.NavGPT)
        self.use_ipe = getattr(config, 'use_ipe', False)
        self.ipe_context_mode = getattr(config, 'ipe_context_mode', 'both')
        self.ipe_hist_pool = getattr(config, 'ipe_hist_pool', 'mean')
        self.ipe_hist_dist_tau = float(getattr(config, 'ipe_hist_dist_tau', 5.0))
        self.ipe_stage = getattr(config, 'ipe_stage', 'post_encoder')
        self.obs_pool_mode = getattr(config, 'obs_pool_mode', 'mean')
        self.obj_phrase_source = getattr(config, 'obj_phrase_source', 'rule')
        self.obj_select_rho = float(getattr(config, 'obj_select_rho', 0.8))
        self.obj_rel_temp = float(getattr(config, 'obj_rel_temp', 0.2))
        self.obj_view_temp = float(getattr(config, 'obj_view_temp', 0.07))
        self.obj_score_space = getattr(config, 'obj_score_space', 'llm')
        self.obj_clip_model_name = getattr(config, 'obj_clip_model_name', 'ViT-g-14')
        self.obj_clip_pretrained = getattr(config, 'obj_clip_pretrained', '')
        if isinstance(self.obj_clip_pretrained, str) and self.obj_clip_pretrained.lower() in {'', 'none', 'null'}:
            self.obj_clip_pretrained = ''
        self.obj_max_phrases = int(getattr(config, 'obj_max_phrases', 12))
        self.use_learned_progress_stop = getattr(config, 'use_learned_progress_stop', False)
        self.progress_aux_weight = float(getattr(config, 'progress_aux_weight', 0.1))
        self.progress_target = getattr(config, 'progress_target', 'gt_path_ratio')
        self.use_history_token = getattr(config, 'use_history_token', False)
        self._object_phrase_cache = {}
        self._object_embed_cache = {}
        self._object_clip_embed_cache = {}
        self._obj_clip_model = None
        self._obj_clip_tokenize = None
        self._obj_clip_warned = False
        self._printed_obj_token_map = set()

        # buffer
        self.scanvp_cands = {}

    def _construct_candidate_dict(self, rel_angles, rel_dists):
        ''' Construct candidate dict. '''
        image_p = '[IMG]<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>[/IMG]'

        candidate_dict = {}
        for i in range(len(rel_angles)):
            heading = np.rad2deg(rel_angles[i, 0])
            if -45 <= heading <= 45:
                direction = "front"
            elif 45 < heading <= 135:
                direction = "right"
            elif -135 <= heading < -45:
                direction = "left"
            else:
                direction = "rear"
            key = f"Candidate {i}, facing {heading:.2f} degrees, {direction}"
            # key = f"Candidate {i}, facing {np.rad2deg(rel_angles[i, 0]):.2f} degrees, {rel_dists[i]:.2f} meters"
            candidate_dict[key] = image_p
        return candidate_dict

    def _get_navgpt_core(self):
        return self.NavGPT.module if hasattr(self.NavGPT, "module") else self.NavGPT

    def _get_node_history_embed(self, gmap, vp):
        if self.use_history_token:
            return gmap.get_node_hist_embed(vp)
        return gmap.get_node_embed(vp)

    def _dedup_phrases(self, phrases):
        seen = set()
        out = []
        for p in phrases:
            q = re.sub(r'\s+', ' ', p.lower().strip())
            if not q:
                continue
            if q in seen:
                continue
            seen.add(q)
            out.append(q)
        return out[:self.obj_max_phrases]

    def _normalize_instruction_for_obj_phrase(self, instruction: str) -> str:
        # Recover common WordPiece artifacts in dataset text (e.g., "##osta ##t" -> "ostat").
        toks = instruction.split()
        merged = []
        for tok in toks:
            if tok.startswith("##"):
                piece = tok[2:]
                if piece:
                    if merged:
                        merged[-1] = merged[-1] + piece
                    else:
                        merged.append(piece)
            else:
                merged.append(tok)
        text = " ".join(merged)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _get_obj_phrase_tokenizer(self):
        core = self._get_navgpt_core()
        blip = core.llm.Blip2InstructNav
        if "t5" in self.args.arch:
            return blip.t5_tokenizer
        return blip.llm_tokenizer

    def _phrase_token_match_mask(self, instruction_ids, phrase_list):
        """
        Build a (P, L) bool mask indicating which instruction token positions match each phrase.
        """
        if instruction_ids is None or phrase_list is None or len(phrase_list) == 0:
            return None
        tokenizer = self._get_obj_phrase_tokenizer()
        valid_ids = [int(x) for x in instruction_ids.tolist()]
        L = len(valid_ids)
        if L == 0:
            return None
        phrase_masks = torch.zeros((len(phrase_list), L), dtype=torch.bool, device=self.device)
        for p_idx, phrase in enumerate(phrase_list):
            toks = tokenizer(
                phrase,
                return_tensors="pt",
                truncation=True,
                max_length=max(2, self.args.max_instr_len // 4),
                add_special_tokens=False,
            )
            p_ids = toks.input_ids[0].tolist()
            if len(p_ids) == 0:
                continue
            # Exact sub-sequence search over instruction token ids.
            for s in range(0, max(L - len(p_ids) + 1, 0)):
                if valid_ids[s : s + len(p_ids)] == p_ids:
                    phrase_masks[p_idx, s : s + len(p_ids)] = True
        return phrase_masks

    def _debug_print_phrase_token_map(self, instruction, instruction_ids, phrase_list, phrase_masks):
        if not getattr(self.args, 'print_obj_phrases', False) or not self.default_gpu:
            return
        if instruction_ids is None or phrase_masks is None:
            return
        norm_instruction = self._normalize_instruction_for_obj_phrase(instruction)
        if norm_instruction in self._printed_obj_token_map:
            return
        self._printed_obj_token_map.add(norm_instruction)
        tokenizer = self._get_obj_phrase_tokenizer()
        ids_list = [int(x) for x in instruction_ids.tolist()]
        instr_toks = tokenizer.convert_ids_to_tokens(ids_list)
        rows = []
        for p_idx, phrase in enumerate(phrase_list):
            pos = torch.where(phrase_masks[p_idx])[0].tolist()
            matched_ids = [ids_list[j] for j in pos]
            matched_toks = [instr_toks[j] for j in pos]
            rows.append(
                {
                    "phrase": phrase,
                    "matched_pos": pos,
                    "matched_ids": matched_ids,
                    "matched_tokens": matched_toks,
                }
            )
        print(
            "[print_obj_token_map]\n"
            f"  instruction: {instruction}\n"
            f"  instruction_ids: {ids_list}\n"
            f"  instruction_tokens: {instr_toks}\n"
            f"  phrase_token_map: {rows}\n",
            flush=True,
        )

    def _get_spacy_nlp_for_obj_phrase(self):
        if hasattr(self, "_spacy_nlp_for_obj_phrase"):
            return self._spacy_nlp_for_obj_phrase
        self._spacy_nlp_for_obj_phrase = None
        if spacy is None:
            return None
        for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
            try:
                self._spacy_nlp_for_obj_phrase = spacy.load(model_name, disable=["ner"])
                break
            except Exception:
                continue
        return self._spacy_nlp_for_obj_phrase

    def _cleanup_object_phrase(self, phrase: str, stopwords: set, boundary_words: set):
        words = [w for w in phrase.split() if w]
        if not words:
            return ""
        dedup_words = []
        for w in words:
            if not dedup_words or dedup_words[-1] != w:
                dedup_words.append(w)
        words = dedup_words
        cut_idx = len(words)
        for idx, w in enumerate(words):
            if idx > 0 and w in boundary_words:
                cut_idx = idx
                break
        words = words[:cut_idx]
        while words and words[0] in {"the", "a", "an"}:
            words.pop(0)
        while words and words[-1] in {
            "the", "a", "an", "of", "on", "in", "at", "to", "from", "with", "near", "by",
            "last", "first", "second", "third", "one", "two", "three", "set",
        }:
            words.pop()
        if not words:
            return ""
        if all(w in stopwords for w in words):
            return ""
        return " ".join(words)

    def _extract_object_phrases_rule(self, instruction: str):
        instruction = self._normalize_instruction_for_obj_phrase(instruction)
        boundary_words = {
            "and", "then", "once", "using", "with", "without", "at", "to", "from", "on",
            "in", "into", "toward", "towards", "through", "past", "passed", "up", "down",
            "left", "right", "straight", "stop", "go", "walk", "turn", "reach", "wait",
        }
        stopwords = {
            "left", "right", "front", "rear", "forward", "back", "straight", "first", "second",
            "third", "floor", "room", "area", "side", "end", "middle", "top", "bottom", "hallway",
            "corridor", "there", "here", "it", "one", "two", "three", "doorway",
        }

        # Prefer spaCy noun_chunks first.
        nlp = self._get_spacy_nlp_for_obj_phrase()
        if nlp is not None:
            try:
                doc = nlp(instruction)
                phrases = []
                for chunk in doc.noun_chunks:
                    cand = re.sub(r"[^a-z0-9\s\-]", " ", chunk.text.lower())
                    cand = re.sub(r"\s+", " ", cand).strip()
                    cleaned = self._cleanup_object_phrase(cand, stopwords, boundary_words)
                    if cleaned:
                        phrases.append(cleaned)
                phrases = self._dedup_phrases(phrases)
                if len(phrases) > 0:
                    return phrases
            except Exception:
                if self.default_gpu and not getattr(self, "_spacy_obj_phrase_warned", False):
                    print("[WARN] spaCy noun-chunk extraction failed; fallback to NLTK/regex extractor.", flush=True)
                    self._spacy_obj_phrase_warned = True
        elif self.default_gpu and not getattr(self, "_spacy_obj_phrase_warned", False):
            print("[WARN] spaCy model not found (en_core_web_sm/md/lg); fallback to NLTK/regex extractor.", flush=True)
            self._spacy_obj_phrase_warned = True

        # Fallback 1: NLTK POS-based noun-phrase extraction.
        if nltk is not None:
            try:
                if not hasattr(self, "_nltk_ready_for_obj_phrase"):
                    # Newer NLTK split resources by language-specific names.
                    try:
                        nltk.data.find("tokenizers/punkt_tab")
                    except LookupError:
                        nltk.data.find("tokenizers/punkt")
                    try:
                        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
                    except LookupError:
                        nltk.data.find("taggers/averaged_perceptron_tagger")
                    self._nltk_ready_for_obj_phrase = True

                tokens = [t.lower() for t in nltk.word_tokenize(instruction)]
                tokens = [t for t in tokens if re.match(r"^[a-z][a-z\-]*$", t)]
                if tokens:
                    tagged = nltk.pos_tag(tokens)
                    noun_tags = {"NN", "NNS", "NNP", "NNPS"}
                    allow_tags = noun_tags | {"JJ"}

                    phrases = []
                    cur = []
                    cur_has_noun = False
                    for tok, tag in tagged:
                        if tok in boundary_words:
                            if cur and cur_has_noun and not all(w in stopwords for w in cur):
                                cleaned = self._cleanup_object_phrase(" ".join(cur), stopwords, boundary_words)
                                if cleaned:
                                    phrases.append(cleaned)
                            cur, cur_has_noun = [], False
                            continue
                        if tag in allow_tags:
                            cur.append(tok)
                            if tag in noun_tags:
                                cur_has_noun = True
                        else:
                            if cur and cur_has_noun and not all(w in stopwords for w in cur):
                                cleaned = self._cleanup_object_phrase(" ".join(cur), stopwords, boundary_words)
                                if cleaned:
                                    phrases.append(cleaned)
                            cur, cur_has_noun = [], False
                    if cur and cur_has_noun and not all(w in stopwords for w in cur):
                        cleaned = self._cleanup_object_phrase(" ".join(cur), stopwords, boundary_words)
                        if cleaned:
                            phrases.append(cleaned)

                    phrases = self._dedup_phrases(phrases)
                    if len(phrases) > 0:
                        return phrases
            except Exception:
                if self.default_gpu and not getattr(self, "_nltk_obj_phrase_warned", False):
                    print("[WARN] NLTK noun-phrase extraction unavailable; fallback to regex rule extractor.", flush=True)
                    self._nltk_obj_phrase_warned = True

        # Fallback regex extractor (kept for environments without NLTK resources).
        text = instruction.lower()
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        phrases = []
        det_pat = r'\b(?:a|an|the)\s+([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,5})'
        prep_pat = r'\b(?:near|beside|next to|in front of|behind|toward|towards|into|through|past)\s+([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,5})'
        phrases.extend(re.findall(det_pat, text))
        phrases.extend(re.findall(prep_pat, text))
        filtered = []
        for p in phrases:
            cleaned = self._cleanup_object_phrase(p, stopwords, boundary_words)
            if cleaned:
                filtered.append(cleaned)
        return self._dedup_phrases(filtered)

    def _extract_object_phrases_llm_style(self, instruction: str):
        instruction = self._normalize_instruction_for_obj_phrase(instruction)
        text = instruction.lower()
        phrases = []
        phrases.extend(re.findall(r'"([^"]+)"', text))
        phrases.extend(re.findall(r"'([^']+)'", text))
        broad_pat = r'\b(?:the|a|an)\s+([a-z][a-z\-]*(?:\s+[a-z][a-z\-]*){0,5})'
        phrases.extend(re.findall(broad_pat, text))
        # Keep longer chunks first as a proxy for richer "LLM-style" phrases.
        phrases = sorted(self._dedup_phrases(phrases), key=lambda x: (-len(x.split()), x))
        if not phrases:
            return self._extract_object_phrases_rule(instruction)
        return phrases[:self.obj_max_phrases]

    def _extract_object_phrases(self, instruction: str):
        normalized_instruction = self._normalize_instruction_for_obj_phrase(instruction)
        key = (self.obj_phrase_source, normalized_instruction)
        if key in self._object_phrase_cache:
            return self._object_phrase_cache[key]
        if self.obj_phrase_source == 'llm':
            phrases = self._extract_object_phrases_llm_style(normalized_instruction)
        else:
            phrases = self._extract_object_phrases_rule(normalized_instruction)
        if (
            getattr(self.args, 'print_obj_phrases', False)
            and self.default_gpu
            and self.obj_phrase_source == 'rule'
        ):
            print(
                "[print_obj_phrases] instruction:\n"
                f"  {instruction}\n"
                f"  normalized_instruction: {normalized_instruction}\n"
                f"  extracted_phrases ({len(phrases)}): {phrases}\n",
                flush=True,
            )
        self._object_phrase_cache[key] = phrases
        return phrases

    def _encode_object_phrases(self, instruction: str):
        key = (self.obj_phrase_source, instruction)
        if key in self._object_embed_cache:
            phrase_llm, phrase_hidden = self._object_embed_cache[key]
            return phrase_llm.to(self.device), phrase_hidden.to(self.device)

        phrases = self._extract_object_phrases(instruction)
        if len(phrases) == 0:
            return None, None

        core = self._get_navgpt_core()
        blip = core.llm.Blip2InstructNav
        if "t5" in self.args.arch:
            tokenizer = blip.t5_tokenizer
            token_embed = blip.t5_model.encoder.embed_tokens
        else:
            tokenizer = blip.llm_tokenizer
            token_embed = blip.llm_model.get_input_embeddings()

        with torch.no_grad():
            toks = tokenizer(
                phrases, return_tensors='pt', padding=True, truncation=True,
                max_length=max(8, self.args.max_instr_len // 4), add_special_tokens=False
            )
            ids = toks.input_ids.to(self.device)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            mask = ids.ne(pad_id)
            tok_embeds = token_embed(ids)
            denom = mask.sum(1, keepdim=True).clamp(min=1)
            phrase_llm = (tok_embeds * mask.unsqueeze(2)).sum(1) / denom
            phrase_hidden = core.img_embeddings.vp_proj(phrase_llm)

        self._object_embed_cache[key] = (phrase_llm.detach().cpu(), phrase_hidden.detach().cpu())
        return phrase_llm, phrase_hidden

    def _get_obj_clip_model(self):
        if self._obj_clip_model is not None:
            return self._obj_clip_model
        # Always use lavis canonical module path to avoid duplicate registry
        # registration from importing the same file under different names.
        clip_model_mod = importlib.import_module("lavis.models.clip_models.model")
        clip_tokenizer_mod = importlib.import_module("lavis.models.clip_models.tokenizer")
        create_clip_model = getattr(clip_model_mod, "create_model")
        self._obj_clip_tokenize = getattr(clip_tokenizer_mod, "tokenize")
        try:
            clip_model = create_clip_model(
                self.obj_clip_model_name,
                pretrained=self.obj_clip_pretrained,
                precision="fp32",
                device=self.device,
            )
        except Exception as exc:
            if self.default_gpu:
                print(
                    f"[WARN] Failed to load CLIP pretrained '{self.obj_clip_pretrained}' "
                    f"for {self.obj_clip_model_name}: {exc}. Falling back to no-pretrained CLIP init.",
                    flush=True,
                )
            clip_model = create_clip_model(
                self.obj_clip_model_name,
                pretrained="",
                precision="fp32",
                device=self.device,
            )
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self._obj_clip_model = clip_model
        return self._obj_clip_model

    def _encode_object_phrases_clip(self, instruction: str):
        key = (self.obj_phrase_source, instruction, self.obj_clip_model_name, self.obj_clip_pretrained)
        if key in self._object_clip_embed_cache:
            return self._object_clip_embed_cache[key].to(self.device)

        phrases = self._extract_object_phrases(instruction)
        if len(phrases) == 0:
            return None

        clip_model = self._get_obj_clip_model()
        tokenize_fn = self._obj_clip_tokenize
        if tokenize_fn is None:
            return None
        with torch.no_grad():
            toks = tokenize_fn(phrases).to(self.device)
            phrase_clip = clip_model.encode_text(toks).float()
            phrase_clip = F.normalize(phrase_clip, dim=-1)

        self._object_clip_embed_cache[key] = phrase_clip.detach().cpu()
        return phrase_clip

    def _project_view_cls_for_clip(self, view_cls):
        """
        Map precomputed per-view EVA-CLIP class features to CLIP embedding space
        using the loaded CLIP vision head (ln_post/proj) when dimensions allow.
        """
        clip_model = self._get_obj_clip_model()
        visual = clip_model.visual
        x = view_cls.float()

        if hasattr(visual, "ln_post"):
            ln_width = int(visual.ln_post.normalized_shape[0])
            if x.size(-1) == ln_width:
                x = visual.ln_post(x)
                if getattr(visual, "proj", None) is not None:
                    x = x @ visual.proj
                return F.normalize(x, dim=-1)

        clip_dim = int(getattr(clip_model, "text_projection").shape[1])
        if x.size(-1) == clip_dim:
            return F.normalize(x, dim=-1)

        raise RuntimeError(
            f"CLIP scoring dimension mismatch: view feature dim={x.size(-1)} cannot be mapped by "
            f"{self.obj_clip_model_name} (expected ln_post width or embed dim). "
            "Choose a compatible CLIP model/checkpoint for the current precomputed view features."
        )

    def _compute_obs_embed(
        self,
        pano_embeds,
        pano_masks,
        view_llm_embeds,
        view_cls_fts,
        instructions,
        instruct_text_embeds,
        instruct_text_masks,
        instruct_text_ids=None,
        debug_infos=None,
    ):
        mean_obs = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                   torch.sum(pano_masks, 1, keepdim=True).clamp(min=1)
        if self.obs_pool_mode != 'obj_adaptive':
            return mean_obs

        out = []
        temp = max(self.obj_rel_temp, 1e-6)
        view_temp = max(self.obj_view_temp, 1e-6)
        for i, instruction in enumerate(instructions):
            phrase_list = self._extract_object_phrases(instruction)
            if len(phrase_list) == 0:
                out.append(mean_obs[i])
                continue
            phrase_llm, phrase_hidden = self._encode_object_phrases(instruction)
            valid = pano_masks[i].bool()
            if phrase_llm is None or phrase_hidden is None or valid.sum().item() == 0:
                out.append(mean_obs[i])
                continue

            txt_mask = instruct_text_masks[i].bool()
            txt = instruct_text_embeds[i][txt_mask]
            if txt.shape[0] == 0:
                out.append(mean_obs[i])
                continue

            instr_ctx = txt.mean(0, keepdim=True)

            # Phrase-to-instruction token mapping: use only matched token spans when available.
            if instruct_text_ids is not None:
                ins_ids = instruct_text_ids[i][txt_mask]
            else:
                ins_ids = None
            phrase_masks = self._phrase_token_match_mask(ins_ids, phrase_list)
            if phrase_masks is not None and phrase_masks.shape[0] == phrase_llm.shape[0]:
                if not torch.any(phrase_masks):
                    out.append(mean_obs[i])
                    continue
                phrase_instr_ctx = []
                for p_idx in range(phrase_masks.shape[0]):
                    m = phrase_masks[p_idx]
                    if torch.any(m):
                        phrase_instr_ctx.append(txt[m].mean(0))
                    else:
                        phrase_instr_ctx.append(instr_ctx.squeeze(0))
                phrase_instr_ctx = torch.stack(phrase_instr_ctx, dim=0)
            else:
                # If ids are available but mapping failed structurally, fall back to mean observation pooling.
                if ins_ids is not None:
                    out.append(mean_obs[i])
                    continue
                phrase_instr_ctx = instr_ctx.expand(phrase_llm.shape[0], -1)
            self._debug_print_phrase_token_map(instruction, ins_ids, phrase_list, phrase_masks)

            rel = F.cosine_similarity(phrase_llm, phrase_instr_ctx, dim=1)
            rel_w = F.softmax(rel / temp, dim=0)
            obj_query = (phrase_instr_ctx * rel_w.unsqueeze(1)).sum(0)
            score_query_vec = obj_query
            score_view_mat = None
            score_query_space = "llm"

            if self.obj_score_space == 'clip':
                phrase_clip = self._encode_object_phrases_clip(instruction)
                if phrase_clip is None:
                    out.append(mean_obs[i])
                    continue
                try:
                    clip_view = self._project_view_cls_for_clip(view_cls_fts[i])
                except Exception as exc:
                    if self.default_gpu and not self._obj_clip_warned:
                        print(f"[WARN] CLIP obj scoring disabled for this batch ({exc}); fallback to llm scoring.", flush=True)
                        self._obj_clip_warned = True
                    clip_view = None
                if clip_view is None:
                    if view_llm_embeds is None or view_llm_embeds.shape[:2] != pano_embeds.shape[:2]:
                        out.append(mean_obs[i])
                        continue
                    scores = F.cosine_similarity(view_llm_embeds[i], obj_query.unsqueeze(0), dim=1)
                else:
                    if phrase_clip.shape[0] != rel_w.shape[0]:
                        out.append(mean_obs[i])
                        continue
                    obj_query_clip = (phrase_clip * rel_w.unsqueeze(1)).sum(0)
                    obj_query_clip = F.normalize(obj_query_clip, dim=0)
                    scores = F.cosine_similarity(clip_view, obj_query_clip.unsqueeze(0), dim=1)
                    score_query_vec = obj_query_clip
                    score_view_mat = clip_view
                    score_query_space = "clip"
            else:
                if view_llm_embeds is None or view_llm_embeds.shape[:2] != pano_embeds.shape[:2]:
                    out.append(mean_obs[i])
                    continue
                scores = F.cosine_similarity(view_llm_embeds[i], obj_query.unsqueeze(0), dim=1)
                score_view_mat = view_llm_embeds[i]
            scores = scores.masked_fill(~valid, float('-inf'))
            attn_weights = F.softmax(scores / view_temp, dim=0)
            attn_weights = attn_weights * valid.float()
            attn_weights = attn_weights / attn_weights.sum().clamp(min=1e-8)
            top_idx = int(torch.argmax(attn_weights).item())
            should_print_obj_debug = (
                getattr(self.args, 'print_obj_phrases', False)
                or getattr(self.args, 'print_obj_cosine', False)
            )
            if (
                should_print_obj_debug
                and self.default_gpu
                and debug_infos is not None
                and i < len(debug_infos)
            ):
                info = debug_infos[i] or {}
                cand_vpids = info.get("cand_vpids", [])
                gt_next_vpid = info.get("gt_next_vpid", None)
                cand_scores = []
                for j, vpid in enumerate(cand_vpids):
                    if j < scores.shape[0]:
                        cand_scores.append(
                            {
                                "cand_idx": j,
                                "vpid": vpid,
                                "cosine": float(scores[j].detach().item()),
                                "attn_weight": float(attn_weights[j].detach().item()),
                                "is_gt_next": bool(gt_next_vpid is not None and vpid == gt_next_vpid),
                                "is_top1": bool(j == top_idx),
                            }
                        )
                phrase_rels = [
                    {
                        "phrase": phrase_list[k] if k < len(phrase_list) else f"phrase_{k}",
                        "cosine_to_instr_tokens": float(rel[k].detach().item()),
                        "weight": float(rel_w[k].detach().item()),
                    }
                    for k in range(rel.shape[0])
                ]
                query_norm = float(torch.norm(score_query_vec).detach().item())
                query_head = [float(x) for x in score_query_vec[:8].detach().cpu().tolist()]
                top_view_norm = None
                top_view_head = None
                if score_view_mat is not None and top_idx < score_view_mat.shape[0]:
                    top_view = score_view_mat[top_idx]
                    top_view_norm = float(torch.norm(top_view).detach().item())
                    top_view_head = [float(x) for x in top_view[:8].detach().cpu().tolist()]
                print(
                    "[print_obj_cosine]\n"
                    f"  instruction: {instruction}\n"
                    f"  score_space: {self.obj_score_space}\n"
                    f"  score_query_space: {score_query_space}\n"
                    f"  gt_next_vpid: {gt_next_vpid}\n"
                    f"  phrase_relevance: {phrase_rels}\n"
                    f"  score_query_norm: {query_norm}\n"
                    f"  score_query_head8: {query_head}\n"
                    f"  top_view_norm: {top_view_norm}\n"
                    f"  top_view_head8: {top_view_head}\n"
                    f"  cand_view_cosine: {cand_scores}\n",
                    flush=True,
                )
            out.append((pano_embeds[i] * attn_weights.unsqueeze(1)).sum(0))

        return torch.stack(out, dim=0)

    def _sanitize_instruction_for_prompt(self, instruction: str) -> str:
        """Strip prompt delimiter tokens that may appear inside RxR text and break [INST] parsing."""
        text = instruction
        for tok in ("[INST]", "[/INST]"):
            text = text.replace(tok, " ")
        return " ".join(text.split())

    def _truncate_instruction(self, instruction: str) -> str:
        """Word-level cap aligned with --max_instr_len (T5 / prompt budget)."""
        words = instruction.split()
        max_w = max(8, int(self.args.max_instr_len))
        if len(words) <= max_w:
            return instruction
        return " ".join(words[:max_w])

    def _language_variable(self, instructions, batch_view_lens, batch_rel_angles, batch_rel_dists, step=None):
        ''' Construct language variable.

        If use_sub_instr is enabled and a step index is provided, the prompt
        highlights the estimated current sub-instruction (Idea B).
        '''
        batch_qformer_text_inputs, batch_text_inputs = [], []
        for i, l in enumerate(batch_view_lens):
            instr = self._truncate_instruction(
                self._sanitize_instruction_for_prompt(instructions[i])
            )
            batch_qformer_text_inputs += [instr] * l

        for i in range(len(batch_view_lens)):
            instr = self._truncate_instruction(
                self._sanitize_instruction_for_prompt(instructions[i])
            )
            prompt = self.prompt.replace("{instruction}", instr)

            # Idea B: inject current sub-instruction focus into prompt
            if self.use_sub_instr and step is not None:
                cur_sub = _get_current_sub_instruction(
                    instr, step, self.args.max_action_steps
                )
                prompt = prompt.replace("{current_sub_instr}", cur_sub)

            candidate_dict = self._construct_candidate_dict(batch_rel_angles[i], batch_rel_dists[i])
            prompt = prompt.replace("{candidate}", str(candidate_dict))
            batch_text_inputs.append(prompt)
            
        return batch_qformer_text_inputs, batch_text_inputs

    def _get_gt_next_vpid(self, ob):
        gt_path = ob.get("gt_path", None)
        cur_vp = ob.get("viewpoint", None)
        if not gt_path or cur_vp is None:
            return None
        try:
            idx = gt_path.index(cur_vp)
        except ValueError:
            return None
        if idx < len(gt_path) - 1:
            return gt_path[idx + 1]
        return None

    def _build_view_llm_padded(self, view_embeds_flat, view_lens):
        """
        Convert flattened image-token embeddings from LLM space
        (sum(view_lens)*32, C) -> padded view-level tensor (B, max_view, C)
        by merging 32 image tokens per view with mean.
        """
        if view_embeds_flat is None:
            return None
        total_views = int(sum(view_lens))
        if total_views <= 0:
            return None
        if view_embeds_flat.size(0) % 32 != 0:
            return None
        n_views_from_flat = view_embeds_flat.size(0) // 32
        if n_views_from_flat != total_views:
            return None
        view_level = view_embeds_flat.reshape(total_views, 32, -1).mean(1)  # (N_view, C)
        view_split = torch.split(view_level, view_lens, 0)
        return pad_tensors_wgrad(view_split)  # (B, N_max, C)

    def _local_feature_variable(self, obs, gmaps, instructions, step=None):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_view_cls_fts, batch_loc_fts = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        batch_rel_angles, batch_rel_dists = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, cand_vpids = [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'])            # (257, 1024) or (3, 224, 224)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])

            cur_rel_angles, cur_rel_dists, cur_cand_pos_fts = gmaps[i].get_pos_fts(       # (n_candidates, 2), (n_candidates, 3), (n_candidates, 7)
                obs[i]['viewpoint'], cand_vpids, 
                obs[i]['heading'], obs[i]['elevation']
            )
            _, _, cur_start_pos_fts = gmaps[i].get_pos_fts(   # (1, 7)
                obs[i]['viewpoint'], [gmaps[i].start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )

            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)          # (n_candidates, 257, 1024) or (n_candidates, 3, 224, 224)
            view_img_cls_fts = view_img_fts[:, 0]             # (n_candidates, 1024)
            vp_loc_fts = np.zeros((len(view_img_fts), 14), dtype=np.float32)
            vp_loc_fts[:, :7] = cur_start_pos_fts
            vp_loc_fts[:, 7:] = cur_cand_pos_fts

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_view_cls_fts.append(torch.from_numpy(view_img_cls_fts))
            batch_loc_fts.append(torch.from_numpy(vp_loc_fts))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))
            batch_rel_angles.append(cur_rel_angles)
            batch_rel_dists.append(cur_rel_dists[:, 0] * 30)  # distances are normalized by 30m

        # pad features to max_len
        batch_view_img_fts = torch.cat(batch_view_img_fts).to(self.device)
        batch_view_cls_fts = pad_tensors(batch_view_cls_fts).to(self.device)
        batch_loc_fts = torch.cat(batch_loc_fts).to(self.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.device)
        batch_qformer_text_inputs, batch_text_inputs = self._language_variable(
            instructions, batch_view_lens, batch_rel_angles, batch_rel_dists, step=step
        )

        return {
            'view_cls_fts': batch_view_cls_fts,
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'view_lens': batch_view_lens, 'cand_vpids': batch_cand_vpids,
            'qformer_text_inputs': batch_qformer_text_inputs,
            'text_inputs': batch_text_inputs,
        }

    def _nav_vp_variable(self, pano_embeds, cand_vpids, view_lens):

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [self._get_node_history_embed(gmap, vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            _, _, gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).to(self.device)
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).to(self.device)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).to(self.device)
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).to(self.device)

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.to(self.device)

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).to(self.device)

    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).to(self.device)

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                cur_vp = ob['viewpoint']
                scan = ob['scan']
                # Use the full connectivity shortest path to avoid invalid jumps in saved trajectories.
                # gmap graph can be partially observed and may return pseudo-direct hops for unseen pairs.
                gt_path = self.env.shortest_paths[scan][cur_vp][action]
                step_path = gt_path[1:] if len(gt_path) > 1 else [action]
                traj[i]['path'].append(step_path)

                if len(step_path) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = step_path[-2]

                scanvp_key = f'{scan}_{prev_vp}'
                if (scanvp_key in self.scanvp_cands) and (action in self.scanvp_cands[scanvp_key]):
                    viewidx = self.scanvp_cands[scanvp_key][action]
                    heading = (viewidx % 12) * math.radians(30)
                    elevation = (viewidx // 12 - 1) * math.radians(30)
                else:
                    # Fallback: keep current camera pose if candidate cache for this edge is unavailable.
                    heading = ob['heading']
                    elevation = ob['elevation']
                self.env.env.sims[i].newEpisode([scan], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'thoughts': [],
        } for ob in obs]

        # Language inputs
        instructions = [ob['instruction'] for ob in obs]
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        IL_loss = 0.   
        acc_IL_loss = 0.
        acc_g_loss = 0.  
        acc_progress_loss = 0.

        for t in range(self.args.max_action_steps):
            # for logging: how many trajectories are still active at this step
            self.logs['total'].append(int(np.sum(np.logical_not(ended))))

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
                # Truncate autograd history between timesteps when step-wise backward is enabled.
                # Otherwise cached node embeddings keep previous-step graphs and trigger
                # "Trying to backward through the graph a second time".
                if self.args.step_update:
                    for k, v in gmap.node_embeds.items():
                        gmap.node_embeds[k][0] = v[0].detach()

            # graph representation  — pass step t for Idea B sub-instruction tracking
            local_inputs = self._local_feature_variable(obs, gmaps, instructions, step=t)
            history_text_embeds = None

            # IPE-before-encoder: Q-Former + merged inputs → panorama/context → T5 encoder (emphasis on raw instruct embeds).
            ipe_pre = self.use_ipe and self.ipe_stage in ('pre_encoder', 'both')
            if ipe_pre:
                prep_out = self.NavGPT('thought_prepare', local_inputs)
                local_inputs['view_llm_fts'] = prep_out['view_embeds']
                view_llm_padded = self._build_view_llm_padded(
                    prep_out['view_embeds'], local_inputs['view_lens'].tolist()
                )
                split_loc_fts = torch.split(local_inputs['loc_fts'], local_inputs['view_lens'].tolist(), 0)
                local_inputs['loc_fts'] = pad_tensors_wgrad(split_loc_fts)
                pano_embeds, pano_masks = self.NavGPT('panorama', local_inputs)
                obs_debug_infos = [
                    {
                        "cand_vpids": local_inputs['cand_vpids'][ii],
                        "gt_next_vpid": self._get_gt_next_vpid(obs[ii]),
                    }
                    for ii in range(len(instructions))
                ]
                avg_pano_embeds = self._compute_obs_embed(
                    pano_embeds,
                    pano_masks,
                    view_llm_padded,
                    local_inputs['view_cls_fts'],
                    instructions,
                    prep_out['instruct_text_embeds_pre'],
                    prep_out['instruct_text_masks_pre'],
                    prep_out.get('instruct_text_ids_pre'),
                    obs_debug_infos,
                )
                hist_embeds = None
                history_text_embeds = None
                if self.use_history_token:
                    core = self._get_navgpt_core()
                    history_text_embeds = core.build_history_text_embed(
                        prep_out['instruct_text_embeds_pre'],
                        prep_out['instruct_text_masks_pre'],
                    )

                if self.ipe_context_mode in ('both', 'hist_only'):
                    hist_embeds_batch = []
                    for i, gmap in enumerate(gmaps):
                        visited_vps = [
                            vp for vp in gmap.node_embeds
                            if gmap.graph.visited(vp)
                        ]
                        if visited_vps:
                            vp_stack = torch.stack(
                                [self._get_node_history_embed(gmap, vp) for vp in visited_vps], dim=0
                            )
                            if self.ipe_hist_pool == 'mean':
                                hist_embeds_batch.append(vp_stack.mean(0))
                            else:
                                cur_vp = obs[i]['viewpoint']
                                dists = torch.tensor(
                                    [gmap.graph.distance(cur_vp, vp) for vp in visited_vps],
                                    device=vp_stack.device, dtype=vp_stack.dtype,
                                )
                                tau = max(self.ipe_hist_dist_tau, 1e-6)
                                w = F.softmax(-dists / tau, dim=0)
                                hist_embeds_batch.append((vp_stack * w.unsqueeze(1)).sum(0))
                        else:
                            hist_embeds_batch.append(torch.zeros_like(avg_pano_embeds[i]))
                    hist_embeds = torch.stack(hist_embeds_batch, dim=0)

                if self.ipe_context_mode == 'both':
                    context_embed = avg_pano_embeds + hist_embeds
                elif self.ipe_context_mode == 'obs_only':
                    context_embed = avg_pano_embeds
                else:
                    context_embed = hist_embeds

                local_inputs['thought_prep'] = prep_out['thought_prep']
                local_inputs['ipe_context_embed'] = context_embed
                local_outputs = self.NavGPT('thought', local_inputs)
                local_inputs.pop('thought_prep', None)
                local_inputs.pop('ipe_context_embed', None)
                view_embeds = local_outputs['view_embeds']
                instruct_text_embeds = local_outputs['instruct_text_embeds']
                instruct_text_masks = local_outputs['instruct_text_masks']
                instruct_text_ids = local_outputs.get('instruct_text_ids')
                thoughts, generation_loss = local_outputs['output_text'], local_outputs['loss']
                local_inputs['text_embeds'] = instruct_text_embeds
                local_inputs['text_masks'] = instruct_text_masks
                local_inputs['view_llm_fts'] = view_embeds
            else:
                local_outputs = self.NavGPT('thought', local_inputs)
                view_embeds = local_outputs['view_embeds']
                instruct_text_embeds = local_outputs['instruct_text_embeds']
                instruct_text_masks = local_outputs['instruct_text_masks']
                instruct_text_ids = local_outputs.get('instruct_text_ids')
                thoughts, generation_loss = local_outputs['output_text'], local_outputs['loss']
                local_inputs['text_embeds'] = instruct_text_embeds
                local_inputs['text_masks'] = instruct_text_masks
                local_inputs['view_llm_fts'] = view_embeds
                view_llm_padded = self._build_view_llm_padded(
                    view_embeds, local_inputs['view_lens'].tolist()
                )
                split_loc_fts = torch.split(local_inputs['loc_fts'], local_inputs['view_lens'].tolist(), 0)
                local_inputs['loc_fts'] = pad_tensors_wgrad(split_loc_fts)
                pano_embeds, pano_masks = self.NavGPT('panorama', local_inputs)
                obs_debug_infos = [
                    {
                        "cand_vpids": local_inputs['cand_vpids'][ii],
                        "gt_next_vpid": self._get_gt_next_vpid(obs[ii]),
                    }
                    for ii in range(len(instructions))
                ]
                avg_pano_embeds = self._compute_obs_embed(
                    pano_embeds,
                    pano_masks,
                    view_llm_padded,
                    local_inputs['view_cls_fts'],
                    instructions,
                    instruct_text_embeds,
                    instruct_text_masks,
                    instruct_text_ids,
                    obs_debug_infos,
                )

            # Idea A: IPE after T5 encoder (legacy) — optional second pass when --ipe_stage post_encoder or both.
            if self.use_ipe and self.ipe_stage in ('post_encoder', 'both'):
                hist_embeds = None
                if self.ipe_context_mode in ('both', 'hist_only'):
                    hist_embeds_batch = []
                    for i, gmap in enumerate(gmaps):
                        visited_vps = [
                            vp for vp in gmap.node_embeds
                            if gmap.graph.visited(vp)
                        ]
                        if visited_vps:
                            vp_stack = torch.stack(
                                [self._get_node_history_embed(gmap, vp) for vp in visited_vps], dim=0
                            )
                            if self.ipe_hist_pool == 'mean':
                                hist_embeds_batch.append(vp_stack.mean(0))
                            else:
                                cur_vp = obs[i]['viewpoint']
                                dists = torch.tensor(
                                    [gmap.graph.distance(cur_vp, vp) for vp in visited_vps],
                                    device=vp_stack.device, dtype=vp_stack.dtype,
                                )
                                tau = max(self.ipe_hist_dist_tau, 1e-6)
                                w = F.softmax(-dists / tau, dim=0)
                                hist_embeds_batch.append((vp_stack * w.unsqueeze(1)).sum(0))
                        else:
                            hist_embeds_batch.append(torch.zeros_like(avg_pano_embeds[i]))
                    hist_embeds = torch.stack(hist_embeds_batch, dim=0)

                if self.ipe_context_mode == 'both':
                    context_embed = avg_pano_embeds + hist_embeds
                elif self.ipe_context_mode == 'obs_only':
                    context_embed = avg_pano_embeds
                else:
                    context_embed = hist_embeds

                ipe_inputs = {
                    'context_embed': context_embed,
                    'instruct_text_embeds': instruct_text_embeds,
                    'instruct_text_masks': instruct_text_masks,
                }
                local_inputs['text_embeds'] = self.NavGPT('ipe', ipe_inputs)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    if self.use_history_token:
                        core = self._get_navgpt_core()
                        if history_text_embeds is None:
                            history_text_embeds = core.build_history_text_embed(
                                instruct_text_embeds,
                                instruct_text_masks,
                            )
                        node_hist = core.build_history_token(
                            avg_pano_embeds[i].unsqueeze(0),
                            history_text_embeds[i].unsqueeze(0),
                        ).squeeze(0)
                        gmap.update_node_hist_embed(i_vp, node_hist, rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(local_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])
                            if self.use_history_token:
                                node_hist = core.build_history_token(
                                    pano_embeds[i, j].unsqueeze(0),
                                    history_text_embeds[i].unsqueeze(0),
                                ).squeeze(0)
                                gmap.update_node_hist_embed(i_cand_vp, node_hist)

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(local_inputs)

            # add [stop] token for pano embeddings
            nav_vp_inputs = self._nav_vp_variable(pano_embeds, local_inputs['cand_vpids'], local_inputs['view_lens'])
            nav_inputs.update(nav_vp_inputs)

            nav_outs = self.NavGPT('action', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                                        
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action_r4r(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                    imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                )

                IL_loss = self.criterion(nav_logits, nav_targets)
                IL_loss = IL_loss * train_ml / (batch_size * self.args.accumulate_grad_step)
                acc_IL_loss += IL_loss
                step_loss = IL_loss

                if self.use_learned_progress_stop and nav_outs.get('progress_score') is not None:
                    progress_pred = nav_outs['progress_score']
                    target_vals = []
                    for i, ob in enumerate(obs):
                        if ended[i]:
                            target_vals.append(float(progress_pred[i].detach().item()))
                            continue
                        if self.progress_target == 'gt_path_ratio':
                            denom = max(len(ob.get('gt_path', [])) - 1, 1)
                            target_vals.append(min(float(t) / float(denom), 1.0))
                        else:
                            denom = max(self.args.max_action_steps - 1, 1)
                            target_vals.append(min(float(t) / float(denom), 1.0))
                    progress_tgt = torch.tensor(
                        target_vals, device=progress_pred.device, dtype=progress_pred.dtype
                    )
                    progress_loss = F.mse_loss(progress_pred, progress_tgt, reduction='mean')
                    progress_loss = (
                        progress_loss
                        * self.progress_aux_weight
                        * train_ml
                        / self.args.accumulate_grad_step
                    )
                    acc_progress_loss += progress_loss
                    step_loss += progress_loss

                if generation_loss is not None:
                    g_loss = generation_loss * train_ml / (batch_size * self.args.accumulate_grad_step)
                    acc_g_loss += g_loss
                    step_loss += g_loss
                
                if self.args.step_update:
                    step_loss.backward()
                                                 
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_steps - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        # Use full connectivity shortest path for the final stop hop as well.
                        # gmap may be incomplete and can yield pseudo-direct hops not valid on eval graph.
                        scan = obs[i]['scan']
                        cur_vp = obs[i]['viewpoint']
                        stop_path = self.env.shortest_paths[scan][cur_vp][stop_node]
                        traj[i]['path'].append(stop_path[1:] if len(stop_path) > 1 else [stop_node])
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }
                if self.args.output_thought:
                    traj[i]['thoughts'].append(thoughts[i])

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            if generation_loss is not None:
                self.logs['generation_loss'].append(acc_g_loss.item())
            if self.use_learned_progress_stop:
                self.logs['progress_loss'].append(acc_progress_loss.item())
            self.logs['IL_loss'].append(acc_IL_loss.item())
            # Loss for the whole trajectory
            self.loss = acc_IL_loss + acc_g_loss + acc_progress_loss

        return traj
