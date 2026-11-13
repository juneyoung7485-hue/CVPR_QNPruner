import argparse
from pathlib import Path
import sys
import torch
import os
import json
import time
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llm_token_pruning import PRUNE_STATE
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_videos, get_model_name_from_path
from llava.eval.scene_cache import (
    SceneCache, enable_scene_feature_cache, first_question_per_scene, group_by_scene,
)
from llava.eval.scene_filter import filter_scene_questions
# The Fig.4 selection exporter lives in the repo-root visualization/ package so
# that figure code is never a dependency of running the model.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualization.export_token_selection import (  # noqa: E402
    add_selection_export_args, apply_selection_export_args, enabled as selection_export_enabled,
    record_llm_prune,
)
# Module scope, not inside the eval loop's function: a function-level import
# would make the name local to the whole function and the per-question export
# below would hit UnboundLocalError.
from llava.model.language_model.llm_token_pruning import installed as _pruner  # noqa: E402
from llava.model.multimodal_encoder.gt_oracle import (
    GT_BOX_POOLING, GT_POOLINGS, attach_gt_oracle, scene_id_from_video,
)

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # The Fig.4 selection export is opt-in; this is a no-op without
    # --save-token-selection, leaving inference on the stock path.
    apply_selection_export_args(args)

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    with open(args.question_file, 'r') as file:
        questions = json.load(file)
    questions = filter_scene_questions(questions, getattr(args, 'scene', None))
    if not questions:
        raise ValueError('The question file is empty.')
    if getattr(args, 'scene', None):
        print(f"[SceneFilter] {args.scene}: {len(questions)} ScanQA questions", flush=True)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if args.group_by_scene:
        questions = group_by_scene(questions)

    if getattr(args, 'one_question_per_scene', False):
        questions = first_question_per_scene(questions)

    # Applied after grouping so a truncated run visits exactly the same
    # scenes, in the same order, as the head of a full evaluation. With
    # --one-question-per-scene it counts scenes rather than questions.
    if getattr(args, 'limit', 0) > 0:
        questions = questions[:args.limit]

    # Load only after validating/filtering the requested benchmark scene.
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    scene_cache = None
    if args.scene_cache_capacity > 0:
        scene_cache = SceneCache(
            capacity=args.scene_cache_capacity,
            device=model.device,
            dtype=torch.bfloat16,
        )
    # --- GT Query-Relevant Object Oracle (diagnostic experiment) ---------
    # Attaching the runtime alone changes nothing: with POOLING=3dzip it only
    # records GT-retention statistics, and POOLING=3dzip_gt_oracle additionally
    # forces GT voxels into the anchor set.
    gt_oracle = None
    gt_stats_file = None
    if args.scanqa_gt_file:
        tower = model.get_model().get_video_tower()
        if args.feature_cache:
            # The scene feature cache is keyed by scene only, so a cache hit
            # would reuse the previous question's compression (and skip the
            # forward that computes GT stats). Correctness first here.
            print('[GT ORACLE][WARN] --feature-cache is unsafe with query-dependent '
                  'compression; disabling it for this run.', flush=True)
            args.feature_cache = False
        gt_oracle = attach_gt_oracle(
            tower,
            scanqa_gt_file=args.scanqa_gt_file,
            scannet_gt_root=args.scannet_gt_root,
            scannet_raw_root=args.scannet_raw_root,
            cache_dir=args.gt_cache_dir,
            nn_max_dist=args.gt_nn_max_dist,
            voxel_min_frac=args.gt_voxel_min_frac,
            max_anchor_frac=args.gt_max_anchor_frac,
            debug=args.gt_oracle_debug,
        )
        stats_path = args.gt_stats_file or os.path.join(
            os.path.dirname(os.path.expanduser(args.answers_file)), 'gt_oracle_stats.jsonl')
        os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
        gt_stats_file = open(stats_path, 'w')
        print(f'[GT ORACLE] stats -> {stats_path}', flush=True)
        if args.v0_gt_dump_dir:
            if os.environ.get('POOLING') != GT_BOX_POOLING:
                raise ValueError(f'--v0-gt-dump-dir needs POOLING={GT_BOX_POOLING}.')
            gt_oracle.dump_enabled = True
            print(f'[v0_gt] per-question selections -> {args.v0_gt_dump_dir}', flush=True)
    elif os.environ.get('POOLING') in GT_POOLINGS:
        raise ValueError(f"POOLING={os.environ.get('POOLING')} needs --scanqa-gt-file "
                         '(the original ScanQA annotation with object_ids).')

    if args.feature_cache and getattr(model.get_model().get_video_tower(), 'query_dependent', False):
        # Query-dependent compression: a scene-keyed cache would reuse the
        # first question's anchors for every later question of that scene.
        print('[QueryGuided][WARN] --feature-cache is unsafe with query-dependent '
              'compression; disabling it for this run.', flush=True)
        args.feature_cache = False
    if args.feature_cache:
        enable_scene_feature_cache(model, capacity=8)

    # Per-question naming for the Fig.4 selection records. A vision-path
    # method never reads the question text, so the once-per-scene default
    # already describes the whole scene and its record stays unlabelled.
    # FastV and SparseVLM do read it -- they prune on the text's attention --
    # so for those --token-selection-every-question is the mode that says
    # anything: it records each question under its own id.
    record_per_question = (selection_export_enabled()
                           and not getattr(args, 'token_selection_once_per_scene', True))

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    n_visual_list = []
    latency_list = []
    memory_list = []
    pbar = tqdm(questions)
    for line in pbar:
        idx = line["question_id"]
        video_file = line["video"]
        video_path = os.path.join(args.video_folder, video_file)
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if scene_cache is not None:
            entry = scene_cache.get(video_path, processor['video'], cur_prompt)
            images_tensor     = entry['images']
            depths_tensor     = entry['depths']
            poses_tensor      = entry['poses']
            intrinsics_tensor = entry['intrinsics']
        else:
            videos_dict = process_videos(
                video_path,
                processor['video'],
                mode='uniform',
                device=model.device,
                text=cur_prompt
            )
            images_tensor     = videos_dict['images'].to(model.device, dtype=torch.bfloat16)
            depths_tensor     = videos_dict['depths'].to(model.device, dtype=torch.bfloat16)
            poses_tensor      = videos_dict['poses'].to(model.device,  dtype=torch.bfloat16)
            intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch.bfloat16)

        # Bind scene id for the feature cache (no-op if the cache isn't installed).
        model.get_model().get_video_tower().scene_key = video_path
        # Name the sample for the compression debug lines. Harmless when no
        # decoder-side pruning is installed.
        PRUNE_STATE['scene_id'] = video_file
        PRUNE_STATE['question_id'] = idx
        if gt_oracle is not None:
            # Always overwrite so the previous question's GT can never leak.
            gt_oracle.set_context(question_id=idx, scene_id=scene_id_from_video(video_file))
        if record_per_question:
            os.environ['TOKEN_SEL_QID'] = str(idx)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                image_sizes=None,
                # The raw user question (SQA3D: situation + question), no
                # system prompt or chat template. Read only by query-guided methods.
                query_texts=[cur_prompt],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=True,
            )
        torch.cuda.synchronize()
        latency_list.append(time.time() - t_start)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if gt_oracle is not None:
            record = dict(gt_oracle.last_stats)
            record.update(question=cur_prompt, prediction=outputs,
                          pooling=os.environ.get('POOLING'),
                          voxel_size=os.environ.get('VOXEL_SIZE'))
            record.setdefault('question_id', idx)
            gt_stats_file.write(json.dumps(record) + '\n')
            gt_stats_file.flush()
            if args.v0_gt_dump_dir:
                gt_oracle.save_dump(args.v0_gt_dump_dir, question=cur_prompt,
                                    prediction=outputs)
            gt_oracle.clear_context()
        # Wall time of model.generate (vision tower, compression and LLM decoding), synchronized above / below.
        question_metadata = {'generate_call_s': latency_list[-1]}

        # Track visual token count + KV cache calculation
        video_tower = model.get_model().get_video_tower()

        # FastV / SparseVLM decide inside the decoder, so their Fig.4 record
        # can only be written once generation has run. It is per question by
        # nature: the same scene compresses differently for a different query.
        if selection_export_enabled():
            record_llm_prune(scene_key=video_path, pruner=_pruner(),
                             video_tower=video_tower,
                             voxel_size=float(os.environ.get('VOXEL_SIZE', '0.2')))

        if hasattr(video_tower, '_voxel_xyz') and video_tower._voxel_xyz is not None:
            _n_vis = video_tower._voxel_xyz[0].shape[0]
            n_visual_list.append(_n_vis)
            # KV cache size: 2(K,V) × n_layers × seq_len × n_heads × head_dim × 2(bf16)
            cfg = model.config
            n_layers = cfg.num_hidden_layers
            n_kv_heads = getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            seq_len = input_ids.shape[1] - 1 + _n_vis  # text tokens + visual tokens
            kv_bytes = 2 * n_layers * seq_len * n_kv_heads * head_dim * 2  # bf16=2bytes
            kv_mb = kv_bytes / 1024**2
            memory_list.append(kv_mb)
            postfix = dict(v_num=_n_vis, ms=f"{latency_list[-1]*1000:.0f}", kv=f"{kv_mb:.0f}MB")
            if hasattr(video_tower, '_sel_times') and video_tower._sel_times:
                postfix["sel_ms"] = f"{video_tower._sel_times[-1]:.1f}"
            if hasattr(video_tower, '_algo_times') and video_tower._algo_times:
                postfix["algo_ms"] = f"{video_tower._algo_times[-1]:.1f}"
            pbar.set_postfix(**postfix)


        # === LLM attention -> voxel visualization (--visualize flag) ===
        if args.visualize:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np

            vis_dir = os.path.join(os.path.dirname(answers_file), 'llm_attn_vis')
            os.makedirs(vis_dir, exist_ok=True)

            with torch.inference_mode():
                fwd_out = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    depths=depths_tensor,
                    poses=poses_tensor,
                    intrinsics=intrinsics_tensor,
                    output_attentions=True,
                    return_dict=True,
                    query_texts=[cur_prompt],
                )
            # voxel xyz
            video_tower = model.get_model().get_video_tower()
            voxel_xyz = video_tower._voxel_xyz[0].cpu().float()  # (n_vox, 3)
            n_visual = voxel_xyz.shape[0]

            # Locate visual token positions
            image_pos = (input_ids[0] == -200).nonzero(as_tuple=True)[0][0].item()
            vis_start = image_pos
            vis_end = vis_start + n_visual

            # Mid-layer (16th) attention: (B, n_heads, seq_len, seq_len)
            last_attn = fwd_out.attentions[15].float()
            last_attn = last_attn.mean(dim=1)  # avg over heads: (B, seq_len, seq_len)
            # query (text after visual) -> visual token attention
            query_to_vis = last_attn[0, vis_end:, vis_start:vis_end]  # (n_query, n_visual)
            avg_attn = query_to_vis.mean(dim=0).cpu().numpy()  # (n_visual,)

            # === Attention distribution analysis: sum of attention from query tokens to each region ===
            seq_len = last_attn.shape[-1]
            # query tokens (vis_end ~ end) -> attention to each region
            query_attn = last_attn[0, vis_end:, :]  # (n_query, seq_len)
            attn_to_system = query_attn[:, :vis_start].sum(dim=-1).mean().item()       # -> system
            attn_to_visual = query_attn[:, vis_start:vis_end].sum(dim=-1).mean().item() # -> 3D visual
            attn_to_query  = query_attn[:, vis_end:].sum(dim=-1).mean().item()          # -> query text
            print(f"[attn dist] {idx} | system={attn_to_system:.4f}, 3D_visual={attn_to_visual:.4f}, "
                  f"query_text={attn_to_query:.4f} | "
                  f"n_sys={vis_start}, n_vis={n_visual}, n_query={seq_len - vis_end}, total={seq_len}")

            xyz = voxel_xyz.numpy()

            # === Load RGB point cloud ===
            from PIL import Image as PILImage
            scene_dir = video_path if os.path.isdir(video_path) else os.path.join(args.video_folder, video_file)
            color_dir = os.path.join(scene_dir, 'color')
            depth_dir = os.path.join(scene_dir, 'depth')
            pose_dir_path = os.path.join(scene_dir, 'pose')
            intrinsic_file = os.path.join(scene_dir, 'intrinsic', 'intrinsic_depth.txt')
            K = np.loadtxt(intrinsic_file)[:3, :3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            axis_file = os.path.join(scene_dir, 'axis_align_matrix.txt')
            if os.path.exists(axis_file):
                #print(f"[AxisAlign] Using {axis_file}")
                axis_align = np.loadtxt(axis_file).reshape(4, 4)
            else:
                axis_align = np.eye(4)

            rgb_frames = sorted([int(f.split('.')[0]) for f in os.listdir(depth_dir) if f.endswith('.png')])
            step = max(1, len(rgb_frames) // 15)
            rgb_frames = rgb_frames[::step][:15]
            all_pts, all_rgb = [], []
            for fi in rgb_frames:
                pose_f = np.loadtxt(os.path.join(pose_dir_path, f'{fi}.txt'))
                if np.any(np.isinf(pose_f)) or np.any(np.isnan(pose_f)):
                    continue
                dep = np.array(PILImage.open(os.path.join(depth_dir, f'{fi}.png')), dtype=np.float32) / 1000.0
                Hd, Wd = dep.shape
                col_path = os.path.join(color_dir, f'{fi}.jpg')
                if not os.path.exists(col_path):
                    col_path = os.path.join(color_dir, f'{fi}.png')
                col = np.array(PILImage.open(col_path).convert('RGB').resize((Wd, Hd))) / 255.0
                v_idx, u_idx = np.where(dep > 0.1)
                z_val = dep[v_idx, u_idx]
                x_val = (u_idx - cx) * z_val / fx
                y_val = (v_idx - cy) * z_val / fy
                pts_cam = np.stack([x_val, y_val, z_val, np.ones_like(z_val)], axis=-1)
                pts_w = (axis_align @ pose_f @ pts_cam.T).T[:, :3]
                if len(pts_w) > 3000:
                    si = np.random.choice(len(pts_w), 3000, replace=False)
                    pts_w = pts_w[si]; rgb_vals = col[v_idx[si], u_idx[si]]
                else:
                    rgb_vals = col[v_idx, u_idx]
                all_pts.append(pts_w); all_rgb.append(rgb_vals)
            scene_pts = np.concatenate(all_pts, axis=0)
            scene_rgb = np.concatenate(all_rgb, axis=0)

            fig = plt.figure(figsize=(28, 8))
            fig.suptitle(f'[{idx}] Q: {cur_prompt}\nA: {outputs}', fontsize=10, wrap=True)

            # Left: attention by voxel index
            ax1 = fig.add_subplot(131)
            ax1.bar(np.arange(n_visual), avg_attn, color='coral', alpha=0.7, width=1.0)
            ax1.set_xlabel('Voxel Index')
            ax1.set_ylabel('LLM Attention')
            ax1.set_title(f'Attention by Index (n={n_visual})')
            top_k = min(10, n_visual)
            top_idx = np.argsort(avg_attn)[-top_k:]
            for ti in top_idx:
                ax1.annotate(f'{ti}', (ti, avg_attn[ti]), fontsize=6, ha='center', va='bottom')

            # Middle: RGB point cloud
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(scene_pts[:, 0], scene_pts[:, 1], scene_pts[:, 2],
                        c=scene_rgb, s=0.3, alpha=0.5)
            ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
            ax2.set_title(f'RGB Point Cloud ({len(scene_pts)} pts)')

            # Right: 3D attention heatmap
            ax3 = fig.add_subplot(133, projection='3d')
            sc = ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                             c=avg_attn, cmap='hot', s=15, alpha=0.8)
            plt.colorbar(sc, ax=ax3, label='LLM Attention', shrink=0.6)
            ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
            ax3.set_title('3D Attention Heatmap')

            vis_prefix = 'scanqa'
            save_path = os.path.join(vis_dir, f'{vis_prefix}_{idx}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            # === Plotly interactive HTML ===
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                print(f"[VIS] {save_path}  (plotly not installed, skipping HTML)")
                del fwd_out, last_attn
                # skip to answer writing
                ans_id = shortuuid.uuid()
                ans_file.write(json.dumps({"question_id": idx,
                                           "prompt": cur_prompt,
                                           "text": outputs,
                                           "answer_id": ans_id,
                                           "model_id": model_name,
                                           "metadata": {}}) + "\n")
                ans_file.flush()
                continue

            fig_html = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=['RGB Point Cloud', '3D Attention Heatmap'],
                horizontal_spacing=0.02,
            )

            # RGB point cloud
            rgb_str = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in scene_rgb]
            fig_html.add_trace(go.Scatter3d(
                x=scene_pts[:, 0], y=scene_pts[:, 1], z=scene_pts[:, 2],
                mode='markers',
                marker=dict(size=1.2, color=rgb_str, opacity=0.7),
                name='RGB',
            ), row=1, col=1)

            # Attention heatmap on voxels
            fig_html.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=avg_attn,
                    colorscale='Hot',
                    opacity=0.85,
                    colorbar=dict(title='Attn', x=1.0, len=0.6),
                    cmin=0,
                ),
                text=[f'vox {i}<br>attn={avg_attn[i]:.6f}' for i in range(n_visual)],
                hoverinfo='text',
                name='Attention',
            ), row=1, col=2)

            scene_cfg = dict(aspectmode='data')
            fig_html.update_layout(
                title=f'[{idx}] Q: {cur_prompt}<br>A: {outputs}',
                width=1600, height=700,
                scene=scene_cfg, scene2=scene_cfg,
                showlegend=False,
            )

            html_path = os.path.join(vis_dir, f'{vis_prefix}_{idx}.html')
            fig_html.write_html(html_path)
            print(f"[VIS] {save_path}  [HTML] {html_path}  (n_vox={n_visual}, n_query={query_to_vis.shape[0]})")
            del fwd_out, last_attn

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": question_metadata}) + "\n")
        ans_file.flush()
    ans_file.close()
    if gt_stats_file is not None:
        gt_stats_file.close()

    # Print visual token statistics
    if n_visual_list:
        import numpy as np
        arr = np.array(n_visual_list)
        print(f"\n[Visual Tokens] avg={arr.mean():.1f}, min={arr.min()}, max={arr.max()}, std={arr.std():.1f}, n={len(arr)}")

    # FastV / SparseVLM prune inside the decoder, so the count above is what
    # *entered* the LLM; this line reports what survived and the layer average.
    if _pruner() is not None and _pruner().summary():
        print(_pruner().summary())

    # Print latency statistics
    if latency_list:
        import numpy as np
        lat = np.array(latency_list)
        print(f"[Latency] avg={lat.mean()*1000:.1f}ms, min={lat.min()*1000:.1f}ms, max={lat.max()*1000:.1f}ms, std={lat.std()*1000:.1f}ms, n={len(lat)}")

    # Print KV cache statistics
    if memory_list:
        import numpy as np
        mem = np.array(memory_list)
        print(f"[KV Cache] avg={mem.mean():.0f}MB, min={mem.min():.0f}MB, max={mem.max():.0f}MB, n={len(mem)}")

    # Pooling algorithm timing statistics
    video_tower = model.get_model().get_video_tower()
    if hasattr(video_tower, '_pool_times') and video_tower._pool_times:
        import numpy as np
        pt = np.array(video_tower._pool_times)
        print(f"[Pooling] avg={pt.mean():.1f}ms, min={pt.min():.1f}ms, max={pt.max():.1f}ms, std={pt.std():.1f}ms, n={len(pt)}")
    if hasattr(video_tower, '_sel_times') and video_tower._sel_times:
        import numpy as np
        st = np.array(video_tower._sel_times)
        print(f"[Selection] avg={st.mean():.1f}ms, min={st.min():.1f}ms, max={st.max():.1f}ms, std={st.std():.1f}ms, n={len(st)}")
    if hasattr(video_tower, '_algo_times') and video_tower._algo_times:
        import numpy as np
        at = np.array(video_tower._algo_times)
        print(f"[Algorithm] avg={at.mean():.1f}ms, min={at.min():.1f}ms, max={at.max():.1f}ms, std={at.std():.1f}ms, n={len(at)}")

    if scene_cache is not None:
        total = scene_cache.hits + scene_cache.misses
        hit_pct = 100.0 * scene_cache.hits / max(1, total)
        print(f"[SceneCache] hits={scene_cache.hits}, misses={scene_cache.misses}, hit_ratio={hit_pct:.1f}%")
    if hasattr(video_tower, 'feature_cache_hits'):
        total_f = video_tower.feature_cache_hits + video_tower.feature_cache_misses
        hit_pct_f = 100.0 * video_tower.feature_cache_hits / max(1, total_f)
        print(f"[FeatureCache] hits={video_tower.feature_cache_hits}, misses={video_tower.feature_cache_misses}, hit_ratio={hit_pct_f:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/llava3d-v1.5-7b-task-v3-tuning")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="/data/LLaVA-3D-Pretrain") #"playground/data/LLaVA-3D-Pretrain"
    parser.add_argument("--question-file", type=str, default="playground/data/annotations/llava3d_sqa3d_val_question.json")
    parser.add_argument("--scene",
                        help="Run only these scenes' existing ScanQA questions. One scene ID or "
                             "path, or a comma-separated list (scene0011_00,scene0015_00). "
                             "Omit to run every scene in the question file.")
    parser.add_argument("--answers-file", type=str, default="./llava3d_sqa3d_val_answer_pred.json")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", help="LLM attention voxel visualization")
    parser.add_argument("--group-by-scene", action="store_true", default=True,
                        help="Sort questions by scene so cache hits maximize.")
    parser.add_argument("--no-group-by-scene", dest="group_by_scene", action="store_false",
                        help="Preserve the original question order (disables scene grouping).")
    parser.add_argument("--scene-cache-capacity", type=int, default=2,
                        help="RGB-D tensor cache: skip disk re-reads for same-scene "
                             "questions (default on, capacity 2). 0 = disable.")
    parser.add_argument("--feature-cache", action="store_true", default=False,
                        help="Per-scene vision-feature cache: run the vision tower "
                             "once per scene, reuse for the rest (default off).")
    parser.add_argument("--no-feature-cache", dest="feature_cache", action="store_false",
                        help="Explicitly disable the per-scene vision-feature cache.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N questions. 0 = all. "
                             "Useful for visualization runs and on/off A/B checks. "
                             "With --one-question-per-scene this counts scenes.")
    # --- GT Query-Relevant Object Oracle -------------------------------
    parser.add_argument("--scanqa-gt-file", type=str, default=None,
                        help="Original ScanQA annotation (with object_ids) joined on question_id. "
                             "Enables GT-retention stats; required for POOLING=3dzip_gt_oracle.")
    parser.add_argument("--scannet-gt-root", type=str, default=None,
                        help="Preprocessed scannet_data/ with <scene>_aligned_vert.npy and "
                             "<scene>_ins_label.npy (instance id = object_id + 1).")
    parser.add_argument("--scannet-raw-root", type=str, default=None,
                        help="Raw ScanNet scene dirs (<scene>.aggregation.json, segs.json, "
                             "*_vh_clean_2.ply, <scene>.txt). Instance id = object_id (no +1).")
    parser.add_argument("--gt-cache-dir", type=str, default=None,
                        help="Where to cache axis-aligned GT point clouds built from raw ScanNet.")
    parser.add_argument("--gt-nn-max-dist", type=float, default=None,
                        help="Max token-to-GT-vertex distance (m) for a trusted GT label "
                             "(default 0.10 / $GT_NN_MAX_DIST).")
    parser.add_argument("--gt-voxel-min-frac", type=float, default=None,
                        help="Min fraction of a voxel's points on a target object for the voxel to "
                             "count as GT-relevant (default 0.25 / $GT_VOXEL_MIN_FRAC).")
    parser.add_argument("--gt-max-anchor-frac", type=float, default=None,
                        help="Cap on the share of K the GT voxels may occupy (default 1.0 / "
                             "$GT_MAX_ANCHOR_FRAC = the plain oracle). 0.5 keeps half of the "
                             "budget for the stock diversity selection.")
    parser.add_argument("--gt-stats-file", type=str, default=None,
                        help="Per-question GT retention stats JSONL "
                             "(default: gt_oracle_stats.jsonl next to --answers-file).")
    parser.add_argument("--v0-gt-dump-dir", type=str, default=None,
                        help="POOLING=v0_gt: write each question's kept tokens and GT boxes as "
                             "<dir>/<scene>/<question_id>.npz for visualization/visualize_v0_gt.py.")
    parser.add_argument("--gt-oracle-debug", action="store_true",
                        help="Print the per-question [GT ORACLE] / [GT ALIGN CHECK] lines.")
    parser.add_argument("--one-question-per-scene", action="store_true",
                        help="Keep just the first question of each scene, so --limit N "
                             "visits N distinct scenes. Token compression ignores the "
                             "question text, so this covers N scenes at 1/66th the cost "
                             "of walking N scenes' worth of questions.")
    add_selection_export_args(parser)
    args = parser.parse_args()

    eval_model(args)
