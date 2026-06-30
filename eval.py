"""
Panoramic 360° Object Localization — Evaluation Script

Evaluates a fine-tuned Qwen3-VL model on 360-degree panoramic image object
localization. The model iteratively calls rotate_and_project_panorama tool
to narrow down azimuth/elevation coordinates.

Usage:
    python eval.py --model <HF_MODEL_ID_OR_LOCAL_PATH> \
                   --test_file eagle360_test/test.json \
                   --pano_dir eagle360_test \
                   [--n_samples 50] \
                   [--max_tokens 512]

Examples:
    # Use HuggingFace model (auto-download)
    python eval.py --model your-org/panoramic-360-grpo-qwen3vl-4b \
                   --pano_dir eagle360_test

    # Use local checkpoint
    python eval.py --model ./checkpoints/hf_merged \
                   --test_file eagle360_test/test.json \
                   --pano_dir eagle360_test \
                   --n_samples 50
"""
import os
import sys
import json
import re
import math
import argparse
import time
from datetime import datetime

os.environ["VLLM_USE_V1"] = "1"

import cv2
import numpy as np
from PIL import Image


# ── panoramic projection ──────────────────────────────────────────────────────

def rotate_panorama_and_crop(image, target_u, target_v,
                              target_height, target_width,
                              img_height, img_width):
    pano_height, pano_width, _ = image.shape

    def sph_to_xyz(u, v):
        x = np.cos(v) * np.sin(u)
        y = np.cos(v) * np.cos(u)
        z = -np.sin(v)
        return np.stack([x, y, z], axis=-1)

    y_axis = sph_to_xyz(target_u, target_v).flatten()
    y_axis /= np.linalg.norm(y_axis)
    z_g = np.array([0., 0., 1.])
    x_axis = np.cross(y_axis, z_g)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1., 0., 0.])
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)

    jj, ii = np.meshgrid(np.arange(img_width), np.arange(img_height))
    x_plane = (jj / img_width - 0.5) * 2 * np.tan(target_width / 2)
    y_plane = (ii / img_height - 0.5) * 2 * np.tan(target_height / 2)
    rays_local = np.stack([x_plane, np.ones_like(x_plane), -y_plane], axis=-1)
    rays_local /= np.linalg.norm(rays_local, axis=-1, keepdims=True)
    rays_global = (rays_local[..., 0:1] * x_axis +
                   rays_local[..., 1:2] * y_axis +
                   rays_local[..., 2:3] * z_axis)

    src_u = np.arctan2(rays_global[..., 0], rays_global[..., 1])
    src_v = -np.arcsin(np.clip(rays_global[..., 2], -1., 1.))
    map_x = ((src_u / (2 * np.pi)) + 0.5) * pano_width
    map_y = ((src_v / np.pi) + 0.5) * pano_height
    return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def project_panorama(pano_path, az_deg, el_deg, fov_deg=100, out_size=512):
    img = cv2.imread(pano_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    az_r = math.radians(az_deg)
    el_r = math.radians(el_deg)
    fov_r = math.radians(fov_deg)
    crop = rotate_panorama_and_crop(img, az_r, -el_r, fov_r, fov_r, out_size, out_size)
    return Image.fromarray(crop)


# ── system prompt & turn templates ───────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for panoramic image understanding.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
[
  {
    "type": "function",
    "function": {
      "name": "rotate_and_project_panorama",
      "description": "Rotate the panoramic image to center on a specific direction and project it to a perspective view with given field of view. You need to repeatedly call it. By adjusting the azimuth and elevation angles, as well as reducing the field of view, you should gradually move the center of the viewing angle closer to the target object until a precise positioning is achieved.",
      "parameters": {
        "properties": {
          "azimuth": {
            "type": "number",
            "description": "The azimuth angle in degrees, range [-180, 180]. 0 is forward, positive is right."
          },
          "elevation": {
            "type": "number",
            "description": "The elevation angle in degrees, range [-90, 90]. 0 is horizontal, positive is up."
          },
          "fov_degrees": {
            "type": "number",
            "description": "The field of view in degrees. Default is 100."
          }
        },
        "required": ["azimuth", "elevation"],
        "type": "object"
      },
      "args_format": "Format the arguments as a JSON object."
    }
  }
]
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Important: Tool Calling Rules
1. If you decide to call a tool, output ONLY the tool call and STOP immediately.
2. DO NOT provide the final answer in the same response as the tool call.
3. The tool will return a projected image, and you should wait for it before giving your answer.
4. Do not repeatedly call tools with the same parameters.

# Response Format
- If you need to use a tool: <think>...</think> <tool_call>...</tool_call> (STOP HERE, wait for tool result)
- After receiving the tool result and ready to answer: <think>...</think> <answer>...</answer>

You MUST provide an answer within the prescribed MAX_TURN of conversations! The MAX_TURN of conversations is 6.

NEVER combine <tool_call> and <answer> in the same response unless you have already received the projected image!"""

CONTINUATION_PROMPT_TEMPLATE = (
    "Turn {turn} : Think step by step, and then decide whether to call tools OR provide "
    "final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
    "(if tool needed) OR <answer>...</answer> (if ready to answer or the conversation "
    "has reached the MAX_TURN = 6). Do not repeatedly call tools with the same parameters."
)
FINAL_TURN_PROMPT_TEMPLATE = (
    "Turn {turn} [FINAL TURN - {turn}/6]: You have reached the maximum number of turns. "
    "You MUST provide your final answer RIGHT NOW. Do NOT call any more tools. "
    "Respond ONLY with: <think>your reasoning</think> <answer>The object is located at:\n"
    "- Azimuth: X°\n- Elevation: Y°</answer>"
)

SFT_BFOV_BASELINE = 43.0


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_angles_from_answer(text, strict_mode=True):
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if strict_mode and not answer_match:
        return None, None
    body = answer_match.group(1) if answer_match else text
    az = el = None
    for pat in [r'azimuth[:\s=]+(-?\d+\.?\d*)\s*(?:degrees?|°)?',
                r'-\s*azimuth[:\s=]+(-?\d+\.?\d*)\s*(?:degrees?|°)?']:
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if -180 <= v <= 180:
                    az = v
                    break
            except Exception:
                pass
    for pat in [r'elevation[:\s=]+(-?\d+\.?\d*)\s*(?:degrees?|°)?',
                r'-\s*elevation[:\s=]+(-?\d+\.?\d*)\s*(?:degrees?|°)?']:
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if -90 <= v <= 90:
                    el = v
                    break
            except Exception:
                pass
    return az, el


def parse_tool_call(text):
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
        args = d.get('arguments', d)
        return {
            'azimuth':     float(args.get('azimuth', 0)),
            'elevation':   float(args.get('elevation', 0)),
            'fov_degrees': float(args.get('fov_degrees', 100)),
        }
    except Exception:
        return None


def detect_answer(text):    return '<answer>' in text and '</answer>' in text
def detect_tool_call(text): return '<tool_call>' in text and '</tool_call>' in text


def great_circle_dist(az1, el1, az2, el2):
    a1, e1 = math.radians(az1), math.radians(el1)
    a2, e2 = math.radians(az2), math.radians(el2)
    c = math.sin(e1) * math.sin(e2) + math.cos(e1) * math.cos(e2) * math.cos(a1 - a2)
    return math.degrees(math.acos(max(-1., min(1., c))))


def is_in_bfov(pred_az, pred_el, gt_az, gt_el, bbox_diag):
    return great_circle_dist(pred_az, pred_el, gt_az, gt_el) <= bbox_diag / 2


def determine_face(az_deg, el_deg):
    if el_deg >= 45:   return 'top'
    if el_deg <= -45:  return 'bottom'
    if -45 <= az_deg < 45:    return 'front'
    if 45  <= az_deg < 135:   return 'right'
    if az_deg >= 135 or az_deg < -135: return 'back'
    return 'left'


def check_tag_valid(text, tag, min_len=5):
    if tag == 'think':
        return f'<{tag}>' in text or f'</{tag}>' in text
    m = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return m is not None and len(m.group(1).strip()) > min_len


# ── vLLM input builder ────────────────────────────────────────────────────────

def build_vllm_input(processor, messages, pano_pil, proj_pils):
    PANO_PAD_ID = 151669
    IMG_PAD_ID  = 151655
    VISION_PADS = {PANO_PAD_ID, IMG_PAD_ID}
    out = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt")
    ids = out["input_ids"][0].tolist()
    deduped, prev = [], None
    for tok in ids:
        if tok in VISION_PADS and tok == prev:
            continue
        deduped.append(tok)
        prev = tok
    mm_data = {"panoramic_image": [pano_pil]}
    if proj_pils:
        mm_data["image"] = proj_pils
    return {"prompt_token_ids": deduped, "multi_modal_data": mm_data}


# ── main evaluation ───────────────────────────────────────────────────────────

def resolve_pano_path(raw_pano, pano_dir):
    if os.path.isabs(raw_pano) and os.path.exists(raw_pano):
        return raw_pano

    candidates = [
        os.path.join(pano_dir, raw_pano),
        os.path.join(pano_dir, os.path.basename(raw_pano)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def get_question(item):
    if item.get("question"):
        return item["question"].replace("<panoramic_image>", "").strip()

    for conv in item.get("conversations", []):
        if conv.get("from") == "human":
            return conv.get("value", "").replace("<panoramic_image>", "").strip()
    return None

def run_eval(model_path, test_file, pano_dir, n_samples=None, max_tokens=512):
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"\n{'='*70}")
    print(f"  Panoramic 360 Object Localization — Evaluation")
    print(f"  model    : {model_path}")
    print(f"  test_file: {test_file}")
    print(f"  pano_dir : {pano_dir}")
    print(f"  samples  : {n_samples or 'all'}")
    print(f"{'='*70}\n")

    llm = LLM(
        model=model_path,
        max_model_len=25000,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 10, "panoramic_image": 1},
    )
    processor = AutoProcessor.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    try:
        from tqdm import tqdm
        _tqdm_ok = True
    except ImportError:
        _tqdm_ok = False

    with open(test_file) as f:
        data = json.load(f)
    if n_samples:
        data = data[:n_samples]
    total = len(data)
    print(f"Loaded {total} test samples from {test_file}\n", flush=True)

    bfov_correct  = 0
    dist_correct  = 0
    failed        = 0
    total_turns   = 0
    total_tc      = 0
    distances     = []
    MAX_TOOL_CALLS = 6

    face_stats = {f: {'total': 0, 'correct': 0}
                  for f in ['front', 'back', 'left', 'right', 'top', 'bottom']}
    distance_bins = {b: 0 for b in range(0, 180, 10)}

    fmt_total = fmt_ok = fmt_no_think = fmt_truncated = 0

    step_distances     = {k: [] for k in range(1, 9)}
    step_rel_distances = {k: [] for k in range(2, 9)}
    answer_step_counts       = {k: 0 for k in range(1, 9)}
    answer_step_bfov_correct = {k: 0 for k in range(1, 9)}

    all_results = []
    t0 = time.time()
    pbar = tqdm(total=total, desc="Evaluating", unit="sample",
                dynamic_ncols=True) if _tqdm_ok else None

    for idx, item in enumerate(data):
        # Resolve panoramic image path for both old absolute-path data and the
        # new Eagle360 release where paths are relative to the dataset root.
        raw_pano = item["panoramic_image"]
        pano_path = resolve_pano_path(raw_pano, pano_dir)

        gt_az  = item["metadata"]["azimuth_degrees"]
        gt_el  = item["metadata"]["elevation_degrees"]
        _bw = item["metadata"].get("bbox_width_degrees", 0.0)
        _bh = item["metadata"].get("bbox_height_degrees", 0.0)
        bbox_diag = (item["metadata"].get("bbox_diagonal_degrees")
                     or (math.sqrt(_bw**2 + _bh**2) if (_bw or _bh) else 30.0))

        question = get_question(item)
        if not question:
            print(f"  [{idx+1:3d}] FORMAT ERROR — skip", flush=True)
            failed += 1
            if pbar: pbar.update(1)
            continue
        if not os.path.exists(pano_path):
            print(f"  [{idx+1:3d}] MISSING IMAGE {pano_path} — skip", flush=True)
            failed += 1
            if pbar: pbar.update(1)
            continue

        face = determine_face(gt_az, gt_el)
        face_stats[face]['total'] += 1

        if pbar:
            pbar.set_postfix({
                "bFOV": f"{bfov_correct/(idx+1)*100:.1f}%" if idx > 0 else "--",
                "ans":  f"{(idx-failed)/(idx+1)*100:.0f}%" if idx > 0 else "--",
                "fail": failed,
            }, refresh=True)

        pano_pil = Image.open(pano_path).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [
                {"type": "panoramic_image", "panoramic_image": pano_pil},
                {"type": "text", "text": question},
            ]},
        ]

        all_responses   = []
        proj_pils       = []
        tool_call_count = 0
        final_answer    = None
        sample_step_dists = []

        for turn in range(MAX_TOOL_CALLS + 1):
            vllm_in  = build_vllm_input(processor, messages, pano_pil, proj_pils)
            out      = llm.generate([vllm_in], sampling_params)
            response = out[0].outputs[0].text.strip()
            all_responses.append(response)

            is_last_resp = detect_answer(response) or (turn == MAX_TOOL_CALLS)
            fmt_total += 1
            ht  = check_tag_valid(response, 'think')
            ha  = check_tag_valid(response, 'answer')
            htc = check_tag_valid(response, 'tool_call')
            is_trunc = '<think>' in response and '</think>' not in response
            if not ht:   fmt_no_think  += 1
            if is_trunc: fmt_truncated += 1
            if is_last_resp:
                if ht and ha and not htc: fmt_ok += 1
            else:
                if ht and htc and not ha: fmt_ok += 1

            tc_az = re.search(r'"azimuth"\s*:\s*(-?\d+\.?\d*)', response)
            tc_el = re.search(r'"elevation"\s*:\s*(-?\d+\.?\d*)', response)
            snippet = response.replace('\n', ' ')[:80]
            if tc_az:
                tag = f" → az={tc_az.group(1)}° el={tc_el.group(1) if tc_el else '?'}°"
            elif detect_answer(response):
                tag = " → ANSWER"
            else:
                tag = " → no-tag"
            print(f"    t{turn+1}: {snippet}{tag}", flush=True)

            messages.append({"role": "assistant",
                              "content": [{"type": "text", "text": response}]})

            if detect_answer(response):
                final_answer = response
                break

            if detect_tool_call(response):
                if tool_call_count >= MAX_TOOL_CALLS:
                    break
                params = parse_tool_call(response)
                if params is None:
                    break

                proj_pil = project_panorama(pano_path,
                                            params['azimuth'],
                                            params['elevation'],
                                            params['fov_degrees'])
                tool_call_count += 1
                sample_step_dists.append(
                    great_circle_dist(gt_az, gt_el,
                                      params['azimuth'], params['elevation']))

                for msg in messages:
                    if msg["role"] == "user" and msg is not messages[1]:
                        new_content = []
                        for c in msg["content"]:
                            if c.get("type") == "image":
                                new_content.append({"type": "text",
                                                    "text": "<project_image>"})
                            else:
                                new_content.append(c)
                        msg["content"] = new_content
                proj_pils = [proj_pil]

                next_is_final = (tool_call_count >= MAX_TOOL_CALLS)
                cont = (FINAL_TURN_PROMPT_TEMPLATE if next_is_final
                        else CONTINUATION_PROMPT_TEMPLATE).format(turn=tool_call_count)
                messages.append({"role": "user", "content": [
                    {"type": "image", "image": proj_pil},
                    {"type": "text", "text": "\n" + cont},
                ]})
            else:
                break

        total_turns += len(all_responses)
        total_tc    += tool_call_count

        pred_az, pred_el = None, None
        if final_answer:
            pred_az, pred_el = extract_angles_from_answer(final_answer)
        if pred_az is None:
            for resp in reversed(all_responses):
                pred_az, pred_el = extract_angles_from_answer(resp)
                if pred_az is not None:
                    break

        if pred_az is not None and pred_el is not None:
            dist    = great_circle_dist(gt_az, gt_el, pred_az, pred_el)
            in_bfov = is_in_bfov(pred_az, pred_el, gt_az, gt_el, bbox_diag)
            distances.append(dist)
            bin_key = min(int(dist // 10) * 10, 170)
            distance_bins[bin_key] += 1
            if in_bfov:
                bfov_correct += 1
                face_stats[face]['correct'] += 1
            if dist < 50:
                dist_correct += 1
            print(f"  [{idx+1:3d}/{total}] face={face:6s} "
                  f"gt=({gt_az:.1f},{gt_el:.1f}) "
                  f"pred=({pred_az:.1f},{pred_el:.1f}) "
                  f"dist={dist:.1f}° turns={len(all_responses)} "
                  f"bfov={'✓' if in_bfov else '✗'}", flush=True)
            all_results.append({
                'idx': idx + 1, 'face': face,
                'gt_az': gt_az, 'gt_el': gt_el,
                'pred_az': pred_az, 'pred_el': pred_el,
                'dist': dist, 'in_bfov': in_bfov,
                'turns': len(all_responses),
                'tool_calls': tool_call_count,
            })
        else:
            failed += 1
            distances.append(180.0)
            distance_bins[170] += 1
            print(f"  [{idx+1:3d}/{total}] face={face:6s} NO ANSWER "
                  f"gt=({gt_az:.1f},{gt_el:.1f}) "
                  f"turns={len(all_responses)}", flush=True)
            all_results.append({
                'idx': idx + 1, 'face': face,
                'gt_az': gt_az, 'gt_el': gt_el,
                'pred_az': None, 'pred_el': None,
                'dist': None, 'in_bfov': False,
                'turns': len(all_responses),
                'tool_calls': tool_call_count,
            })

        for k, d in enumerate(sample_step_dists):
            if k + 1 <= 8:
                step_distances[k + 1].append(d)
        final_step = min(len(all_responses), 8)
        if pred_az is not None and pred_el is not None:
            _fd = great_circle_dist(gt_az, gt_el, pred_az, pred_el)
            step_distances[final_step].append(_fd)
            all_s = sample_step_dists + [_fd]
            for k in range(1, len(all_s)):
                step_rel_distances[min(k + 1, 8)].append(all_s[k] - all_s[k - 1])
        if final_answer is not None and final_step >= 1:
            answer_step_counts[final_step] += 1
            if (pred_az is not None and pred_el is not None and
                    is_in_bfov(pred_az, pred_el, gt_az, gt_el, bbox_diag)):
                answer_step_bfov_correct[final_step] += 1

        if pbar:
            pbar.update(1)
            pbar.set_postfix({
                "bFOV": f"{bfov_correct/(idx+1)*100:.1f}%",
                "ans":  f"{(idx+1-failed)/(idx+1)*100:.0f}%",
                "fail": failed,
            }, refresh=True)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            spd = (idx + 1) / elapsed
            eta = (total - idx - 1) / spd if spd > 0 else 0
            print(f"\n--- [{idx+1}/{total}]  "
                  f"bFOV={bfov_correct}/{idx+1}={bfov_correct/(idx+1)*100:.1f}%  "
                  f"dist<50={dist_correct}/{idx+1}={dist_correct/(idx+1)*100:.1f}%  "
                  f"avg_dist={sum(distances)/len(distances):.1f}°  "
                  f"no_ans={failed}  ETA={eta/60:.1f}min ---\n", flush=True)

    # ── final summary ─────────────────────────────────────────────────────────
    if pbar: pbar.close()
    elapsed_total = time.time() - t0

    bfov_acc  = bfov_correct / total * 100 if total else 0
    dist_acc  = dist_correct / total * 100 if total else 0
    ans_rate  = (total - failed) / total * 100 if total else 0
    avg_turns = total_turns / total if total else 0
    avg_tc    = total_tc    / total if total else 0
    avg_dist  = sum(distances) / len(distances) if distances else 0

    fmt_ok_rate    = fmt_ok         / fmt_total * 100 if fmt_total else 0
    fmt_think_rate = (fmt_total - fmt_no_think) / fmt_total * 100 if fmt_total else 0
    fmt_trunc_rate = fmt_truncated  / fmt_total * 100 if fmt_total else 0

    delta = bfov_acc - SFT_BFOV_BASELINE
    sign  = '+' if delta >= 0 else ''

    W = 72
    print(f"\n{'='*W}")
    print(f"  Evaluation Results  ({total} samples)")
    print(f"{'='*W}")
    print(f"  bFOV accuracy : {bfov_correct}/{total} = {bfov_acc:.2f}%"
          f"  (SFT baseline {SFT_BFOV_BASELINE:.0f}%,  delta {sign}{delta:.1f}%)")
    print(f"  dist<50°      : {dist_correct}/{total} = {dist_acc:.2f}%")
    print(f"  avg dist      : {avg_dist:.2f}°")
    print(f"  answer rate   : {total-failed}/{total} = {ans_rate:.2f}%")
    print(f"  avg turns     : {avg_turns:.2f}")
    print(f"  avg tool calls: {avg_tc:.2f}")
    print(f"  elapsed time  : {elapsed_total/60:.1f} min")
    print(f"{'='*W}")

    print(f"\nFormat check ({fmt_total} turns):")
    print(f"  format_ok   : {fmt_ok}/{fmt_total} = {fmt_ok_rate:.1f}%")
    print(f"  has_think   : {fmt_total-fmt_no_think}/{fmt_total} = {fmt_think_rate:.1f}%")
    print(f"  truncated   : {fmt_truncated}/{fmt_total} = {fmt_trunc_rate:.1f}%")
    print(f"{'='*W}")

    max_step_seen = max((k for k, v in step_distances.items() if v), default=0)
    print(f"\nPer-step average great-circle distance:")
    print(f"{'='*W}")
    for step in range(1, max_step_seen + 1):
        dists = step_distances[step]
        if dists:
            print(f"  Step {step}: avg_dist = {sum(dists)/len(dists):.2f}°  (n={len(dists)})")

    print(f"\nPer-step relative distance (negative = approaching target):")
    print(f"{'='*W}")
    for step in range(2, max_step_seen + 1):
        rels = step_rel_distances[step]
        if rels:
            print(f"  Step {step} - Step {step-1}: avg_delta = {sum(rels)/len(rels):.2f}°  (n={len(rels)})")

    print(f"\nAnswer step distribution:")
    print(f"{'='*W}")
    for step in range(1, max_step_seen + 1):
        cnt = answer_step_counts[step]
        pct = cnt / total * 100 if total else 0
        bfov_c = answer_step_bfov_correct[step]
        bfov_pct = bfov_c / cnt * 100 if cnt > 0 else 0
        print(f"  Step {step}: {cnt}/{total} = {pct:.1f}%  bFOV={bfov_c}/{cnt} = {bfov_pct:.1f}%")
    print(f"{'='*W}")

    print(f"\nDistance distribution (10° bins):")
    print(f"{'='*W}")
    for b in range(0, 180, 10):
        cnt = distance_bins[b]
        pct = cnt / total * 100 if total else 0
        bar = '█' * int(pct / 2)
        print(f"  {b:3d}°-{b+10:3d}°: {cnt:4d} ({pct:5.1f}%) {bar}")
    print(f"{'='*W}")

    print(f"\nPer-face accuracy (bFOV):")
    print(f"{'='*W}")
    for face_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        s = face_stats[face_name]
        acc = s['correct'] / s['total'] * 100 if s['total'] else 0
        bar = '█' * int(acc / 5)
        print(f"  {face_name:8s}: {s['correct']:4d}/{s['total']:4d} = {acc:6.2f}%  {bar}")
    print(f"{'='*W}\n")

    model_name = os.path.basename(model_path.rstrip('/'))
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file   = f"eval_{model_name}_{timestamp}.json"
    output_data = {
        "model_path":  model_path,
        "timestamp":   timestamp,
        "n_samples":   total,
        "summary": {
            "bfov_accuracy":   bfov_acc,
            "dist50_accuracy": dist_acc,
            "avg_distance":    avg_dist,
            "answer_rate":     ans_rate,
            "avg_turns":       avg_turns,
            "avg_tool_calls":  avg_tc,
            "failed":          failed,
            "elapsed_min":     round(elapsed_total / 60, 2),
        },
        "format_check": {
            "fmt_ok":    fmt_ok_rate,
            "has_think": fmt_think_rate,
            "truncated": fmt_trunc_rate,
        },
        "distance_distribution": {f"{b}-{b+10}": distance_bins[b] for b in range(0, 180, 10)},
        "face_statistics": {
            fn: {
                "total":    face_stats[fn]['total'],
                "correct":  face_stats[fn]['correct'],
                "accuracy": (face_stats[fn]['correct'] / face_stats[fn]['total'] * 100
                             if face_stats[fn]['total'] else 0),
            }
            for fn in face_stats
        },
        "detailed_results": all_results,
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_file}\n")

    return bfov_acc


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3-VL model on panoramic 360° object localization.")
    parser.add_argument(
        "--model", type=str,
        default="your-org/panoramic-360-grpo-qwen3vl-4b",
        help="HuggingFace model ID or local path to merged HF checkpoint")
    parser.add_argument(
        "--test_file", type=str,
        default="eagle360_test/test.json",
        help="Path to test JSON file (default: eagle360_test/test.json)")
    parser.add_argument(
        "--pano_dir", type=str,
        default="eagle360_test",
        help="Dataset root or directory containing panoramic JPEG images (default: eagle360_test)")
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Number of test samples to evaluate (default: all)")
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Max new tokens per turn (default: 512)")
    args = parser.parse_args()

    if not os.path.isdir(args.pano_dir):
        print(f"ERROR: pano_dir not found: {args.pano_dir}")
        sys.exit(1)
    if not os.path.isfile(args.test_file):
        print(f"ERROR: test_file not found: {args.test_file}")
        sys.exit(1)

    acc = run_eval(
        model_path=args.model,
        test_file=args.test_file,
        pano_dir=args.pano_dir,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
    )
    print(f"Final bFOV accuracy: {acc:.1f}%")
