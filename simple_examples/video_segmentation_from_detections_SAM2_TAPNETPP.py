#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
══════════════════════════════════════════════════════════════════════
  REAL-TIME MULTI-OBJECT SEGMENTATION & TRACKING  (v2)
  Detector : SAM 3.1  (text-prompted, adaptive frequency)
  Tracker  : SAM 2.1 Tiny (per-object memory, high frequency)
══════════════════════════════════════════════════════════════════════
Features:
  - Non-blocking prompt editing (type anytime, Enter to apply)
  - Re-identification: purged objects reclaim their old ID
  - Intra-label duplicate merging (anti-drift)
  - Memory pruning per object (bounded memory, no stale drift)
  - Semantic Heartbeat: Ultra-fast semantic validation
══════════════════════════════════════════════════════════════════════
Controls:
  - Type in the terminal at ANY time to change prompts (semicolon-separated)
  - Press 'q' or Esc in the OpenCV window to quit
══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import threading
import queue
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d


# --- Import hack ---
try:
    import muggled_sam
except ModuleNotFoundError:
    parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        sys.path.insert(0, os.path.abspath("."))

from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
from muggled_sam.demo_helpers.bounding_boxes import get_2box_iou
from simple_examples.utils import kabsch

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import tracker_certainty

# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_SOURCE = 0
DETECTION_MODEL_PATH = "/home/flanthier/.cache/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"
TRACKING_MODEL_PATH  = "/home/flanthier/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"

# Detection
# =====================================================================
# RealSense CONFIGURATION
# =====================================================================
USE_REALSENSE = True # Switch to False to use the standard webcam

if USE_REALSENSE:
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("Error: pyrealsense2 is not installed. Please run: pip install pyrealsense2")
        sys.exit(1)
DETECTION_SCORE_THRESHOLD  = 0.40
EXISTING_BOX_IOU_THRESHOLD = 0.25

# Tracking health
REMOVE_AFTER_N_MISSED = 8

# Adaptive detection cadence (AJUSTÉ POUR FAIBLE LATENCE)
DET_INTERVAL_STABLE  = 0.15    # Intervalle max de SAM3. Doit être inférieur au timeout.
DET_INTERVAL_HUNTING = 0.05   # missing targets   → hunt

# Validation sémantique : Temps max (en secondes) avant d'abandonner un objet non-confirmé par SAM 3
SEMANTIC_TIMEOUT = 0.6

# Memory pruning: max stored frames per object (keeps tracking fast + prevents stale drift)
MAX_MEMORY_FRAMES = 6

# Re-identification: how long (seconds) we remember a purged object for re-ID
REID_MEMORY_DURATION = 15.0

# Anti-drift: if two masks of the SAME label overlap more than this, merge them
SAME_LABEL_MERGE_IOU = 0.20

# Palette (BGR)
COLORS = [
    (0, 0, 255),     # Red
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (0, 165, 255),   # Orange
    (200, 100, 255), # Purple
    (100, 255, 200), # Mint
    (50, 50, 255),   # Dark red
]

# Modèle DA3 Métrique pour avoir les distances réelles (en mètres)
DA3_MODEL_PATH = "depth-anything/da3metric-large"
# Vos vrais paramètres intrinsèques (Camera Matrix)
INTRINSICS = {
    'fx': 631.697237, 'fy': 637.581120,
    'cx': 331.492407, 'cy': 232.271983
}

# Matrice formatée pour OpenCV
CAMERA_MATRIX = np.array([
    [INTRINSICS['fx'], 0.0, INTRINSICS['cx']],
    [0.0, INTRINSICS['fy'], INTRINSICS['cy']],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Vos coefficients de distorsion lenticulaire
DISTORTION = np.array([0.030101, -0.124685, -0.007313, 0.004155, 0.000000], dtype=np.float32)

# Mise à jour pour Depth Anything 3 (Moyenne de fx et fy)
FOCAL_LENGTH = (INTRINSICS['fx'] + INTRINSICS['fy']) / 2.0  # ~634.64

# Chemin du modèle TAPNet++
TAPNEXTPP_MODEL_PATH = "/home/flanthier/Github/src/vision_processing/third_party/tapnet/checkpoints/tapnextpp_ckpt.pt"

# =====================================================================
# FPS COUNTER
# =====================================================================
class FPSCounter:
    def __init__(self, window=20):
        self._times, self._window, self._fps = [], window, 0.0
        self._lock = threading.Lock()

    def tick(self):
        with self._lock:
            self._times.append(time.time())
            if len(self._times) > self._window:
                self._times.pop(0)
            if len(self._times) > 1:
                self._fps = (len(self._times) - 1) / (self._times[-1] - self._times[0])

    @property
    def fps(self):
        with self._lock:
            return self._fps


# =====================================================================
# FAST WEBCAM (dedicated grab thread)
# =====================================================================
class FastWebcam:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self._lock = threading.Lock()
        self._stopped = False

    def start(self):
        threading.Thread(target=self._grab, daemon=True).start()
        return self

    def _grab(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return (self.ret, self.frame.copy(), None) if self.frame is not None else (False, None, None)

    def stop(self):
        self._stopped = True
        self.cap.release()

# =====================================================================
# FAST REALSENSE (dedicated grab thread for D435i)
# =====================================================================
class FastRealSense:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        profile = self.pipeline.start(config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        
        self._lock = threading.Lock()
        self._stopped = False
        self.ret = False
        self.frame = None
        self.depth = None
        
        # Grab first frame synchronously
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if color_frame and depth_frame:
            self.ret = True
            self.frame = np.asanyarray(color_frame.get_data())
            self.depth = np.asanyarray(depth_frame.get_data())

    def start(self):
        threading.Thread(target=self._grab, daemon=True).start()
        return self

    def _grab(self):
        while not self._stopped:
            try:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if color_frame and depth_frame:
                    c_frame = np.asanyarray(color_frame.get_data())
                    d_frame = np.asanyarray(depth_frame.get_data())
                    with self._lock:
                        self.ret = True
                        self.frame = c_frame
                        self.depth = d_frame
            except RuntimeError:
                pass

    def read(self):
        with self._lock:
            return (self.ret, self.frame.copy(), self.depth.copy()) if self.frame is not None else (False, None, None)

    def stop(self):
        self._stopped = True
        self.pipeline.stop()

# =====================================================================
# TRACKED OBJECT
# =====================================================================
class TrackedObject:
    def __init__(self, obj_id, label, color, memory, init_box_norm):
        self.obj_id = obj_id
        self.label = label
        self.color = color
        self.memory: SAMVideoObjectResults = memory
        self.missed = 0
        self.last_mask_np = None
        self.last_box_norm = init_box_norm
        self.last_score = 0.0
        self.stored_frame_count = 0
        self.last_semantically_verified = time.time()
        self.needs_tap_init = True
        self.last_refresh_time = 0.0
        self.mask_lowres_gpu = None


# =====================================================================
# RE-ID GHOST — remembers purged objects for later re-identification
# =====================================================================
class ReIDGhost:
    def __init__(self, obj_id, label, color, last_box_norm, purge_time):
        self.obj_id = obj_id
        self.label = label
        self.color = color
        self.last_box_norm = last_box_norm
        self.purge_time = purge_time


# =====================================================================
# NON-BLOCKING PROMPT INPUT THREAD
# =====================================================================
def stdin_listener(prompts_ref, shutdown):
    while not shutdown.is_set():
        try:
            line = input()
        except EOFError:
            break

        new_prompts = [p.strip() for p in line.split(";") if p.strip()]
        if new_prompts:
            with prompts_ref["lock"]:
                prompts_ref["prompts"] = new_prompts
                prompts_ref["det_interval"] = DET_INTERVAL_HUNTING  
                prompts_ref["changed"] = True
            print(f"[PROMPTS] Now tracking: {new_prompts}")


# =====================================================================
# THREAD 1 — DETECTOR  (SAM 3.1)
# =====================================================================
def detector_worker(detmodel, cam, prompts_ref, det_queue, shutdown, fps):
    print("[DETECTOR] Started.")
    enc_cfg = {"max_side_length": 512, "use_square_sizing": True}

    while not shutdown.is_set():
        ret, frame, depth = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        with prompts_ref["lock"]:
            prompts = list(prompts_ref["prompts"])

        if not prompts:
            time.sleep(0.1)
            continue

        try:
            det_enc, _, _ = detmodel.encode_detection_image(frame, **enc_cfg)

            for prompt_text in prompts:
                exemplars = detmodel.encode_exemplars(det_enc, text=prompt_text)
                masks, boxes, _, _ = detmodel.generate_detections(
                    det_enc, exemplars,
                    detection_filter_threshold=DETECTION_SCORE_THRESHOLD,
                )
                n = masks.shape[1] if masks is not None else 0
                if n > 0:
                    det_queue.put({
                        "masks": masks,
                        "boxes": boxes,
                        "source_frame": frame,
                        "prompt": prompt_text,
                    })
        except Exception as e:
            print(f"[DETECTOR] Error: {e}")

        fps.tick()

        with prompts_ref["lock"]:
            interval = prompts_ref.get("det_interval", DET_INTERVAL_STABLE)
        time.sleep(interval)


# =====================================================================
# HELPER: bounding box from binary mask
# =====================================================================
def box_from_mask(mask_np, shape_hw):
    # mask_np is bool or >0 already; project to get column/row extents
    cols = mask_np.any(axis=0)
    rows = mask_np.any(axis=1)
    if not cols.any():
        return None
    h, w = shape_hw
    xs = np.where(cols)[0]
    ys = np.where(rows)[0]
    return torch.tensor(
        [[xs[0] / w, ys[0] / h],
         [xs[-1] / w, ys[-1] / h]], dtype=torch.float32
    )


# =====================================================================
# THREAD 2 — TRACKER  (SAM 2.1 Tiny)
# =====================================================================
def tracker_worker(track_model, cam, det_queue, shared_state, prompts_ref, shutdown, fps):
    print("[TRACKER] Started.")
    enc_cfg = {"max_side_length": 512, "use_square_sizing": True}

    tracked: dict[int, TrackedObject] = {}
    ghosts: list[ReIDGhost] = []
    next_id = 0
    frame_idx = 0

    while not shutdown.is_set():
        ret, frame, depth = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

        with shared_state["lock"]:
            refresh_ids = shared_state.pop("refresh_ids", set())
        with prompts_ref["lock"]:
            if prompts_ref.get("changed", False):
                prompts_ref["changed"] = False
                tracked.clear()
                ghosts.clear()
                print("[TRACKER] Prompts changed → all objects cleared.")

        # ==========================================================
        # A) INGEST NEW DETECTIONS & REFRESH HEARTBEATS
        # ==========================================================
        while not det_queue.empty():
            det = det_queue.get()
            new_masks = det["masks"]
            new_boxes = det["boxes"]
            src_frame = det["source_frame"]
            prompt    = det["prompt"]
            n_det     = new_masks.shape[1]

            src_enc, _, _ = track_model.encode_image(src_frame, **enc_cfg)

            for i in range(n_det):
                new_box = new_boxes[0, i]

                # ---- VALIDATION SÉMANTIQUE (HEARTBEAT REFRESH) ----
                matched_obj = None
                for obj in tracked.values():
                    if obj.label == prompt and obj.last_box_norm is not None:
                        iou = get_2box_iou(new_box, obj.last_box_norm.to(new_boxes.device))
                        if iou > EXISTING_BOX_IOU_THRESHOLD:
                            matched_obj = obj
                            break

                if matched_obj is not None:
                    # L'objet existe déjà et SAM 3 confirme que c'est la bonne classe !
                    matched_obj.last_semantically_verified = time.time()
                    continue

                # ---- NOUVEL OBJET OU RE-IDENTIFICATION ----
                reused_ghost = None
                now = time.time()
                for ghost in ghosts:
                    if ghost.label != prompt:
                        continue
                    if (now - ghost.purge_time) > REID_MEMORY_DURATION:
                        continue
                    if ghost.last_box_norm is not None:
                        iou = get_2box_iou(new_box, ghost.last_box_norm.to(new_boxes.device))
                        same_region = iou > 0.05
                    else:
                        same_region = False

                    label_ghosts = [g for g in ghosts if g.label == prompt
                                    and (now - g.purge_time) < REID_MEMORY_DURATION]
                    if same_region or len(label_ghosts) == 1:
                        reused_ghost = ghost
                        break

                raw_mask = new_masks[0, i]
                init_mem, init_ptr = track_model.initialize_from_mask(src_enc, raw_mask > 0)
                obj_mem = SAMVideoObjectResults.create()
                obj_mem.store_prompt_result(frame_idx, init_mem, init_ptr)

                if reused_ghost is not None:
                    obj = TrackedObject(
                        reused_ghost.obj_id, prompt, reused_ghost.color,
                        obj_mem, new_box.cpu().float()
                    )
                    ghosts.remove(reused_ghost)
                    print(f"[TRACKER] Re-identified '{prompt}' → restored ID {obj.obj_id}")
                else:
                    color = COLORS[next_id % len(COLORS)]
                    obj = TrackedObject(next_id, prompt, color, obj_mem, new_box.cpu().float())
                    print(f"[TRACKER] New '{prompt}' → ID {next_id}")
                    next_id += 1

                tracked[obj.obj_id] = obj

        # ==========================================================
        # B) PROPAGATE TRACKING & PURGE TIMEOUTS
        # ==========================================================
        if tracked:
            enc_imgs, _, _ = track_model.encode_image(frame, **enc_cfg)
            dead_ids = []
            now = time.time()

            results = []
            for obj in tracked.values():
                score, best_idx, preds, mem_enc, obj_ptr = track_model.step_video_masking(
                    enc_imgs, **obj.memory.to_dict()
                )
                results.append((obj, score, best_idx, preds, mem_enc, obj_ptr))

            # Single sync for all scores
            scores_cpu = torch.stack([r[1] for r in results]).float().cpu().numpy()

            # Second pass: per-object logic uses scores_cpu[i]
            for i, (obj, _, best_idx, preds, mem_enc, obj_ptr) in enumerate(results):
                obj.last_score = float(scores_cpu[i])
                if obj.last_score < 0:
                    # Perte spatiale par SAM 2
                    obj.missed += 1
                    if obj.missed > REMOVE_AFTER_N_MISSED:
                        dead_ids.append(obj.obj_id)
                else:
                    # Vérification du Heartbeat sémantique
                    if now - obj.last_semantically_verified > SEMANTIC_TIMEOUT:
                        print(f"[TRACKER] ID {obj.obj_id} ('{obj.label}') purgé (Timeout sémantique).")
                        dead_ids.append(obj.obj_id)
                        continue  # On coupe l'herbe sous le pied de SAM 2, on ne stocke rien

                    obj.missed = 0
                    obj.memory.store_frame_result(frame_idx, mem_enc, obj_ptr)
                    obj.stored_frame_count += 1

                    if hasattr(obj.memory, '_frame_results'):
                        while len(obj.memory._frame_results) > MAX_MEMORY_FRAMES:
                            obj.memory._frame_results.pop(0)
                    elif hasattr(obj.memory, 'frame_results'):
                        while len(obj.memory.frame_results) > MAX_MEMORY_FRAMES:
                            obj.memory.frame_results.pop(0)

                    mask_lowres = preds[0, best_idx]                 # still on GPU, small
                    # Defer full-res resize: only do it if we're about to display or init TAP
                    # obj.mask_lowres_gpu = mask_lowres.detach()
                    if (obj.needs_tap_init or obj.obj_id in refresh_ids or
                        frame_idx % 2 == 0):                         # resize every other frame for display
                        mask_f = mask_lowres.cpu().float().numpy().squeeze()
                        mask_full = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
                        obj.last_mask_np = mask_full
                        obj.last_box_norm = box_from_mask(mask_full > 0, (h, w))

                    # --- INJECTION VERS TAPNET (MULTI-OBJETS & AUTO-REFRESH) ---
                    # On utilise getattr pour initialiser à True par défaut pour tout nouvel objet
                    REFRESH_COOLDOWN = 3.0   # seconds
                    now_t = time.time()
                    is_init = getattr(obj, "needs_tap_init", True)
                    is_refresh = (obj.obj_id in refresh_ids and
                                  now_t - obj.last_refresh_time >= REFRESH_COOLDOWN)

                    if is_init or is_refresh:
                        obj.last_refresh_time = now_t   # stamp it before we act
                        binary_mask = (mask_full > 0.0).astype(np.uint8)
                        kernel = np.ones((7, 7), np.uint8) 
                        safe_mask = cv2.erode(binary_mask, kernel, iterations=1)
                        
                        ys_m, xs_m = np.where(safe_mask > 0)
                        
                        if len(ys_m) > 0:
                            min_y, max_y = ys_m.min(), ys_m.max()
                            min_x, max_x = xs_m.min(), xs_m.max()
                            
                            step = max(6, min(max_y - min_y, max_x - min_x) // 15) 
                            grid_y, grid_x = np.mgrid[min_y:max_y:step, min_x:max_x:step]
                            grid_pts = np.vstack((grid_y.ravel(), grid_x.ravel())).T
                            
                            valid_3d_candidates = []
                            valid_2d_candidates = []
                            
                            for (y_int, x_int) in grid_pts:
                                if safe_mask[y_int, x_int] > 0: 
                                    z_val = 0.0
                                    if depth is not None:
                                        depth_scale = getattr(cam, 'depth_scale', 0.001)
                                        window = depth[max(0, y_int-2):min(h, y_int+3), max(0, x_int-2):min(w, x_int+3)]
                                        valid_depths = window[window > 0] * depth_scale
                                        
                                        if len(valid_depths) > 10: 
                                            z_val = np.median(valid_depths)
                                            
                                    if z_val > 0.1:
                                        X = (x_int - INTRINSICS['cx']) * z_val / INTRINSICS['fx']
                                        Y = (y_int - INTRINSICS['cy']) * z_val / INTRINSICS['fy']
                                        valid_3d_candidates.append([X, Y, z_val])
                                        valid_2d_candidates.append([y_int / h, x_int / w])
                            
                            target_pts = 150 
                            if len(valid_2d_candidates) > target_pts:
                                indices = np.random.choice(len(valid_2d_candidates), size=target_pts, replace=False)
                                final_pts_2d = [valid_2d_candidates[i] for i in indices]
                                selected_pts_3d = [valid_3d_candidates[i] for i in indices]
                            else:
                                final_pts_2d = valid_2d_candidates
                                selected_pts_3d = valid_3d_candidates
                                
                            if final_pts_2d:
                                n_with_3d = sum(1 for p in selected_pts_3d if np.linalg.norm(p) > 0)
                                if n_with_3d < 20:
                                    print(f"[TRACKER] init '{obj.label}' ID {obj.obj_id}: rejected "
                                        f"(only {n_with_3d} valid 3D points, need 20)")
                                    # don't set needs_tap_init = False — we'll retry next detection
                                    continue
                                obj.needs_tap_init = False
                                with shared_state["lock"]:
                                    if "new_tap_queries" not in shared_state:
                                        shared_state["new_tap_queries"] = []
                                    shared_state["new_tap_queries"].append({
                                        "id": obj.obj_id,
                                        "label": obj.label,
                                        "points": final_pts_2d,
                                        "points_3d": selected_pts_3d,
                                        "source_frame": frame.copy(), 
                                    })

            # Nettoyage des objets morts
            for did in dead_ids:
                if did in tracked:
                    dead_obj = tracked.pop(did)
                    ghosts.append(ReIDGhost(
                        dead_obj.obj_id, dead_obj.label, dead_obj.color,
                        dead_obj.last_box_norm, time.time()
                    ))

            ghosts[:] = [g for g in ghosts if (now - g.purge_time) < REID_MEMORY_DURATION]

        # ==========================================================
        # C) ANTI-DRIFT: MERGE DUPLICATE MASKS
        # ==========================================================
        labels_seen = {}
        for obj in list(tracked.values()):
            if obj.last_mask_np is None:
                continue
            if obj.label not in labels_seen:
                labels_seen[obj.label] = [obj]
            else:
                labels_seen[obj.label].append(obj)

        merge_kills = []
        for label, objs in labels_seen.items():
            if len(objs) < 2:
                continue
            for i in range(len(objs)):
                if objs[i].obj_id in merge_kills:
                    continue
                for j in range(i + 1, len(objs)):
                    if objs[j].obj_id in merge_kills:
                        continue
                    mask_a = objs[i].last_mask_np > 0
                    mask_b = objs[j].last_mask_np > 0
                    inter = np.logical_and(mask_a, mask_b).sum()
                    union = np.logical_or(mask_a, mask_b).sum()
                    iou = inter / union if union > 0 else 0

                    if iou > SAME_LABEL_MERGE_IOU:
                        if objs[i].last_score >= objs[j].last_score:
                            victim = objs[j]
                        else:
                            victim = objs[i]
                        merge_kills.append(victim.obj_id)

        for mid in merge_kills:
            if mid in tracked:
                tracked.pop(mid)

        # ==========================================================
        # D) PUBLISH FOR DISPLAY
        # ==========================================================
        display_list = []
        for obj in tracked.values():
            if obj.last_mask_np is not None:
                display_list.append({
                    "id":    obj.obj_id,
                    "label": obj.label,
                    "color": obj.color,
                    "mask":  obj.last_mask_np,
                    "score": obj.last_score,
                })

        with shared_state["lock"]:
            shared_state["objects"] = display_list

        with prompts_ref["lock"]:
            active_labels = {o.label for o in tracked.values()}
            all_found = all(p in active_labels for p in prompts_ref["prompts"])
            prompts_ref["det_interval"] = DET_INTERVAL_STABLE if all_found else DET_INTERVAL_HUNTING

        fps.tick()
        frame_idx += 1

# =====================================================================
# THREAD 3 — DEPTH (Depth Anything 3)
# =====================================================================
def depth_worker(da3_model, cam, shared_state, shutdown, fps):
    print("[DEPTH] Started.")
    
    while not shutdown.is_set():
        ret, frame, depth = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # On capture la vraie taille de la webcam (Hauteur, Largeur)
        original_h, original_w = frame.shape[:2]

        with torch.no_grad():
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            prediction = da3_model.inference([rgb_frame])
            raw_depth = prediction.depth[0]  # [H, W]
            # Sécurité : Si le modèle renvoie un Tensor PyTorch, on le passe en NumPy
            if hasattr(raw_depth, 'cpu'):
                raw_depth = raw_depth.cpu().numpy()

            # === LE FIX EST ICI ===
            # On étire la carte de profondeur pour qu'elle corresponde exactement 
            # à la taille de l'image de la webcam et du masque SAM 2 (Largeur, Hauteur)
            raw_depth_resized = cv2.resize(raw_depth, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            # Formule officielle DA3METRIC-LARGE pour convertir en mètres
            metric_depth = FOCAL_LENGTH * raw_depth_resized / 300.0

        with shared_state["lock"]:
            shared_state["depth_map"] = metric_depth

        fps.tick()

# =====================================================================
# THREAD 4 — TAPNET++ POINT TRACKING  (BATCHED, ONE FORWARD PASS / FRAME)
# =====================================================================
def tapnet_worker(tapnext_model, cam, shared_state, shutdown, fps):
    print("[TAPNET++] Started.")
    device = next(tapnext_model.parameters()).device
    dtype  = next(tapnext_model.parameters()).dtype

    TAPIR_SIZE = (256, 256)        # (W, H) used by cv2.resize
    MAX_TOTAL_POINTS = 1024         # global cap — matches the paper's demo

    # ---- per-object bookkeeping (all indexing into the global point dim) ----
    # obj_id -> {"start": int, "count": int, "label": str, "initial_3d": (n,3) np.float32}
    objects_meta = {}
    global_state = None              # the single recurrent state for *all* points
    last_tracks_2d = None            # (N_total, 2) in TAPIR coords, kept for re-init
    last_visible = None              # (N_total,) bool, for pruning
    need_reinit = True               # True whenever the set of points changed

    # pre-allocate a pinned CPU buffer for fast upload
    cpu_buf = torch.empty((TAPIR_SIZE[1], TAPIR_SIZE[0], 3),
                          dtype=torch.uint8, pin_memory=(device.type == "cuda"))

    def _encode_frame(frame_bgr):
        """BGR np.uint8 -> model-ready tensor [1,1,H,W,3] in model dtype, on device."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, TAPIR_SIZE)         # (H,W,3) uint8
        cpu_buf.copy_(torch.from_numpy(rgb_resized))      # into pinned CPU
        gpu_u8 = cpu_buf.to(device, non_blocking=True)    # async H2D copy
        # normalize to [-1, 1] on GPU in model dtype
        t = gpu_u8.to(dtype).div_(127.5).sub_(1.0)
        return t.unsqueeze(0).unsqueeze(0)                # [B=1, T=1, H, W, 3]


    def _rebuild_state(live_tensor_img):
        """
        Bootstrap new objects on their source frames (so moving-object init works),
        advance everyone to the live frame, then do ONE unifying re-init on the
        live frame. This avoids concatenating heterogeneous TAPNext states.
        """
        nonlocal global_state, last_tracks_2d, need_reinit

        if not objects_meta:
            global_state = None
            last_tracks_2d = None
            need_reinit = False
            return

        print(f"[TAPNET++] rebuilding state for {len(objects_meta)} objects, "
              f"total points = {sum(m['count'] for m in objects_meta.values())}")

        # Step 1: group by source frame, bootstrap each group to live frame
        groups = {}
        for obj_id, m in objects_meta.items():
            key = id(m["pending_src"])
            groups.setdefault(key, {"src": m["pending_src"], "items": []})["items"].append(obj_id)

        live_tracks_per_group = []   # tracks at the live frame, in group order
        ordered_ids = []             # object ids in the same order
        cursor = 0
        for g in groups.values():
            n_group = sum(objects_meta[oid]["pending_pts"].shape[0] for oid in g["items"])
            cq = torch.zeros((1, n_group, 3), device=device, dtype=dtype)
            sub_cursor = 0
            for oid in g["items"]:
                m = objects_meta[oid]
                pts = m.pop("pending_pts")
                _    = m.pop("pending_src")
                k = pts.shape[0]
                m["start"] = cursor + sub_cursor
                m["count"] = k
                cq[0, sub_cursor:sub_cursor+k, 0] = 0.0
                cq[0, sub_cursor:sub_cursor+k, 1] = torch.from_numpy(pts[:, 0]).to(device, dtype)
                cq[0, sub_cursor:sub_cursor+k, 2] = torch.from_numpy(pts[:, 1]).to(device, dtype)
                sub_cursor += k
                ordered_ids.append(oid)

            if g["src"] is live_tensor_img:
                tracks_live, _, _, _ = tapnext_model(g["src"], query_points=cq, state=None)
            else:
                _, _, _, g_state = tapnext_model(g["src"], query_points=cq, state=None)
                tracks_live, _, _, _ = tapnext_model(
                    live_tensor_img, query_points=None, state=g_state
                )
            live_tracks_per_group.append(tracks_live[0, 0])
            cursor += n_group

        # Step 2: unify everyone with ONE clean init on the live frame
        all_live_tracks = torch.cat(live_tracks_per_group, dim=0)   # [total, 2] (y, x)
        n_total = all_live_tracks.shape[0]
        cq_unified = torch.zeros((1, n_total, 3), device=device, dtype=dtype)
        cq_unified[0, :, 0] = 0.0
        cq_unified[0, :, 1] = all_live_tracks[:, 0]   # y
        cq_unified[0, :, 2] = all_live_tracks[:, 1]   # x

        tracks_final, _, _, global_state = tapnext_model(
            live_tensor_img, query_points=cq_unified, state=None
        )
        last_tracks_2d = tracks_final[0, 0].detach()
        need_reinit = False

    while not shutdown.is_set():
        ret, frame, depth = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        tensor_img = _encode_frame(frame)

        # ---- pull new queries / clear signals from the tracker thread ----
        with shared_state["lock"]:
            new_queries = shared_state.pop("new_tap_queries", None)
            clear_tap   = shared_state.pop("clear_tap", False)

        if clear_tap:
            objects_meta.clear()
            global_state = None
            last_tracks_2d = None
            need_reinit = True

        # ---- add or replace objects (this triggers a re-init) ----
        if new_queries:
            # Capture current tracked positions before we rebuild
            if last_tracks_2d is not None:
                last_np = last_tracks_2d.float().cpu().numpy()

            # Group incoming queries by source frame (they may come from different frames)
            # We key on id(source_frame) since arrays aren't hashable
            groups = {}
            for q in new_queries:
                key = id(q["source_frame"])
                groups.setdefault(key, {"frame": q["source_frame"], "queries": []})["queries"].append(q)

            # For each group: encode the *source* frame, init/refresh those objects on it
            for g in groups.values():
                src_tensor = _encode_frame(g["frame"])
                for q in g["queries"]:
                    pts = np.asarray(q["points"], dtype=np.float32)
                    if pts.size == 0:
                        continue
                    pts_tapir = np.stack([pts[:, 0] * TAPIR_SIZE[1],
                                        pts[:, 1] * TAPIR_SIZE[0]], axis=1)
                    pts_3d = np.asarray(q.get("points_3d", np.zeros((len(pts), 3))),
                                        dtype=np.float32)
                    objects_meta[q["id"]] = {
                        "label": q.get("label", str(q["id"])),
                        "initial_3d": pts_3d,
                        "pending_pts": pts_tapir,
                        "pending_src": src_tensor,    # <-- the frame these pts came from
                        "start": 0, "count": len(pts_tapir),
                    }

            # Keep surviving objects with their current tracked positions (anchored on live frame)
            for obj_id, m in list(objects_meta.items()):
                if "pending_pts" in m:
                    continue
                s, c = m["start"], m["count"]
                N = last_np.shape[0]
                s_clip = min(s, N)
                e_clip = min(s + c, N)
                pts_slice = last_np[s_clip:e_clip].copy()
                m["pending_pts"] = pts_slice
                m["pending_src"] = tensor_img
                m["initial_3d"] = m["initial_3d"][:pts_slice.shape[0]]
            need_reinit = True

        # ---- Nothing to track? Publish empty and move on. ----
        if not objects_meta and not need_reinit:
            with shared_state["lock"]:
                shared_state["tap_points"] = {}
                shared_state["kabsch_metrics"] = {}
            fps.tick()
            continue

        with torch.inference_mode():
            did_reinit = False
            if need_reinit:
                print(f"[TAPNET++] rebuilding state for {len(objects_meta)} objects, "
                      f"total points = {sum(m['count'] for m in objects_meta.values())}")
                _rebuild_state(tensor_img)
                did_reinit = True                           # ← add this

            if not objects_meta:
                with shared_state["lock"]:
                    shared_state["tap_points"] = {}
                    shared_state["kabsch_metrics"] = {}
                fps.tick()
                continue

            if did_reinit:
                # _rebuild_state already produced tracks for this frame
                tracks_2d = last_tracks_2d
                good = torch.ones(tracks_2d.shape[0], dtype=torch.bool, device=device)
            else:
                # THE ONE forward pass per frame
                tracks, track_logits, visible_logits, global_state = tapnext_model(
                    tensor_img, query_points=None, state=global_state
                )
                tracks_2d = tracks[0, 0]
                last_tracks_2d = tracks_2d.detach()
                cert = tracker_certainty(tracks, track_logits)[0, 0, :, 0]
                vis  = torch.sigmoid(visible_logits)[0, 0, :, 0]
                good = (cert > 0.7) & (vis > 0.5)

                # DEBUG: print per-object health after the forward pass
                for obj_id, m in objects_meta.items():
                    s, c = m["start"], m["count"]
                    c_mean = cert[s:s+c].float().mean().item()
                    v_mean = vis[s:s+c].float().mean().item()
                    g_count = good[s:s+c].sum().item()
                    print(f"  obj {obj_id}: cert={c_mean:.2f} vis={v_mean:.2f} good={g_count}/{c}")

            # Single GPU->CPU sync for everything we need this frame  (both branches)
            tracks_cpu = tracks_2d.float().cpu().numpy()
            good_cpu   = good.cpu().numpy()

        # ---- per-object: slice, rescale to camera px, solve PnP ----
        sx = w / TAPIR_SIZE[0]
        sy = h / TAPIR_SIZE[1]
        updated_tap_dict = {}
        kabsch_metrics = {}
        to_refresh_ids = set()
        to_delete_ids = []

        for obj_id, m in objects_meta.items():
            s, c = m["start"], m["count"]
            lbl  = m["label"]
            initial_3d = m["initial_3d"]

            obj_tracks = tracks_cpu[s:s+c]
            obj_good   = good_cpu[s:s+c]

            # Skip degenerate objects instead of crashing
            if obj_tracks.shape[0] == 0 or initial_3d.shape[0] != obj_tracks.shape[0]:
                m["pnp_fail_count"] = m.get("pnp_fail_count", 0) + 1
                continue

            # rescale (y, x) TAPIR -> (x, y) image px, vectorized
            cy = obj_tracks[:, 0] * sy
            cx = obj_tracks[:, 1] * sx
            inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            keep = obj_good & inside

            if np.any(keep):
                updated_tap_dict[obj_id] = list(zip(cx[keep].astype(int), cy[keep].astype(int)))
            
            # only points with a real 3D anchor are used for PnP
            has3d = np.linalg.norm(initial_3d, axis=1) > 0
            pnp_mask = keep & has3d
            visible_inliers = 0
            if pnp_mask.sum() >= 4:
                pts3d = initial_3d[pnp_mask].astype(np.float32)
                pts2d = np.stack([cx[pnp_mask], cy[pnp_mask]], axis=1).astype(np.float32)
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    pts3d, pts2d, CAMERA_MATRIX, DISTORTION,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                if ok:
                    visible_inliers = len(inl) if inl is not None else 0
                    R_np, _ = cv2.Rodrigues(rvec)
                    t_np = tvec.flatten()
                    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, CAMERA_MATRIX, DISTORTION)
                    proj = proj.reshape(-1, 2)
                    rmse = float(np.sqrt(np.mean(np.sum((pts2d - proj) ** 2, axis=1))))
                    full_cloud = initial_3d @ R_np.T + t_np
                    kabsch_metrics[obj_id] = {
                        "label": lbl,                    # keep the label as a field
                        "rmse": rmse,
                        "pts_count_visible": visible_inliers,
                        "pts_count_total": len(initial_3d),
                        "transform": {"R": R_np, "t": t_np},
                        "full_cloud": full_cloud,
                    }

            min_inliers = max(8, int(0.15 * m["count"]))
            if visible_inliers < min_inliers:
                print(f"[TAPNET++] obj {obj_id} ({m['label']}): "
                      f"{visible_inliers}/{m['count']} inliers, "
                      f"fail_count={m.get('pnp_fail_count', 0) + 1}")
                m["pnp_fail_count"] = m.get("pnp_fail_count", 0) + 1
                if m["pnp_fail_count"] % 5 == 0:
                    print(f"[TAPNET++] obj {obj_id} ({m['label']}): "
                          f"{visible_inliers}/{m['count']} inliers, "
                          f"fail_count={m['pnp_fail_count']}")
            else:
                m["pnp_fail_count"] = 0

        # ---- After the per-object loop: decide who to refresh ----
        PNP_FAIL_GRACE = 15     # ~1.5s at 10Hz
        to_delete_ids = []
        to_refresh_ids = set()
        for obj_id, m in list(objects_meta.items()):
            if m.get("pnp_fail_count", 0) >= PNP_FAIL_GRACE:
                to_refresh_ids.add(obj_id)
                to_delete_ids.append(obj_id)

        # ---- schedule refreshes / deletions; rebuild state next frame if needed ----
        if to_delete_ids:
            for oid in to_delete_ids:
                objects_meta.pop(oid, None)
            if objects_meta and last_tracks_2d is not None:
                last_np = last_tracks_2d.float().cpu().numpy()
                N = last_np.shape[0]
                for oid, m in objects_meta.items():
                    s, c = m["start"], m["count"]
                    s_clip = min(s, N)
                    e_clip = min(s + c, N)
                    pts_slice = last_np[s_clip:e_clip].copy()
                    m["pending_pts"] = pts_slice
                    m["pending_src"] = tensor_img
                    m["initial_3d"] = m["initial_3d"][:pts_slice.shape[0]]
                need_reinit = True
            else:
                global_state = None
                last_tracks_2d = None
                need_reinit = True

        if to_refresh_ids:
            with shared_state["lock"]:
                shared_state.setdefault("refresh_ids", set()).update(to_refresh_ids)


        with shared_state["lock"]:
            shared_state["tap_points"] = updated_tap_dict
            shared_state["kabsch_metrics"] = kabsch_metrics

        fps.tick()

# =====================================================================
# DISPLAY HELPERS
# =====================================================================
def draw_outlined_text(img, text, org, scale=0.6, fg=(255, 255, 255), bg=(0, 0, 0), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def overlay_masks(display, objects):
    for obj in objects:
        mask_bool = obj["mask"] > 0
        color_f = np.array(obj["color"], dtype=np.float32)
        display[mask_bool] = (
            0.6 * display[mask_bool].astype(np.float32) + 0.4 * color_f
        ).astype(np.uint8)

        ys, xs = np.where(mask_bool)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            
            # Affichage de l'ID, du Label ET de la distance en Z
            z_val = obj.get("z_dist", 0.0)
            tag = f"{obj['label']} #{obj['id']} | Z: {z_val:.2f}m"
            
            draw_outlined_text(display, tag, (cx - 30, cy),
                               scale=0.5, fg=obj["color"], thickness=1)


# =====================================================================
# MAIN
# =====================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    raw = input("Enter objects to track (semicolon-separated): ").strip()
    initial_prompts = [p.strip() for p in raw.split(";") if p.strip()]
    if not initial_prompts:
        print("No prompts given. Exiting.")
        return

    prompts_ref = {
        "prompts":      initial_prompts,
        "det_interval": DET_INTERVAL_HUNTING,
        "changed":      False,
        "lock":         threading.Lock(),
    }

    print("Loading SAM 3.1 (detector)...")
    _, sam3 = make_sam_from_state_dict(DETECTION_MODEL_PATH)
    sam3.to(device=device, dtype=dtype)
    detmodel = sam3.make_detector_model()

    print("Loading SAM 2.1 Tiny (tracker)...")
    _, sam2 = make_sam_from_state_dict(TRACKING_MODEL_PATH)
    sam2.to(device=device, dtype=dtype)
    print("Models ready.\n")

    # print("Loading Depth Anything 3 Metric...")
    # da3_model = DepthAnything3.from_pretrained(DA3_MODEL_PATH)
    # da3_model = da3_model.to(device=device)
    # da3_model.eval()
    # print("Models ready.\n")

    print("Loading TAPNet++ online tracker...")
    tapnext_model = TAPNext(image_size=(256, 256))
    # 1. Charger l'archive PyTorch Lightning
    checkpoint = torch.load(TAPNEXTPP_MODEL_PATH, map_location=device)
    
    # 2. Extraire le dictionnaire des poids
    raw_state_dict = checkpoint["state_dict"]
    
    # 3. Nettoyer le préfixe "model." ou "net." généré par Lightning
    clean_state_dict = {}
    for k, v in raw_state_dict.items():
        # On retire 'tapnext.' s'il est au début du nom
        if k.startswith('tapnext.'):
            clean_key = k[8:] # 8 est la longueur de "tapnext."
        else:
            clean_key = k.replace('model.', '').replace('net.', '')
            
        clean_state_dict[clean_key] = v
        
    # 4. Injecter les poids propres dans le modèle
    tapnext_model.load_state_dict(clean_state_dict)
    tapnext_model = tapnext_model.to(device=device, dtype=torch.float32)
    tapnext_model.eval()
    print("TAPNext++ Ready.\n")

    shared_state = {
        "objects": [], 
        # "depth_map": None,
        "tap_points": {}, # Dictionnaire {obj_id: [(x, y), (x, y)...]}
        "lock": threading.Lock()
    }
    det_queue = queue.Queue()
    shutdown = threading.Event()

    if USE_REALSENSE:
        cam = FastRealSense().start()
    else:
        cam = FastWebcam(VIDEO_SOURCE).start()
        
    time.sleep(0.5)

    fps_det = FPSCounter()
    fps_trk = FPSCounter()
    # fps_depth = FPSCounter()
    fps_tap = FPSCounter()
    threading.Thread(target=stdin_listener, args=(prompts_ref, shutdown), daemon=True).start()

    threading.Thread(
        target=detector_worker,
        args=(detmodel, cam, prompts_ref, det_queue, shutdown, fps_det),
        daemon=True,
    ).start()

    threading.Thread(
        target=tracker_worker,
        args=(sam2, cam, det_queue, shared_state, prompts_ref, shutdown, fps_trk),
        daemon=True,
    ).start()

    # threading.Thread(
    #     target=depth_worker,
    #     args=(da3_model, cam, shared_state, shutdown, fps_depth),
    #     daemon=True,
    # ).start()

    threading.Thread(
        target=tapnet_worker,
        args=(tapnext_model, cam, shared_state, shutdown, fps_tap),
        daemon=True,
    ).start()

    print(f"Tracking: {initial_prompts}")
    print("Type new prompts in the terminal at any time (semicolon-separated).")
    print("Press 'q' or Esc in the window to quit.\n")

    # ==========================================================
    # INITIALISATION DU VISUALISEUR 3D (Débogage Spatial Avancé)
    # ==========================================================
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Debug 3D - Pose & Occlusions", width=1000, height=800)
    
    # 1. Lire une frame factice pour obtenir la vraie résolution de ta caméra
    ret, test_frame, _ = cam.read()
    if ret:
        h_cam, w_cam = test_frame.shape[:2]
    else:
        h_cam, w_cam = 720, 1280 # Fallback de sécurité
        
    # 2. Calcul du VRAI champ de vision (FOV) mathématique
    Z_SCREEN = 0.5  # Profondeur du cadre virtuel (50 cm, la zone de manipulation)
    fx, fy = INTRINSICS['fx'], INTRINSICS['fy']
    cx, cy = INTRINSICS['cx'], INTRINSICS['cy']

    # Projections inverses pour trouver les bordures physiques de l'image à 50 cm
    x_min = (0 - cx) * Z_SCREEN / fx
    x_max = (w_cam - cx) * Z_SCREEN / fx
    y_min = (0 - cy) * Z_SCREEN / fy
    y_max = (h_cam - cy) * Z_SCREEN / fy

    # 3. Création des points de la Pyramide Focale
    cam_points = [
        [0, 0, 0], # Origine (La lentille physique de ta caméra)
        [x_min, y_min, Z_SCREEN], # Haut-Gauche
        [x_max, y_min, Z_SCREEN], # Haut-Droit
        [x_max, y_max, Z_SCREEN], # Bas-Droit
        [x_min, y_max, Z_SCREEN]  # Bas-Gauche
    ]
    
    # 4. Définition des Lignes et de leurs Couleurs
    cam_lines = [
        [0,1], [0,2], [0,3], [0,4], # Rayons de projection
        [1,2], [2,3], [3,4], [4,1]  # Le CADRE physique de l'écran
    ]
    
    # Couleurs : Les rayons sont gris discret, le cadre de l'écran est VERT FLUO
    cam_colors = [
        [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], [0.3, 0.3, 0.3], # Rayons
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]  # Cadre
    ]
    
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_points),
        lines=o3d.utility.Vector2iVector(cam_lines)
    )
    frustum.colors = o3d.utility.Vector3dVector(cam_colors)
    vis.add_geometry(frustum)
    
    # 5. AJOUT D'UNE GRILLE DE VISÉE (HUD) à l'intérieur du cadre
    grid_points = []
    grid_lines = []
    steps = 4 # Divise l'écran en un quadrillage 4x4
    
    idx = 0
    for i in range(1, steps):
        # Lignes verticales
        vx = x_min + (x_max - x_min) * (i / steps)
        grid_points.extend([[vx, y_min, Z_SCREEN], [vx, y_max, Z_SCREEN]])
        grid_lines.append([idx, idx + 1])
        idx += 2
        
        # Lignes horizontales
        hy = y_min + (y_max - y_min) * (i / steps)
        grid_points.extend([[x_min, hy, Z_SCREEN], [x_max, hy, Z_SCREEN]])
        grid_lines.append([idx, idx + 1])
        idx += 2
        
    grid_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_points),
        lines=o3d.utility.Vector2iVector(grid_lines)
    )
    # Grille en vert très sombre / translucide pour ne pas surcharger la vue
    grid_lineset.paint_uniform_color([0.0, 0.3, 0.0]) 
    vis.add_geometry(grid_lineset)

    # Repère visuel Global (X=Rouge, Y=Vert, Z=Bleu)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    
    # Dictionnaire pour stocker les nuages de points de chaque objet
    o3d_objects = {}
    
    # Configuration de la vue Isométrique
    view_control = vis.get_view_control()
    view_control.set_lookat([0.0, 0.0, Z_SCREEN]) 
    view_control.set_up([0.0, -1.0, 0.0]) 
    view_control.set_front([-0.6, -0.4, -0.8]) 
    view_control.set_zoom(0.95)
    # ==========================================================

    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    video_writer = None 
    next_write_time = 0.0
    print("Enregistrement vidéo prêt (Compatible PowerPoint). Le fichier sera finalisé à la fermeture.")
    try:
        while True:
            ret, frame, depth = cam.read()
            if not ret or frame is None:
                continue
            frame = cv2.undistort(frame, CAMERA_MATRIX, DISTORTION)
            display = frame.copy()

            with shared_state["lock"]:
                objects = list(shared_state["objects"])
                # Copie locale de la dernière depth_map pour éviter les conflits d'accès
                # current_depth_map = shared_state["depth_map"].copy() if shared_state["depth_map"] is not None else None

            # Calculer la profondeur médiane pour chaque objet traqué
            # if current_depth_map is not None:
            #     for obj in objects:
            #         mask_bool = obj["mask"] > 0
            #         if np.any(mask_bool):
            #             # Extraire uniquement les valeurs Z qui tombent DANS le masque de l'outil
            #             obj_depths = current_depth_map[mask_bool]
            #             # Utiliser la médiane pour filtrer le bruit aux bordures du masque
            #             median_z = np.median(obj_depths)
            #             obj["z_dist"] = median_z

            overlay_masks(display, objects)

            with shared_state["lock"]:
                tap_points = shared_state.get("tap_points", {})
                kabsch_metrics = shared_state.get("kabsch_metrics", {})

            # Dessiner les métriques Kabsch
            y_offset = 180
            for lbl, metrics in kabsch_metrics.items():
                # NOUVEAU TEXTE POUR MAIN()
                text = f"{lbl} PnP Reproj Error: {metrics['rmse']:.2f} px (Vis: {metrics['pts_count_visible']} / Total: {metrics['pts_count_total']})"
                draw_outlined_text(display, text, (10, y_offset), fg=(100, 200, 255))
                y_offset += 24
                
                for i, comp in enumerate(metrics.get("comparisons", [])):
                    gt = comp['gt']
                    cp = comp['computed']
                    pt_txt = f" P{i+1}: GT({gt[0]:.2f}, {gt[1]:.2f}, {gt[2]:.2f}) | Comp({cp[0]:.2f}, {cp[1]:.2f}, {cp[2]:.2f})"
                    draw_outlined_text(display, pt_txt, (10, y_offset), fg=(200, 255, 200), scale=0.4, thickness=1)
                    y_offset += 16
                y_offset += 8

            # Dessiner les points TAPNet++
            for obj_id, pts in tap_points.items():
                for pt in pts:
                    # Cercle jaune plein pour chaque point tracké par TAPNet
                    cv2.circle(display, pt, radius=3, color=(0, 255, 255), thickness=-1)
                    cv2.circle(display, pt, radius=4, color=(0, 0, 0), thickness=1)

            # N'oubliez pas d'afficher les FPS de TAPNet !
            
            draw_outlined_text(display, f"TAP: {fps_tap.fps:.1f} Hz", (10, 84), fg=(255, 255, 0))

            draw_outlined_text(display, f"Det: {fps_det.fps:.1f} Hz", (10, 28), fg=(0, 255, 255))
            draw_outlined_text(display, f"Trk: {fps_trk.fps:.1f} Hz", (10, 56), fg=(0, 255, 0))
            # draw_outlined_text(display, f"Dep: {fps_depth.fps:.1f} Hz", (10, 84+24), fg=(255, 100, 255))
            
            with prompts_ref["lock"]:
                prompt_str = " | ".join(prompts_ref["prompts"])
            draw_outlined_text(display, f"Prompts: {prompt_str}", (10, 84+24*2),
                               fg=(255, 200, 100), scale=0.5, thickness=1)
            draw_outlined_text(display, f"Objects: {len(objects)}", (10, 84+24*3),
                               fg=(255, 255, 255), scale=0.5, thickness=1)

            # 1. Récupérer les points en toute sécurité
            with shared_state["lock"]:
                current_tap_points = shared_state.get("tap_points", {})

            # 2. Dessiner les points sur l'image 2D
            for obj_id, pts in current_tap_points.items():
                for pt in pts:
                    # pt contient les coordonnées (x, y)
                    
                    # Un cercle jaune plein au centre
                    cv2.circle(
                        display, # Remplacez par le nom de votre variable d'image OpenCV
                        pt, 
                        radius=4, 
                        color=(0, 255, 255), # Jaune en BGR
                        thickness=-1
                    )
                    
                    # Un fin contour noir pour garantir la visibilité, 
                    # même si l'objet est clair
                    cv2.circle(
                        display, 
                        pt, 
                        radius=5, 
                        color=(0, 0, 0), 
                        thickness=1
                    )

            # === GÉNÉRATION DU NUAGE DE POINTS 3D ===
            # if current_depth_map is not None:
            #     h, w = current_depth_map.shape
                
            #     # 1. Convertir l'image et la profondeur au format natif C++ d'Open3D
            #     # On utilise 'display' pour avoir les masques colorés sur le modèle 3D !
            #     color_img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            #     o3d_color = o3d.geometry.Image(color_img)
            #     o3d_depth = o3d.geometry.Image(current_depth_map.astype(np.float32))
                
            #     # 2. Créer une image RGBD (Couleur + Profondeur)
            #     # depth_scale=1.0 car DA3 crache déjà des mètres.
            #     # depth_trunc=3.0 coupe l'affichage des murs à plus de 3 mètres (pour y voir clair)
            #     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            #         o3d_color, o3d_depth, 
            #         depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False
            #     )
                
            #     # 3. Paramètres de votre caméra calibrée
            #     intrinsic = o3d.camera.PinholeCameraIntrinsic(
            #         w, h, 
            #         INTRINSICS['fx'], INTRINSICS['fy'], 
            #         INTRINSICS['cx'], INTRINSICS['cy']
            #     )
                
            #     # 4. Projection mathématique instantanée
            #     temp_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                
            #     # 5. Mettre à jour les données du nuage principal
            #     pcd.points = temp_pcd.points
            #     pcd.colors = temp_pcd.colors
                
            #     if is_first_pcd:
            #         vis.add_geometry(pcd)
            #         is_first_pcd = False
            #     else:
            #         vis.update_geometry(pcd)
                
            #     # 6. Rafraîchir la fenêtre 3D
            #     vis.poll_events()
            #     vis.update_renderer()

            # === MISE À JOUR DE LA SCÈNE 3D ===
            with shared_state["lock"]:
                current_kabsch = shared_state.get("kabsch_metrics", {})
                
            for obj_id, metrics in current_kabsch.items():
                lbl = metrics["label"]
                full_cloud_np = metrics["full_cloud"]
                if obj_id not in o3d_objects:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(full_cloud_np)
                    pcd.paint_uniform_color([0.0, 1.0, 1.0])
                    o3d_objects[obj_id] = pcd              # key by obj_id here too
                    vis.add_geometry(pcd, reset_bounding_box=False)
                else:
                    o3d_objects[obj_id].points = o3d.utility.Vector3dVector(full_cloud_np)
                    vis.update_geometry(o3d_objects[obj_id])
            
            # Rafraîchissement de la fenêtre (très important pour que ça ne fige pas)
            # Rafraîchissement de la fenêtre 3D
            vis.poll_events()
            vis.update_renderer()
            
            # --- CAPTURE & FUSION POUR LA VIDÉO (3 PANNEAUX) ---
            # 1. Capture Open3D
            o3d_img = np.asarray(vis.capture_screen_float_buffer(do_render=False))
            o3d_img_uint8 = (o3d_img * 255).astype(np.uint8)
            o3d_img_bgr = cv2.cvtColor(o3d_img_uint8, cv2.COLOR_RGB2BGR)
            
            # Redimensionnement Open3D pour s'aligner sur la hauteur de la frame OpenCV
            h_cv, w_cv = display.shape[:2] # Remplacer 'nom_de_la_variable' par ta frame annotée
            h_o3d, w_o3d = o3d_img_bgr.shape[:2]
            ratio = h_cv / float(h_o3d)
            new_w_o3d = int(w_o3d * ratio)
            o3d_resized = cv2.resize(o3d_img_bgr, (new_w_o3d, h_cv))
            
            # 2. Préparation de la vraie Depth Map (Style RealSense Viewer)
            if depth is not None:
                # Créer un masque logique où True = pas de donnée (pixel noir)
                invalid_pixels = (depth == 0)
                
                # Normalisation standard de 0 à 255
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # Application de la carte thermique (Jet)
                depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                
                # === LA MAGIE REALSENSE ===
                # Forcer tous les pixels sans donnée en NOIR ABSOLU (BGR: 0, 0, 0)
                depth_color[invalid_pixels] = [0, 0, 0]
                
                depth_color_resized = cv2.resize(depth_color, (w_cv, h_cv))
            else:
                depth_color_resized = np.zeros((h_cv, w_cv, 3), dtype=np.uint8)
            
            # 3. Fusion horizontale : [RGB Annoté] + [Depth Colorisée] + [Modèle 3D]
            combined_frame = np.hstack((display, depth_color_resized, o3d_resized))
            
            # 4. Enregistrement
            if video_writer is None:
                final_h, final_w = combined_frame.shape[:2]
                # Format ultra-large pour accueillir les 3 images
                video_writer = cv2.VideoWriter('output_tracking_RGBD_PnP.avi', fourcc, 30.0, (final_w, final_h))
                next_write_time = time.time()
                
            curr_time = time.time()
            if video_writer is not None and curr_time >= next_write_time:
                video_writer.write(combined_frame)
                next_write_time += 1.0 / 30.0
                if curr_time > next_write_time + 0.1:
                    next_write_time = curr_time
            # ---------------------------------------------------

            # On affiche la frame d'origine (ou la combinée si tu préfères) dans la fenêtre OpenCV
            cv2.imshow("Tracking", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        print("\nShutting down...")
        shutdown.set()
        cam.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    torch.set_num_threads(4)
    os.environ["OMP_NUM_THREADS"] = "4"
    main()
