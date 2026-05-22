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
from cotracker.predictor import CoTrackerPredictor

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


# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_SOURCE = 0
DETECTION_MODEL_PATH = "/home/flanthier/.cache/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"
TRACKING_MODEL_PATH  = "/home/flanthier/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"

# Detection
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

# Co-Tracker Point Tracker Configurations
COTRACKER_MAX_POINTS = 500
COTRACKER_MIN_DIST = 15

def calculate_rigid_transform(points_source, points_target):
    """
    Calcule la matrice de transformation 4x4 (Rotation + Translation) 
    pour aligner points_source sur points_target en utilisant la SVD.
    """
    # Centrer les points
    centroid_src = np.mean(points_source, axis=0)
    centroid_tgt = np.mean(points_target, axis=0)
    
    src_centered = points_source - centroid_src
    tgt_centered = points_target - centroid_tgt
    
    # Calculer la matrice de covariance
    H = src_centered.T @ tgt_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Gérer la réflexion (cas où le déterminant est négatif)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    t = centroid_tgt.T - R @ centroid_src.T
    
    # Créer la matrice 4x4
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform

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
            return (self.ret, self.frame.copy()) if self.frame is not None else (False, None)

    def stop(self):
        self._stopped = True
        self.cap.release()


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
        self.last_semantically_verified = time.time()  # HEARTBEAT
        
        # Point Tracking State
        self.active_points = None      # np.ndarray of shape (N, 2), floats
        self.point_visibilities = None # np.ndarray of shape (N,), bools
        self.point_invisible_frames = None # np.ndarray of shape (N,), ints


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
# THREAD 1 — DETECTOR  (SAM 3)
# =====================================================================
def detector_worker(detmodel, cam, prompts_ref, det_queue, shutdown, fps):
    print("[DETECTOR] Started.")
    enc_cfg = {"max_side_length": 512, "use_square_sizing": True}

    while not shutdown.is_set():
        ret, frame = cam.read()
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
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    h, w = shape_hw
    return torch.tensor(
        [[xs.min() / w, ys.min() / h],
         [xs.max() / w, ys.max() / h]], dtype=torch.float32
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
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]

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

            for obj in tracked.values():
                score, best_idx, preds, mem_enc, obj_ptr = track_model.step_video_masking(
                    enc_imgs, **obj.memory.to_dict()
                )
                obj.last_score = score.item()

                if score.item() < 0:
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

                    mask_f = preds[0, best_idx].cpu().float().numpy().squeeze()
                    mask_full = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)
                    obj.last_mask_np = mask_full
                    obj.last_box_norm = box_from_mask(mask_full > 0, (h, w))

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
        # C2) PUBLISH TRACKED DICT FOR POINT TRACKER
        # ==========================================================
        with shared_state["lock"]:
            shared_state["tracked_objects"] = dict(tracked)

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
                    "points": obj.active_points,
                    "vis": obj.point_visibilities,
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
            tag = f"{obj['label']} #{obj['id']}"
            draw_outlined_text(display, tag, (cx - 30, cy),
                               scale=0.5, fg=obj["color"], thickness=1)

        # Draw Co-Tracker Points
        pts = obj.get("points")
        vis = obj.get("vis")
        if pts is not None and vis is not None:
            pt_color = tuple(int(c) for c in obj["color"]) # BGR
            # Use contrasting color for perimeter
            inv_color = (255 - pt_color[0], 255 - pt_color[1], 255 - pt_color[2])
            for (x, y), is_vis in zip(pts, vis):
                if is_vis:
                    cv2.circle(display, (int(x), int(y)), 3, pt_color, -1)
                    cv2.circle(display, (int(x), int(y)), 3, inv_color, 1)
                else:
                    cv2.circle(display, (int(x), int(y)), 3, pt_color, 1)

# =====================================================================
# THREAD 3 — POINT TRACKER (CoTracker)
# =====================================================================
def point_tracker_worker(cotracker_model, cam, shared_state, prompts_ref, shutdown, fps):
    print("[POINT TRACKER] Started.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Needs to process 2-frame chunks.
    prev_frame = None
    
    while not shutdown.is_set():
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
            
        with prompts_ref["lock"]:
            if prompts_ref.get("changed", False):
                prev_frame = None
                
        if prev_frame is None:
            prev_frame = frame.copy()
            time.sleep(0.03)  # Wait for next frame
            continue
            
        # We have prev_frame and frame. Form a 2-frame chunk.
        h, w = frame.shape[:2]
        
        # 1. Gather all currently tracked objects and their active points
        objects_to_track = []
        queries_list = []
        
        with shared_state["lock"]:
            # We can grab a snapshot of the objects to process.
            tracked_dict = shared_state.get("tracked_objects", {})
            current_objects = list(tracked_dict.values())
            
        # Format the video chunk: shape (B, T, C, H, W)
        video_chunk = np.stack([prev_frame, frame]) # shape (2, H, W, 3)
        video_tensor = torch.from_numpy(video_chunk).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) # (1, 2, 3, H, W)
        
        # Determine global point budget and assignments
        total_queries = 0
        valid_objects = []
        
        for obj in current_objects:
            # Skip if object has no mask currently
            if obj.last_mask_np is None:
                continue
            
            pts = obj.active_points
            
            # 2. Resampling logic: Fill up to COTRACKER_MAX_POINTS
            mask_indices = np.argwhere(obj.last_mask_np > 0)
            if len(mask_indices) > 0:
                needed_points = COTRACKER_MAX_POINTS if pts is None else max(0, COTRACKER_MAX_POINTS - len(pts))
                if needed_points > 0:
                    # Randomly sample candidate points from the mask
                    samples = mask_indices[np.random.choice(len(mask_indices), size=min(needed_points * 5, len(mask_indices)), replace=False)]
                    # samples are (y, x)
                    new_pts_list = []
                    for (y, x) in samples:
                        if len(new_pts_list) >= needed_points:
                            break
                        pt = np.array([x, y], dtype=np.float32)
                        
                        # Distance check against existing pts
                        if pts is not None and len(pts) > 0:
                            dists = np.linalg.norm(pts - pt, axis=1)
                            if np.min(dists) < COTRACKER_MIN_DIST:
                                continue
                                
                        # Distance check against newly added pts
                        if len(new_pts_list) > 0:
                            dists = np.linalg.norm(np.array(new_pts_list) - pt, axis=1)
                            if np.min(dists) < COTRACKER_MIN_DIST:
                                continue
                                
                        new_pts_list.append(pt)
                    
                    if new_pts_list:
                        new_pts_arr = np.array(new_pts_list, dtype=np.float32)
                        new_inv_frames = np.zeros(len(new_pts_arr), dtype=int)
                        
                        if pts is None:
                            pts = new_pts_arr
                            obj.point_invisible_frames = new_inv_frames
                        else:
                            pts = np.vstack((pts, new_pts_arr))
                            obj.point_invisible_frames = np.concatenate((obj.point_invisible_frames, new_inv_frames))
                            
            if pts is not None and len(pts) > 0:
                obj.active_points = pts
                # Format for cotracker: (t, x, y). Queries start at t=0 (prev_frame)
                q = np.zeros((len(pts), 3), dtype=np.float32)
                q[:, 1:] = pts
                queries_list.append(q)
                valid_objects.append(obj)
                total_queries += len(pts)
                
        if total_queries > 0:
            all_queries = np.vstack(queries_list)
            queries_tensor = torch.from_numpy(all_queries).unsqueeze(0).to(device) # (1, N, 3)
            
            # Predict
            with torch.no_grad():
                pred_tracks, pred_vis = cotracker_model(video=video_tensor, queries=queries_tensor)
                
            # pred_tracks: (1, 2, N, 2), pred_vis: (1, 2, N)
            new_points = pred_tracks[0, 1].cpu().numpy() # (N, 2)
            new_vis = pred_vis[0, 1].cpu().numpy() # (N,)
            
            # Distribute back to objects and filter
            offset = 0
            for obj in valid_objects:
                N_obj = len(obj.active_points)
                obj_pts = new_points[offset:offset+N_obj]
                obj_vis = new_vis[offset:offset+N_obj]
                
                # Correct the trajectories of occluded points!
                # Because CoTracker runs on 2-frame chunks, an occluded point will latch onto 
                # the occluder (e.g., the hand). We need to correct its position by applying 
                # the rigid motion of the VISIBLE points to the INVISIBLE points.
                
                on_sam_mask = np.zeros(N_obj, dtype=bool)
                for i in range(N_obj):
                    px, py = int(round(obj_pts[i, 0])), int(round(obj_pts[i, 1]))
                    if 0 <= px < w and 0 <= py < h:
                        if obj.last_mask_np[py, px] > 0:
                            on_sam_mask[i] = True
                            
                visible_indices = np.where(on_sam_mask)[0]
                invisible_indices = np.where(~on_sam_mask)[0]
                
                if len(visible_indices) >= 4 and len(invisible_indices) > 0:
                    src_pts = obj.active_points[visible_indices]
                    tgt_pts = obj_pts[visible_indices]
                    M, inliers = cv2.estimateAffinePartial2D(src_pts, tgt_pts)
                    if M is not None:
                        inv_pts = obj.active_points[invisible_indices]
                        inv_pts_homo = np.hstack([inv_pts, np.ones((len(inv_pts), 1))])
                        corrected_pts = (M @ inv_pts_homo.T).T
                        obj_pts[invisible_indices] = corrected_pts.astype(np.float32)
                    else:
                        translation = np.median(tgt_pts - src_pts, axis=0)
                        obj_pts[invisible_indices] = obj.active_points[invisible_indices] + translation
                elif len(visible_indices) > 0 and len(invisible_indices) > 0:
                    translation = np.median(obj_pts[visible_indices] - obj.active_points[visible_indices], axis=0)
                    obj_pts[invisible_indices] = obj.active_points[invisible_indices] + translation
                elif len(visible_indices) == 0 and len(invisible_indices) > 0:
                    # Object is completely occluded, freeze points
                    obj_pts[invisible_indices] = obj.active_points[invisible_indices]
                
                # Point updates: We ALWAYS keep all points, but we override visibility 
                # so that if a point is off the SAM2 mask (e.g. covered by a hand), 
                # it is forced to be 'invisible' (hollow contour) regardless of CoTracker.
                keep_mask = np.ones(N_obj, dtype=bool)
                for i in range(N_obj):
                    # We recalculate projection in case the point was corrected!
                    px, py = int(round(obj_pts[i, 0])), int(round(obj_pts[i, 1]))
                    
                    # Check if the corrected point is inside the projected SAM mask
                    sam_vis = False
                    if 0 <= px < w and 0 <= py < h:
                        if obj.last_mask_np[py, px] > 0:
                            sam_vis = True
                            
                    final_vis = obj_vis[i] and sam_vis
                    obj_vis[i] = final_vis
                    
                    if final_vis:
                        obj.point_invisible_frames[i] = 0
                    else:
                        obj.point_invisible_frames[i] += 1
                        
                obj.active_points = obj_pts[keep_mask]
                obj.point_visibilities = obj_vis[keep_mask]
                obj.point_invisible_frames = obj.point_invisible_frames[keep_mask]
                offset += N_obj

        # Loop cleanup, the current frame becomes prev_frame
        prev_frame = frame.copy()
        
        # Publish point tracker FPS
        fps.tick()
        with shared_state["lock"]:
            shared_state["pt_fps"] = fps.fps



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

    print("Loading SAM 3 (detector)...")
    _, sam3 = make_sam_from_state_dict(DETECTION_MODEL_PATH)
    sam3.to(device=device, dtype=dtype)
    detmodel = sam3.make_detector_model()

    print("Loading SAM 2.1 Tiny (tracker)...")
    _, sam2 = make_sam_from_state_dict(TRACKING_MODEL_PATH)
    sam2.to(device=device, dtype=dtype)
    
    print("Loading Co-Tracker 3 (point tracker)...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    cotracker = cotracker.to(device)
    # optional: warmup
    # cotracker(video=torch.zeros(1, 2, 3, 224, 224, device=device), queries=torch.zeros(1, 1, 3, device=device))
    
    print("Models ready.\n")

    shared_state = {"objects": [], "tracked_objects": {}, "lock": threading.Lock(), "pt_fps": 0.0}
    det_queue = queue.Queue()
    shutdown = threading.Event()

    cam = FastWebcam(VIDEO_SOURCE).start()
    time.sleep(0.5)

    fps_det = FPSCounter()
    fps_trk = FPSCounter()
    fps_pt  = FPSCounter()

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

    threading.Thread(
        target=point_tracker_worker,
        args=(cotracker, cam, shared_state, prompts_ref, shutdown, fps_pt),
        daemon=True,
    ).start()

    print(f"Tracking: {initial_prompts}")
    print("Type new prompts in the terminal at any time (semicolon-separated).")
    print("Press 'q' or Esc in the window to quit.\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            display = frame.copy()

            with shared_state["lock"]:
                objects = list(shared_state["objects"])

            overlay_masks(display, objects)

            draw_outlined_text(display, f"Det: {fps_det.fps:.1f} Hz", (10, 28), fg=(0, 255, 255))
            draw_outlined_text(display, f"Trk: {fps_trk.fps:.1f} Hz", (10, 56), fg=(0, 255, 0))

            with prompts_ref["lock"]:
                prompt_str = " | ".join(prompts_ref["prompts"])
            draw_outlined_text(display, f"Prompts: {prompt_str}", (10, 84),
                               fg=(255, 200, 100), scale=0.5, thickness=1)
            draw_outlined_text(display, f"Objects: {len(objects)}", (10, 108),
                               fg=(255, 255, 255), scale=0.5, thickness=1)

            # Retrieve point tracker FPS if present
            pt_fps = shared_state.get("pt_fps", 0.0)
            draw_outlined_text(display, f"Pts: {pt_fps:.1f} Hz", (10, 132), fg=(255, 0, 255))

            cv2.imshow("SAM3 + SAM2.1 Tracker", display)

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
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    main()
