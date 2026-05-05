#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
import queue
import cv2
import numpy as np
import torch

# --- HACK D'IMPORTATION POUR MUGGLED_SAM ---
try:
    import muggled_sam
except ModuleNotFoundError:
    parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        sys.path.insert(0, os.path.abspath("."))

from muggled_sam.make_sam import make_sam_from_state_dict

# --- IMPORTATIONS CUTIE ---
import cutie
from cutie.model.cutie import CUTIE
from cutie.utils.get_default_model import get_default_model
from cutie.inference.inference_core import InferenceCore

# =====================================================================
# OUTILS DE MESURE (FPS)
# =====================================================================
class FPSCounter:
    def __init__(self, window_size=15):
        self.times = []
        self.window_size = window_size
        self.current_fps = 0.0
        self.lock = threading.Lock()

    def tick(self):
        with self.lock:
            self.times.append(time.time())
            if len(self.times) > self.window_size:
                self.times.pop(0)
            if len(self.times) > 1:
                self.current_fps = (len(self.times) - 1) / (self.times[-1] - self.times[0])

    def get_fps(self):
        with self.lock:
            return self.current_fps

# Statistiques globales
fps_detector = FPSCounter(window_size=5)
fps_tracker = FPSCounter(window_size=30)

# =====================================================================
# THREAD WEBCAM (ANTI-LAG)
# =====================================================================
class FastWebcam:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.stopped = True
        self.stream.release()

# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_SOURCE = 0
TEXT_PROMPT = "coffee mug"
DETECTION_MODEL_PATH = "/home/flanthier/.cache/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"

# --- PARAMÈTRES OPTIMISÉS ---
DETECTION_SCORE_THRESHOLD = 0.5
MASK_IOU_THRESHOLD = 0.35 # Pour l'anti-fantôme avec Cutie

# Palette de couleurs VIVES et distinctes (BGR pour OpenCV)
DISTINCT_COLORS = [
    (0, 0, 255),   # 1: Rouge vif
    (255, 0, 0),   # 2: Bleu pur
    (0, 255, 0),   # 3: Vert fluo
    (0, 255, 255), # 4: Jaune
    (255, 0, 255), # 5: Magenta
    (255, 255, 0), # 6: Cyan
    (0, 165, 255), # 7: Orange
    (200, 100, 255)# 8: Mauve clair
]

# =====================================================================
# VARIABLES PARTAGÉES
# =====================================================================
webcam_stream = None
shared_multi_mask = None
new_detections_queue = queue.Queue()
mask_lock = threading.Lock()
shutdown_flag = threading.Event()

# =====================================================================
# THREAD 1 : LE DÉTECTEUR (SAM 3.1)
# =====================================================================
def detector_worker(detmodel):
    print("[DÉTECTEUR] Thread démarré.")
    det_imgenc_config = {"max_side_length": 512, "use_square_sizing": True}
    
    while not shutdown_flag.is_set():
        ret, frame_to_analyze = webcam_stream.read()
        if not ret or frame_to_analyze is None:
            time.sleep(0.01)
            continue
            
        try:
            det_encimgs, _, _ = detmodel.encode_detection_image(frame_to_analyze, **det_imgenc_config)
            det_exemplars = detmodel.encode_exemplars(det_encimgs, text=TEXT_PROMPT)
            det_masks, det_boxes, _, _ = detmodel.generate_detections(
                det_encimgs, det_exemplars, detection_filter_threshold=DETECTION_SCORE_THRESHOLD
            )
            
            if det_masks is not None and det_masks.shape[1] > 0:
                target_size = frame_to_analyze.shape[0:2]
                resized_masks = torch.nn.functional.interpolate(
                    det_masks.float(), size=target_size, mode="bilinear", align_corners=False
                ) > 0.0
                
                masks_np = resized_masks.cpu().squeeze(0).numpy() # [N, H, W]
                
                # --- ÉTAPE CRUCIALE : NMS INTRA-BATCH (Retirer les doublons de SAM 3) ---
                valid_masks = []
                for i in range(masks_np.shape[0]):
                    m_candidate = masks_np[i]
                    is_duplicate = False
                    
                    for m_valid in valid_masks:
                        inter = np.logical_and(m_candidate, m_valid).sum()
                        union = np.logical_or(m_candidate, m_valid).sum()
                        iou = inter / union if union > 0 else 0
                        # Si SAM 3 a trouvé deux masques qui se chevauchent à + de 40%, on jette le doublon
                        if iou > 0.4: 
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        valid_masks.append(m_candidate)
                
                if len(valid_masks) > 0:
                    new_detections_queue.put(valid_masks)
                
        except Exception as e:
            print(f"[DÉTECTEUR] Erreur : {e}")

        fps_detector.tick() # On enregistre la fréquence
        time.sleep(0.1)     # On force un petit repos pour ne pas étouffer Cutie

# =====================================================================
# THREAD 2 : LE TRAQUEUR (CUTIE)
# =====================================================================
def tracker_worker(processor):
    global shared_multi_mask
    print("[TRAQUEUR] Thread démarré.")
    
    device = "cuda"
    tracked_ids = []
    next_obj_id = 1
    last_known_mask_np = None

    def prepare_image(cv_img):
        img_t = torch.from_numpy(cv_img).to(device, non_blocking=True)
        return img_t.permute(2, 0, 1).float().div_(255.0)

    while not shutdown_flag.is_set():
        ret, current_frame = webcam_stream.read()
        if not ret or current_frame is None:
            time.sleep(0.01)
            continue
            
        image_t = prepare_image(current_frame)
        update_required = False
        masks_to_add = []
        
        # -------------------------------------------------------------
        # VÉRIFICATION DES NOUVEAUX OBJETS
        # -------------------------------------------------------------
        while not new_detections_queue.empty():
            new_valid_masks = new_detections_queue.get() # Liste de masques [H, W]
            
            for new_mask_bin in new_valid_masks:
                is_known = False
                
                if last_known_mask_np is not None and len(tracked_ids) > 0:
                    for obj_id in tracked_ids:
                        existing_mask_bin = (last_known_mask_np == obj_id)
                        inter = np.logical_and(new_mask_bin, existing_mask_bin).sum()
                        union = np.logical_or(new_mask_bin, existing_mask_bin).sum()
                        
                        # Double vérification : IoU global OU si le nouveau masque est contenu dans l'ancien
                        iou = inter / union if union > 0 else 0
                        coverage = inter / new_mask_bin.sum() if new_mask_bin.sum() > 0 else 0
                        
                        if iou > MASK_IOU_THRESHOLD or coverage > 0.5:
                            is_known = True
                            break 
                            
                if not is_known:
                    print(f"[TRAQUEUR] Nouvelle cible '{TEXT_PROMPT}' validée ! (ID: {next_obj_id})")
                    masks_to_add.append((next_obj_id, new_mask_bin))
                    tracked_ids.append(next_obj_id)
                    next_obj_id += 1
                    update_required = True

        # -------------------------------------------------------------
        # INFÉRENCE CUTIE
        # -------------------------------------------------------------
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            if update_required:
                mask_t = torch.zeros(current_frame.shape[0:2], dtype=torch.int64, device=device)
                if last_known_mask_np is not None:
                    mask_t = torch.from_numpy(last_known_mask_np).to(device).long()
                
                for obj_id, mask_bin in masks_to_add:
                    mask_t[torch.from_numpy(mask_bin).to(device)] = obj_id
                    
                output_prob = processor.step(image_t, mask_t, objects=tracked_ids)
                
            elif len(tracked_ids) > 0:
                output_prob = processor.step(image_t)
            else:
                output_prob = None

            if output_prob is not None:
                last_known_mask_np = processor.output_prob_to_mask(output_prob).cpu().numpy().squeeze().astype(np.uint8)
            else:
                last_known_mask_np = None

        with mask_lock:
            shared_multi_mask = last_known_mask_np.copy() if last_known_mask_np is not None else None
            
        fps_tracker.tick() # On enregistre la fréquence du Traqueur

# =====================================================================
# THREAD PRINCIPAL : AFFICHAGE
# =====================================================================
def main():
    global webcam_stream, shared_multi_mask
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print("--- CHARGEMENT DES MODÈLES ---")
    _, sam3_model = make_sam_from_state_dict(DETECTION_MODEL_PATH)
    sam3_model.to(device=device, dtype=dtype)
    detmodel = sam3_model.make_detector_model()
    
    cutie_model = get_default_model() 
    processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)
    processor.max_internal_size = 240 
    
    print("--- DÉMARRAGE DE LA WEBCAM ULTRA-RAPIDE ---")
    webcam_stream = FastWebcam(VIDEO_SOURCE).start()
    time.sleep(1.0) 

    t_detect = threading.Thread(target=detector_worker, args=(detmodel,), daemon=True)
    t_track = threading.Thread(target=tracker_worker, args=(processor,), daemon=True)
    t_detect.start()
    t_track.start()

    print(f">>> SYSTÈME ACTIF. Cherche : '{TEXT_PROMPT}'. Appuyez sur 'q' pour quitter. <<<")
    
    try:
        while True:
            ret, display_frame = webcam_stream.read()
            if not ret or display_frame is None:
                continue
            
            # --- OVERLAY COULEURS OPTIMISÉ ---
            with mask_lock:
                local_mask = shared_multi_mask.copy() if shared_multi_mask is not None else None
            
            if local_mask is not None:
                overlay_layer = display_frame.copy()
                active_ids = np.unique(local_mask)
                
                for obj_id in active_ids:
                    if obj_id == 0: continue 
                    
                    color = DISTINCT_COLORS[(obj_id - 1) % len(DISTINCT_COLORS)]
                    overlay_layer[local_mask == obj_id] = color
                    
                    y_idx, x_idx = np.where(local_mask == obj_id)
                    if len(x_idx) > 0:
                        cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
                        cv2.putText(display_frame, f"ID: {obj_id}", (cx - 20, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(display_frame, f"ID: {obj_id}", (cx - 20, cy), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

                # Mélange parfait: 60% Image pure, 40% Masque de couleur (sans assombrir le reste)
                display_frame = cv2.addWeighted(overlay_layer, 0.4, display_frame, 0.6, 0)
                
            # --- AFFICHAGE DES FPS ---
            det_hz = fps_detector.get_fps()
            trk_hz = fps_tracker.get_fps()
            cv2.putText(display_frame, f"SAM 3 (Detection): {det_hz:.1f} Hz", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Cutie (Tracking): {trk_hz:.1f} Hz", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow(f"Cutie Temps Reel : {TEXT_PROMPT}", display_frame)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    finally:
        print("\nArrêt en cours...")
        shutdown_flag.set()
        webcam_stream.stop()
        cv2.destroyAllWindows()
        print("Système éteint.")

if __name__ == "__main__":
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    main()