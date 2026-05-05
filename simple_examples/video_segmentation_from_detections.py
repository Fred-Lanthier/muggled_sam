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

# --- HACK D'IMPORTATION (Permet de lancer depuis n'importe où dans le repo) ---
try:
    import muggled_sam
except ModuleNotFoundError:
    parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "muggled_sam" in os.listdir(parent_folder):
        sys.path.insert(0, parent_folder)
    else:
        # Tente de l'ajouter si on est à la racine
        sys.path.insert(0, os.path.abspath("."))

from muggled_sam.make_sam import make_sam_from_state_dict
from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
from muggled_sam.demo_helpers.bounding_boxes import get_2box_iou

# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_SOURCE = 0  # 0 pour la webcam par défaut
TEXT_PROMPT = "coffee mug"

# Vos chemins de modèles
DETECTION_MODEL_PATH = "/home/flanthier/.cache/huggingface/hub/models--facebook--sam3/snapshots/2afe64078f4420bdfbc063162d1336003efadc81/sam3.pt"
TRACKING_MODEL_PATH = "/home/flanthier/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt"

# Paramètres de détection et suivi
DETECTION_SCORE_THRESHOLD = 0.5
EXISTING_BOX_IOU_THRESHOLD = 0.25 # Si IoU > 25%, on considère que c'est la même bouche
REMOVE_AFTER_N_MISSED = 5         # Nombre de frames avant d'oublier une cible perdue

# =====================================================================
# VARIABLES PARTAGÉES (THREADS)
# =====================================================================
shared_frame = None
shared_masks_to_draw = []
new_detections_queue = queue.Queue()
frame_lock = threading.Lock()
mask_lock = threading.Lock()
shutdown_flag = threading.Event()

# =====================================================================
# THREAD 1 : LE DÉTECTEUR (SAM 3.1 - Basse Fréquence)
# =====================================================================
def detector_worker(detmodel):
    global shared_frame
    
    print("[DÉTECTEUR] Thread démarré (SAM 3.1). En attente de frames...")
    
    # Configuration légère pour l'encodage (accélère SAM3)
    det_imgenc_config = {"max_side_length": 512, "use_square_sizing": True}
    
    while not shutdown_flag.is_set():
        # 1. Copier la dernière image disponible
        with frame_lock:
            if shared_frame is None:
                time.sleep(0.01)
                continue
            frame_to_analyze = shared_frame.copy()
            
        t1 = time.perf_counter()
        
        # 2. Exécuter l'inférence SAM 3
        try:
            det_encimgs, _, _ = detmodel.encode_detection_image(frame_to_analyze, **det_imgenc_config)
            det_exemplars = detmodel.encode_exemplars(det_encimgs, text=TEXT_PROMPT)
            det_masks, det_boxes, _, _ = detmodel.generate_detections(
                det_encimgs, det_exemplars, detection_filter_threshold=DETECTION_SCORE_THRESHOLD
            )
            
            num_detections = det_masks.shape[1] if det_masks is not None else 0
            
            # 3. Envoyer les résultats au traqueur s'il y a des détections
            if num_detections > 0:
                # On transmet la frame source pour que le tracker s'initialise correctement
                new_detections_queue.put({
                    "masks": det_masks,
                    "boxes": det_boxes,
                    "source_frame": frame_to_analyze
                })
                
        except Exception as e:
            print(f"[DÉTECTEUR] Erreur d'inférence: {e}")

        # Petite pause pour laisser respirer le GPU et le Traqueur
        time.sleep(0.1)

# =====================================================================
# THREAD 2 : LE TRAQUEUR (SAM 2.1 Tiny - Haute Fréquence)
# =====================================================================
def tracker_worker(track_model):
    global shared_frame, shared_masks_to_draw
    
    print("[TRAQUEUR] Thread démarré (SAM 2.1 Tiny).")
    
    memory_per_obj_dict = {}
    missed_frames_dict = {}
    next_obj_id = 0
    frame_idx = 0
    
    track_imgenc_config = {"max_side_length": 512, "use_square_sizing": True}
    
    while not shutdown_flag.is_set():
        with frame_lock:
            if shared_frame is None:
                time.sleep(0.01)
                continue
            current_frame = shared_frame.copy()
            
        # -------------------------------------------------------------
        # ÉTAPE A : GÉRER LES NOUVELLES DÉTECTIONS (Boîte aux lettres)
        # -------------------------------------------------------------
        while not new_detections_queue.empty():
            det_data = new_detections_queue.get()
            new_masks = det_data["masks"]
            new_boxes = det_data["boxes"]
            source_frame = det_data["source_frame"]
            
            num_detections = new_masks.shape[1]
            
            # Obtenir les boîtes des objets qu'on traque DÉJÀ
            known_boxes_list = []
            if len(memory_per_obj_dict) > 0:
                # On calcule les boites connues à partir de leur dernier masque
                with mask_lock:
                    for mask_tensor in shared_masks_to_draw:
                        mask_uint8 = (mask_tensor > 0).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            x, y, w, h = cv2.boundingRect(c)
                            # Normaliser la boite pour la comparaison
                            box_norm = torch.tensor([[x, y], [x+w, y+h]], dtype=torch.float32) / torch.tensor([mask_uint8.shape[1], mask_uint8.shape[0]])
                            known_boxes_list.append(box_norm.to(new_boxes.device))
            
            # Encodage de la frame source (celle où SAM3 a vu la bouche) pour l'initialisation
            source_encimgs, _, _ = track_model.encode_image(source_frame, **track_imgenc_config)
            
            for idx_det in range(num_detections):
                new_box = new_boxes[0, idx_det]
                
                # Vérifier si c'est vraiment un NOUVEL objet (IoU)
                is_known = False
                if len(known_boxes_list) > 0:
                    is_known = any(get_2box_iou(new_box, b) > EXISTING_BOX_IOU_THRESHOLD for b in known_boxes_list)
                
                if not is_known:
                    print(f"[TRAQUEUR] Nouvelle cible '{TEXT_PROMPT}' verrouillée ! (ID: {next_obj_id})")
                    raw_mask = new_masks[0, idx_det]
                    
                    # Initialisation dans la mémoire du traqueur
                    init_mem = track_model.initialize_from_mask(source_encimgs, raw_mask > 0)
                    
                    obj_memory = SAMVideoObjectResults.create()
                    obj_memory.store_prompt_result(frame_idx, init_mem)
                    
                    memory_per_obj_dict[next_obj_id] = obj_memory
                    missed_frames_dict[next_obj_id] = 0
                    next_obj_id += 1

        # -------------------------------------------------------------
        # ÉTAPE B : FAIRE AVANCER LE SUIVI (Frame Courante)
        # -------------------------------------------------------------
        current_masks_for_display = []
        objs_to_remove = []
        
        if len(memory_per_obj_dict) > 0:
            # Encoder la frame actuelle une seule fois pour tous les objets
            encoded_imgs_list, _, _ = track_model.encode_image(current_frame, **track_imgenc_config)
            
            for obj_id, obj_memory in memory_per_obj_dict.items():
                obj_score, best_mask_idx, mask_preds, mem_enc, obj_ptr = track_model.step_video_masking(
                    encoded_imgs_list, **obj_memory.to_dict()
                )
                
                if obj_score.item() < 0: # Cible perdue ou cachée
                    missed_frames_dict[obj_id] += 1
                    if missed_frames_dict[obj_id] > REMOVE_AFTER_N_MISSED:
                        objs_to_remove.append(obj_id)
                else:
                    missed_frames_dict[obj_id] = 0
                    # Sauvegarder la mémoire pour la prochaine itération
                    obj_memory.store_frame_result(frame_idx, mem_enc, obj_ptr)
                    
                    # Extraire le masque pour l'affichage (format numpy)
                    best_mask = mask_preds[0, best_mask_idx, :, :].cpu().float().numpy().squeeze()
                    current_masks_for_display.append(best_mask)
            
            # Nettoyer les objets perdus définitivement
            for obj_id in objs_to_remove:
                memory_per_obj_dict.pop(obj_id)
                missed_frames_dict.pop(obj_id)
                print(f"[TRAQUEUR] Cible {obj_id} perdue et purgée de la mémoire.")
                
        # Mettre à jour les masques partagés pour le thread principal
        with mask_lock:
            shared_masks_to_draw = current_masks_for_display
            
        frame_idx += 1

# =====================================================================
# THREAD PRINCIPAL : CAMÉRA ET AFFICHAGE
# =====================================================================
def main():
    global shared_frame, shared_masks_to_draw
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print("--- CHARGEMENT DES MODÈLES ---")
    print("1. Modèle Détecteur (SAM 3)...")
    _, sam3_model = make_sam_from_state_dict(DETECTION_MODEL_PATH)
    sam3_model.to(device=device, dtype=dtype)
    detmodel = sam3_model.make_detector_model()
    
    print("2. Modèle Traqueur (SAM 2.1 Tiny)...")
    _, track_model = make_sam_from_state_dict(TRACKING_MODEL_PATH)
    track_model.to(device=device, dtype=dtype)
    print("--- MODÈLES PRÊTS ---")

    # Démarrage des threads
    t_detect = threading.Thread(target=detector_worker, args=(detmodel,))
    t_track = threading.Thread(target=tracker_worker, args=(track_model,))
    t_detect.start()
    t_track.start()

    # Démarrage de la webcam
    print("Ouverture de la webcam...")
    vcap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not vcap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        shutdown_flag.set()
        return

    print(">>> SYSTÈME ACTIF. Appuyez sur 'q' ou 'Echap' pour quitter. <<<")
    
    try:
        while True:
            ret, frame = vcap.read()
            if not ret:
                break
            
            # Mettre à jour l'image pour les workers
            with frame_lock:
                shared_frame = frame
                
            # Préparer l'affichage
            display_frame = frame.copy()
            
            # Dessiner les masques
            with mask_lock:
                local_masks = list(shared_masks_to_draw)
            
            if len(local_masks) > 0:
                combined_mask = np.zeros(display_frame.shape[0:2], dtype=bool)
                for mask_array in local_masks:
                    # Redimensionner le masque si la résolution diffère
                    mask_resized = cv2.resize(mask_array, (display_frame.shape[1], display_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask = np.bitwise_or(combined_mask, mask_resized > 0)
                
                # Appliquer une couleur (vert semi-transparent)
                color_mask = np.zeros_like(display_frame)
                color_mask[combined_mask] = [0, 255, 0] 
                display_frame = cv2.addWeighted(display_frame, 1.0, color_mask, 0.4, 0)
                
                # Afficher le nombre de cibles
                cv2.putText(display_frame, f"Cibles traquees: {len(local_masks)}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            flipped_frame = cv2.flip(display_frame, 1)
            cv2.imshow(f"Suivi Asynchrone : {TEXT_PROMPT}", flipped_frame)
            
            # Quitter proprement
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

    finally:
        print("\nArrêt en cours...")
        shutdown_flag.set()
        vcap.release()
        cv2.destroyAllWindows()
        t_detect.join()
        t_track.join()
        print("Système éteint.")

if __name__ == "__main__":
    # Paramètre requis par PyTorch pour le multithreading GPU intensif
    torch.set_num_threads(1)
    main()