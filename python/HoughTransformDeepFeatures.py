import torch
import numpy as np
import cv2
import os
import DeepFeatures

VIDEO_PATH = '../Sequences/Antoine_Mug_fixed.mp4'
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
ROI_LAYER = 3
TOP_K_CHANNELS = 8
BINS_PER_CHANNEL = 32
FRAMES_TO_SAVE = [1, 117, 195]
MODEL_NAME = 'squeezenet'
EXPERIMENT_NAME = f'HoughDeep_FeatureWeighted/layers{ROI_LAYER}_top{TOP_K_CHANNELS}_B{BINS_PER_CHANNEL}_{VIDEO_NAME}'

SEARCH_WINDOW = 20 # prevent the tracker from jumping around too much by only allowing votes within a certain pixel distance from the last position

def get_stride_for_layer(layer_idx):
    if layer_idx < 2:
        return 2
    elif layer_idx < 5:
        return 4
    elif layer_idx < 8:
        return 8  
    else:
        return 16

# Set stride dynamically
STRIDE = get_stride_for_layer(ROI_LAYER)

# roi selection
r, c, w, h = 0, 0, 0, 0
roi_defined = False

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r) # actually width
        w = abs(c2 - c) # actually height
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True
        print(f"ROI defined: x={r}, y={c}, width={h}, height={w}")

# R table for deep features
def get_R_table_deep(roi_quantized, n_bins):
    rows, cols, k_channels = roi_quantized.shape
    row_c, col_c = rows // 2, cols // 2
    R_tables = []

    for k in range(k_channels):
        channel_table = [[] for _ in range(n_bins)]
        feat_map = roi_quantized[:, :, k]
        for b in range(n_bins):
            y_coords, x_coords = np.where(feat_map == b)
            dr = y_coords - row_c
            dc = x_coords - col_c
            if len(dr) > 0:
                channel_table[b] = np.stack((dr, dc), axis=1)
            else:
                channel_table[b] = np.empty((0, 2), dtype=int)
        R_tables.append(channel_table)
    return R_tables

# deep hough transform
def houghTransform_deep(frame_quantized, R_tables, n_bins, prev_center=None, window_size=15):
    rows, cols, k_channels = frame_quantized.shape
    counter_image = np.zeros((rows, cols), dtype=float)
    
    # voting
    for k in range(k_channels):
        feat_map = frame_quantized[:, :, k]
        current_R_table = R_tables[k]
        for b in range(n_bins):
            y_idx, x_idx = np.where(feat_map == b)
            if len(y_idx) == 0: continue
            vectors = current_R_table[b]
            if len(vectors) == 0: continue
            for dr, dc in vectors:
                target_y = y_idx - dr
                target_x = x_idx - dc
                valid = (target_y >= 0) & (target_y < rows) & (target_x >= 0) & (target_x < cols)
                np.add.at(counter_image, (target_y[valid], target_x[valid]), 1)

    # apply search window
    if prev_center is not None:
        mask = np.zeros_like(counter_image, dtype=float)
        pr, pc = prev_center 
        r_min = max(0, int(pr - window_size))
        r_max = min(rows, int(pr + window_size))
        c_min = max(0, int(pc - window_size))
        c_max = min(cols, int(pc + window_size))
        mask[r_min:r_max, c_min:c_max] = 1.0
        counter_image = counter_image * mask

    # Find the max value to threshold noise
    max_val = np.max(counter_image)
    if max_val == 0: return (0,0), counter_image
    
    threshold = 0.9 * max_val
    peak_mask = counter_image >= threshold
    
    # center of mass
    coords = np.argwhere(peak_mask)
    weights = counter_image[peak_mask]
    
    # weighted average of coordinates
    w_y = np.sum(coords[:, 0] * weights) / np.sum(weights)
    w_x = np.sum(coords[:, 1] * weights) / np.sum(weights)
    
    return (w_y, w_x), counter_image



# feature visualization
def visualize_topk_features(frame, frame_features, top_k_idx):
    maps = []
    for i, idx in enumerate(top_k_idx):
        fmap = frame_features[idx].detach().cpu().numpy()
        fmap_norm = ((fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6) * 255).astype(np.uint8)
        fmap_resized = cv2.resize(fmap_norm, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(f'Feature Map {i+1}', fmap_resized)
        maps.append(fmap_resized)
    return maps


def run_hough_deep_feature_viz(video_path):
    global r, c, w, h, roi_defined

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # load
    model = DeepFeatures.load_model_feature_extractor(name=MODEL_NAME, n_layers=ROI_LAYER, print_net=False)
    experiment_path = os.path.join('../Experiments', EXPERIMENT_NAME)
    os.makedirs(experiment_path, exist_ok=True)

    # read first frame and select ROI
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return
    
    clone = frame.copy()
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", define_ROI)
    print("Select the ROI with your mouse, then press 'q'.")
    while True:
        display = frame.copy()
        if roi_defined:
            cv2.rectangle(display, (r, c), (r + h, c + w), (0, 255, 0), 2)
        cv2.imshow("Select ROI", display)
        if cv2.waitKey(1) & 0xFF == ord("q") and roi_defined:
            cv2.destroyWindow("Select ROI")
            break

    # extract ROI
    roi = clone[c:c+w, r:r+h]
    
    # convert ROI to RGB 
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    roi_features = DeepFeatures.computeRoiFeatures(roi_rgb, model)
    actual_k = min(TOP_K_CHANNELS, roi_features.shape[0])
    channel_scores = roi_features.mean(dim=(1,2))
    _, top_k_idx = torch.topk(channel_scores, actual_k)

    roi_quantized, f_min, f_max = DeepFeatures.get_quantized_k_channels(model, roi_rgb, top_k_idx, bins=BINS_PER_CHANNEL)
    R_tables = get_R_table_deep(roi_quantized, BINS_PER_CHANNEL)

    # initialize last position for the search window
    # r, c are top-left. Add half width/height to get center
    init_center_row = (c + w // 2) // STRIDE
    init_center_col = (r + h // 2) // STRIDE
    last_pos = (init_center_row, init_center_col)

    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # convert Frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_features = DeepFeatures.computeRoiFeatures(frame_rgb, model)
        
        # quantize
        frame_quantized, _, _ = DeepFeatures.get_quantized_k_channels(
            model, frame_rgb, top_k_idx, bins=BINS_PER_CHANNEL, f_min=f_min, f_max=f_max
        )
        
        # run Hough with search window
        (row_center_feat, col_center_feat), response = houghTransform_deep(
            frame_quantized, 
            R_tables, 
            BINS_PER_CHANNEL,
            prev_center=last_pos,      # Pass history
            window_size=SEARCH_WINDOW  # Pass constraint
        )

        # update history
        last_pos = (row_center_feat, col_center_feat)

        # convert back to pixel space
        pixel_r = row_center_feat * STRIDE
        pixel_c = col_center_feat * STRIDE

        # draw rectangle (adjusting for center)
        draw_x = pixel_c - (h // 2) 
        draw_y = pixel_r - (w // 2)
        
        tracking_frame = frame.copy()
        cv2.rectangle(
            tracking_frame, 
            (int(draw_x), int(draw_y)), 
            (int(draw_x + h), int(draw_y + w)), 
            (255, 0, 0), 
            2
        )

        # visualize
        feature_maps = visualize_topk_features(frame, frame_features, top_k_idx)

        # save specific frames
        if frame_idx in FRAMES_TO_SAVE:
            cv2.imwrite(os.path.join(experiment_path, f"Tracking_{frame_idx:04d}.png"), tracking_frame)
            response_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(experiment_path, f"Response_{frame_idx:04d}.png"), response_norm)
            for i, fmap in enumerate(feature_maps):
                cv2.imwrite(os.path.join(experiment_path, f"FeatureMap{frame_idx:04d}_{i+1}.png"), fmap)
            print(f"Saved frame {frame_idx}")

        cv2.imshow("Tracking", tracking_frame)
        frame_idx += 1
        if cv2.waitKey(30) & 0xFF == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Frames saved in:", experiment_path)

run_hough_deep_feature_viz(VIDEO_PATH)