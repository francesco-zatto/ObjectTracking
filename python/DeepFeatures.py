import torchvision.models
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOAD_SQUEEZENET = lambda: torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT).eval()
LOAD_VGG16 = lambda: torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT).eval()
LOAD_MODELS = {
    'squeezenet': LOAD_SQUEEZENET,
    'vgg16': LOAD_VGG16
}

def load_model_feature_extractor(name='squeezenet', n_layers=3, print_net=False):
    model = LOAD_MODELS[name]()
    feature_extractor = model.features
    if print_net:
        for i, layer in enumerate(feature_extractor):
            print(f'{i}\t{layer.__class__.__name__}\t{layer}')
    return feature_extractor[:n_layers].to(DEVICE)

@torch.no_grad()
def computeRoiFeatures(roi, model):
    roi = (roi.astype(np.float32) / 255.0).transpose(2, 0, 1)
    roi = torch.from_numpy(roi).unsqueeze(0).to(DEVICE)

    features = model(roi)

    return features.squeeze(0)  # C x H x W

@torch.no_grad()
def get_quantized_k_channels(model, frame, top_k_indices, bins=32, f_min=None, f_max=None):
    # Extract and upsample frame features
    frame_torch = torch.from_numpy(frame.transpose(2,0,1)).float().unsqueeze(0).to(DEVICE) / 255.0
    frame_features = model(frame_torch).squeeze(0)  # C x H x W
    
    # Select only k specific channels
    selected_feat = frame_features[top_k_indices]
    
    # Quantize to [0, bins-1]
    if f_min is None or f_max is None:
        f_min, f_max = selected_feat.min(), selected_feat.max()
    quantized = ((selected_feat - f_min) / (f_max - f_min + 1e-6) * (bins - 1))
    
    return quantized.permute(1, 2, 0).cpu().numpy().astype(np.uint8), f_min, f_max