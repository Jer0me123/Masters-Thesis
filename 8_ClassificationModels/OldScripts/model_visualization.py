"""
Model Interpretation Visualization Script

This script generates visual explanations for ResNet-50 and ConvNeXt-Tiny
classification models by showing which regions of input images contribute
most to the model's predictions.

Supported visualization methods:
1. Grad-CAM: Gradient-weighted Class Activation Mapping
2. Integrated Gradients: Attribution method showing pixel importance

The script loads trained model weights, processes test set images, and
outputs side-by-side visualizations with heatmaps overlaid on original images.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import cv2
import os


# ============================================================
# Model Definitions (matching training script)
# ============================================================

class ResNet50Probe(nn.Module):
    """ResNet-50 probe model for attribute prediction."""
    def __init__(self, num_classes: int):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1) if out.shape[1] == 1 else out


class ConvNeXtTinyProbe(nn.Module):
    """ConvNeXt-Tiny probe model for attribute prediction."""
    def __init__(self, num_classes: int):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        out = self.backbone(x)
        return out.squeeze(1) if out.shape[1] == 1 else out


# ============================================================
# Grad-CAM Implementation
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps showing which spatial regions of the input image
    contribute most to the model's prediction for a specific class.
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network model
            target_layer: The convolutional layer to visualize (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to capture forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to capture backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Preprocessed input tensor [1, C, H, W]
            target_class: Class index to visualize (None for predicted class)
            
        Returns:
            cam: Heatmap array [H, W] with values in [0, 1]
        """
        # Forward pass
        model_output = self.model(input_image)
        
        # Determine target class
        if target_class is None:
            if model_output.dim() == 1:  # Binary classification
                target_class = (model_output > 0).long().item()
            else:  # Multiclass
                target_class = model_output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        if model_output.dim() == 1:  # Binary classification
            score = model_output
        else:  # Multiclass
            score = model_output[0, target_class]
        
        score.backward()
        
        # Calculate Grad-CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        # FIX: Create cam tensor on same device as activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


# ============================================================
# Integrated Gradients Implementation
# ============================================================

class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    
    Computes pixel-level importance by integrating gradients along
    a path from a baseline (e.g., black image) to the input image.
    """
    
    def __init__(self, model):
        self.model = model
    
    def generate_attribution(self, input_image, target_class=None, steps=50, baseline=None):
        """
        Generate integrated gradients attribution map.
        
        Args:
            input_image: Preprocessed input tensor [1, C, H, W]
            target_class: Class index (None for predicted class)
            steps: Number of integration steps
            baseline: Baseline image (None for black image)
            
        Returns:
            attribution: Attribution map [H, W]
        """
        if baseline is None:
            baseline = torch.zeros_like(input_image)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_image)
            if target_class is None:
                if output.dim() == 1:
                    target_class = (output > 0).long().item()
                else:
                    target_class = output.argmax(dim=1).item()
        
        # Detach input and baseline for interpolation
        input_clean = input_image.detach()
        baseline_clean = baseline.detach()
        
        # Compute gradients for each interpolated input
        grads = []
        for i in range(steps + 1):
            # Create interpolated input as a new leaf tensor
            alpha = float(i) / steps
            scaled_input = baseline_clean + alpha * (input_clean - baseline_clean)
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            
            # Forward pass
            output = self.model(scaled_input)
            
            # Compute score for target class
            if output.dim() == 1:
                score = output
            else:
                score = output[0, target_class]
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Store gradient
            grads.append(scaled_input.grad.detach())
        
        # Average gradients
        avg_grads = torch.stack(grads).mean(dim=0)
        
        # Multiply by (input - baseline)
        integrated_grads = (input_clean - baseline_clean) * avg_grads
        
        # Sum across color channels and get absolute values
        attribution = integrated_grads.squeeze(0).abs().sum(dim=0)
        attribution = attribution.cpu().numpy()
        attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        return attribution


# ============================================================
# Visualization Functions
# ============================================================

def denormalize_image(tensor):
    """
    Denormalize image tensor using ImageNet statistics.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        
    Returns:
        numpy array: Denormalized image [H, W, C] in range [0, 1]
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on image.
    
    Args:
        image: Original image array [H, W, C] in range [0, 1]
        heatmap: Heatmap array [H, W] in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap for heatmap
        
    Returns:
        overlay: Combined image with heatmap overlay
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), 
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    overlay = alpha * heatmap_colored + (1 - alpha) * image
    return overlay


def get_target_layer(model, backbone_type):
    """
    Get the target layer for Grad-CAM visualization.
    
    Args:
        model: The neural network model
        backbone_type: 'resnet50' or 'convnext_tiny'
        
    Returns:
        target_layer: The layer to visualize
    """
    if backbone_type == 'resnet50':
        # Last convolutional layer in ResNet-50
        return model.backbone.layer4[-1].conv3
    elif backbone_type == 'convnext_tiny':
        # Last convolutional layer in ConvNeXt-Tiny
        return model.backbone.features[-1][-1].block[5]
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


# ============================================================
# Main Visualization Pipeline
# ============================================================

def visualize_predictions(
    model,
    image_paths,
    labels,
    output_dir,
    backbone_type,
    num_classes,
    device,
    max_images=50,
    methods=['gradcam', 'integrated_gradients']
):
    """
    Generate and save visualization for a set of test images.
    
    Args:
        model: Trained model
        image_paths: List of paths to test images
        labels: True labels for images
        output_dir: Directory to save visualizations
        backbone_type: 'resnet50' or 'convnext_tiny'
        num_classes: Number of output classes (1 for binary, >1 for multiclass)
        device: torch device
        max_images: Maximum number of images to visualize
        methods: List of visualization methods to use
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    
    # Initialize visualization methods
    visualizers = {}
    if 'gradcam' in methods:
        target_layer = get_target_layer(model, backbone_type)
        visualizers['gradcam'] = GradCAM(model, target_layer)
    if 'integrated_gradients' in methods:
        visualizers['ig'] = IntegratedGradients(model)
    
    # Process images
    num_to_process = min(len(image_paths), max_images)
    
    for idx in tqdm(range(num_to_process), desc="Generating visualizations"):
        img_path = image_paths[idx]
        true_label = labels[idx]
        
        try:
            # Load and preprocess image
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                if num_classes == 1:
                    pred_prob = torch.sigmoid(output).item()
                    pred_label = int(pred_prob > 0.5)
                    confidence = pred_prob if pred_label == 1 else 1 - pred_prob
                else:
                    probs = F.softmax(output, dim=1)
                    pred_label = output.argmax(dim=1).item()
                    confidence = probs[0, pred_label].item()
            
            # Denormalize for visualization
            img_denorm = denormalize_image(img_tensor.squeeze(0))
            
            # Generate visualizations
            n_methods = len(visualizers)
            fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
            if n_methods == 0:
                axes = [axes]
            
            # Original image
            axes[0].imshow(img_denorm)
            axes[0].set_title(f'Original\nTrue: {true_label} | Pred: {pred_label}\nConf: {confidence:.3f}')
            axes[0].axis('off')
            
            # Generate each visualization method
            viz_idx = 1
            
            if 'gradcam' in visualizers:
                cam = visualizers['gradcam'].generate_cam(img_tensor, target_class=pred_label)
                overlay = overlay_heatmap(img_denorm, cam, alpha=0.5)
                axes[viz_idx].imshow(overlay)
                axes[viz_idx].set_title('Grad-CAM')
                axes[viz_idx].axis('off')
                viz_idx += 1
            
            if 'ig' in visualizers:
                attribution = visualizers['ig'].generate_attribution(
                    img_tensor.clone(), 
                    target_class=pred_label, 
                    steps=30
                )
                overlay_ig = overlay_heatmap(img_denorm, attribution, alpha=0.5)
                axes[viz_idx].imshow(overlay_ig)
                axes[viz_idx].set_title('Integrated Gradients')
                axes[viz_idx].axis('off')
                viz_idx += 1
            
            plt.tight_layout()
            
            # Save figure
            save_path = output_dir / f'vis_{idx:04d}_true{true_label}_pred{pred_label}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"\nError processing image {idx} ({img_path}): {e}")
            continue


def load_test_data(splits_json, image_dir):
    """Load test split from JSON file."""
    with open(splits_json) as f:
        data = json.load(f)
    
    test_data = data['test']
    image_paths = [os.path.join(image_dir, x['image']) for x in test_data]
    labels = [x['label'] for x in test_data]
    
    return image_paths, labels


def load_model(model_path, backbone_type, num_classes, device):
    """Load trained model from checkpoint."""
    if backbone_type == 'resnet50':
        model = ResNet50Probe(num_classes)
    elif backbone_type == 'convnext_tiny':
        model = ConvNeXtTinyProbe(num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate visual explanations for trained classification models"
    )
    p.add_argument(
        "--model_path", 
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    p.add_argument(
        "--splits_json", 
        required=True,
        help="Path to splits JSON file"
    )
    p.add_argument(
        "--task", 
        choices=["gender", "skintone"], 
        required=True,
        help="Classification task"
    )
    p.add_argument(
        "--backbone", 
        choices=["resnet50", "convnext_tiny"], 
        required=True,
        help="Model backbone architecture"
    )
    p.add_argument(
        "--image_dir", 
        required=True,
        help="Directory to load images from"
    )
    p.add_argument(
        "--output_dir", 
        default="visualizations",
        help="Directory to save visualization outputs"
    )
    p.add_argument(
        "--max_images", 
        type=int, 
        default=50,
        help="Maximum number of test images to visualize"
    )
    p.add_argument(
        "--methods",
        nargs='+',
        choices=['gradcam', 'integrated_gradients'],
        default=['gradcam', 'integrated_gradients'],
        help="Visualization methods to use"
    )
    
    return p.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine number of classes
    num_classes = 1 if args.task == "gender" else 3
    
    # Load test data
    print("Loading test data...")
    image_paths, labels = load_test_data(args.splits_json, args.image_dir)
    print(f"Found {len(image_paths)} test images")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.backbone, num_classes, device)
    
    # Generate visualizations
    print(f"Generating visualizations using methods: {args.methods}")
    visualize_predictions(
        model=model,
        image_paths=image_paths,
        labels=labels,
        output_dir=args.output_dir,
        backbone_type=args.backbone,
        num_classes=num_classes,
        device=device,
        max_images=args.max_images,
        methods=args.methods
    )
    
    print(f"\nVisualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


# Example usage:
# python model_visualization.py \
#     --model_path outputs_ResNet50_gender/gender_best_seed_0.pt \
#     --splits_json splits_gender.json \
#     --task gender \
#     --backbone resnet50 \
#     --image_dir path/to/images \
#     --output_dir visualizations/gender_resnet50 \
#     --max_images 100 \
#     --methods gradcam integrated_gradients