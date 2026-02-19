"""
CLIP Skin Tone Interrogator
============================

Uses CLIP in REVERSE to discover what words it associates with different skin tones.

Instead of:
  Text prompts → CLIP → Image classification
  
We do:
  MST images → CLIP → Find best matching words

This reveals CLIP's "internal vocabulary" for skin tones,
which we can use to create better prompts!

Usage:
    python clip_interrogator.py \
        --mst_e_csv "MST-E/annotations.csv" \
        --mst_e_dir "MST-E/" \
        --output_json "clip_skin_tone_vocabulary.json"
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel


##############################################################
# SKIN TONE VOCABULARY
##############################################################

# Comprehensive list of skin tone descriptors
# CLIP will tell us which ones it associates with each MST level
SKIN_TONE_VOCABULARY = {
    # Lightness descriptors
    'lightness': [
        'very pale', 'pale', 'fair', 'light', 'medium', 'dark', 'very dark',
        'lightest', 'lighter', 'darker', 'darkest',
        'extremely pale', 'extremely light', 'extremely dark',
    ],
    
    # Color descriptors
    'colors': [
        'white', 'ivory', 'cream', 'porcelain', 'alabaster',
        'beige', 'sand', 'tan', 'honey', 'golden',
        'olive', 'bronze', 'copper', 'caramel',
        'brown', 'chocolate', 'mocha', 'coffee', 'espresso',
        'mahogany', 'ebony', 'umber', 'sienna',
        'peachy', 'pinkish', 'yellowish', 'reddish',
    ],
    
    # Descriptive adjectives
    'descriptors': [
        'translucent', 'rosy', 'flushed', 'ruddy',
        'sun-kissed', 'tanned', 'suntanned', 'bronzed',
        'warm-toned', 'cool-toned', 'neutral-toned',
        'golden-undertone', 'pink-undertone', 'red-undertone',
        'rich', 'deep', 'luminous', 'glowing',
    ],
    
    # Comparative
    'comparatives': [
        'lighter than average', 'darker than average',
        'very light compared to most', 'very dark compared to most',
        'middle range', 'median tone',
    ],
    
    # Food/object comparisons (common in descriptions)
    'metaphors': [
        'milk', 'cream', 'vanilla', 'peach',
        'almond', 'hazelnut', 'walnut', 'chestnut',
        'cinnamon', 'nutmeg', 'cocoa', 'fudge',
        'onyx', 'obsidian', 'jet',
    ],
    
    # Sun response
    'sun_response': [
        'burns easily', 'tans easily',
        'never burns', 'always burns',
        'sensitive to sun', 'resistant to sun',
    ],
    
    # Ethnic/geographic (be careful with these, but CLIP may use them)
    'geographic': [
        'Northern European', 'Southern European', 'Mediterranean',
        'East Asian', 'South Asian', 'Southeast Asian',
        'Latin American', 'Middle Eastern',
        'African', 'Sub-Saharan African', 'North African',
        'Caucasian', 'Nordic', 'Celtic',
    ],
}


def create_candidate_prompts():
    """
    Create comprehensive list of candidate text prompts.
    
    Returns:
        dict: {category: [prompts]}
    """
    all_prompts = {}
    
    for category, words in SKIN_TONE_VOCABULARY.items():
        prompts = []
        
        # Create multiple prompt templates
        templates = [
            "a photo of a person with {} skin",
            "a portrait of someone with {} skin tone",
            "a face with {} complexion",
            "{} skin",
            "a person with {} skin color",
        ]
        
        for word in words:
            for template in templates:
                prompts.append(template.format(word))
        
        all_prompts[category] = prompts
    
    return all_prompts


##############################################################
# CLIP INTERROGATION
##############################################################

def interrogate_images(model, processor, images, candidate_texts, device, batch_size=32):
    """
    Find which candidate texts best match the images.
    
    Args:
        model: CLIP model
        processor: CLIP processor
        images: List of PIL Images
        candidate_texts: List of text candidates
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        dict: {text: similarity_score} for each image
    """
    model.eval()
    
    results = []
    
    with torch.no_grad():
        # Process images
        image_inputs = processor(images=images, return_tensors="pt", padding=True)
        pixel_values = image_inputs['pixel_values'].to(device)
        
        # Get image embeddings
        image_outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = image_outputs.pooler_output
        image_embeds = model.visual_projection(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        
        # Process texts in batches
        text_similarities = []
        
        for i in range(0, len(candidate_texts), batch_size):
            batch_texts = candidate_texts[i:i+batch_size]
            
            # Tokenize
            text_inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # Get text embeddings
            text_outputs = model.text_model(**text_inputs)
            text_embeds = text_outputs.pooler_output
            text_embeds = model.text_projection(text_embeds)
            text_embeds = F.normalize(text_embeds, dim=-1)
            
            # Compute similarity
            similarities = image_embeds @ text_embeds.T  # [num_images, batch_size]
            text_similarities.append(similarities.cpu())
        
        # Concatenate all similarities
        all_similarities = torch.cat(text_similarities, dim=1)  # [num_images, num_texts]
        
        # For each image, get top matching texts
        for img_idx in range(len(images)):
            img_sims = all_similarities[img_idx]
            top_k = 20  # Top 20 matches
            
            top_scores, top_indices = torch.topk(img_sims, k=top_k)
            
            matches = []
            for score, idx in zip(top_scores, top_indices):
                matches.append({
                    'text': candidate_texts[idx],
                    'similarity': score.item()
                })
            
            results.append(matches)
    
    return results


##############################################################
# ANALYSIS
##############################################################

def analyze_mst_vocabulary(mst_e_csv, mst_e_dir, model, processor, device):
    """
    Analyze what words CLIP associates with each MST level.
    
    Returns:
        dict: {mst_level: {category: top_words}}
    """
    # Load MST-E annotations
    df = pd.read_csv(mst_e_csv)
    
    # Create candidate prompts
    print("[Creating] Candidate vocabulary...")
    candidate_prompts = create_candidate_prompts()
    
    # Flatten all prompts
    all_texts = []
    text_to_category = {}
    for category, prompts in candidate_prompts.items():
        for prompt in prompts:
            all_texts.append(prompt)
            text_to_category[prompt] = category
    
    print(f"[Vocabulary] {len(all_texts)} candidate descriptions")
    
    # Group images by MST level
    mst_images = defaultdict(list)
    mst_paths = defaultdict(list)
    
    for _, row in df.iterrows():
        mst = int(row['mst_label'])
        img_path = Path(mst_e_dir) / row['image_path'] if 'image_path' in row else Path(mst_e_dir) / row[df.columns[0]]
        
        if img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                mst_images[mst].append(img)
                mst_paths[mst].append(str(img_path))
            except Exception as e:
                print(f"[WARN] Failed to load {img_path}: {e}")
    
    print(f"[Loaded] Images for MST levels: {sorted(mst_images.keys())}")
    
    # Interrogate each MST level
    results = {}
    
    for mst in sorted(mst_images.keys()):
        images = mst_images[mst]
        print(f"\n[MST {mst}] Interrogating {len(images)} images...")
        
        # Get top matches for all images
        matches = interrogate_images(model, processor, images, all_texts, device)
        
        # Aggregate scores across all images in this MST level
        text_scores = defaultdict(list)
        for img_matches in matches:
            for match in img_matches:
                text_scores[match['text']].append(match['similarity'])
        
        # Average scores
        text_avg_scores = {
            text: np.mean(scores)
            for text, scores in text_scores.items()
        }
        
        # Sort by score
        sorted_texts = sorted(text_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Group by category
        category_results = defaultdict(list)
        for text, score in sorted_texts[:50]:  # Top 50
            category = text_to_category[text]
            category_results[category].append({
                'text': text,
                'score': score
            })
        
        results[mst] = {
            'top_overall': [
                {'text': text, 'score': score}
                for text, score in sorted_texts[:20]
            ],
            'by_category': dict(category_results)
        }
        
        # Print summary
        print(f"\n[MST {mst}] Top 10 descriptions:")
        for i, (text, score) in enumerate(sorted_texts[:10], 1):
            print(f"  {i:2d}. {text:60s} ({score:.3f})")
    
    return results


##############################################################
# PROMPT GENERATION
##############################################################

def generate_optimal_prompts(analysis_results, num_classes=4):
    """
    Use interrogation results to generate optimal prompts.
    
    Args:
        analysis_results: Output from analyze_mst_vocabulary
        num_classes: Number of classes (4 or 5)
    
    Returns:
        dict: Suggested prompts for each class
    """
    # Group MST levels into classes
    if num_classes == 4:
        mst_to_class = {
            1: 0, 2: 0, 3: 0,
            4: 1, 5: 1,
            6: 2, 7: 2,
            8: 3, 9: 3, 10: 3
        }
        class_names = ["MST 1-3", "MST 4-5", "MST 6-7", "MST 8-10"]
    else:  # 5 classes
        mst_to_class = {
            1: 0, 2: 0,
            3: 1, 4: 1,
            5: 2, 6: 2,
            7: 3, 8: 3,
            9: 4, 10: 4
        }
        class_names = ["MST 1-2", "MST 3-4", "MST 5-6", "MST 7-8", "MST 9-10"]
    
    # Aggregate top texts for each class
    class_texts = defaultdict(lambda: defaultdict(float))
    
    for mst, results in analysis_results.items():
        class_idx = mst_to_class.get(mst)
        if class_idx is None:
            continue
        
        for item in results['top_overall'][:20]:
            text = item['text']
            score = item['score']
            class_texts[class_idx][text] += score
    
    # Get top prompts for each class
    optimal_prompts = {}
    
    for class_idx in range(num_classes):
        # Sort by aggregated score
        sorted_texts = sorted(
            class_texts[class_idx].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top 10-12 unique prompts
        prompts = [text for text, score in sorted_texts[:12]]
        
        optimal_prompts[class_names[class_idx]] = prompts
    
    return optimal_prompts


##############################################################
# MAIN
##############################################################

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # Load CLIP
    print(f"\n[CLIP] Loading {args.clip_model}...")
    model = CLIPModel.from_pretrained(args.clip_model).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    print("[CLIP] Loaded successfully")
    
    # Analyze MST-E dataset
    print("\n" + "="*70)
    print("CLIP INTERROGATION: Discovering Skin Tone Vocabulary")
    print("="*70)
    
    results = analyze_mst_vocabulary(
        args.mst_e_csv,
        args.mst_e_dir,
        model,
        processor,
        device
    )
    
    # Generate optimal prompts
    print("\n" + "="*70)
    print("GENERATING OPTIMAL PROMPTS")
    print("="*70)
    
    for num_classes in [4, 5]:
        print(f"\n{num_classes}-Class Grouping:")
        optimal_prompts = generate_optimal_prompts(results, num_classes=num_classes)
        
        for class_name, prompts in optimal_prompts.items():
            print(f"\n  {class_name}:")
            for i, prompt in enumerate(prompts[:5], 1):
                print(f"    {i}. {prompt}")
    
    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output
        output = {
            'analysis': results,
            'optimal_prompts_4class': generate_optimal_prompts(results, num_classes=4),
            'optimal_prompts_5class': generate_optimal_prompts(results, num_classes=5),
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[Saved] Results: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nCLIP's Internal Skin Tone Vocabulary Discovered!")
    print("\nNext steps:")
    print("1. Review the optimal prompts above")
    print("2. Update get_text_prompts() in clip_hf_skin_tone.py")
    print("3. Re-run zero-shot test")
    print("\nExpected improvement:")
    print("  Current: 41% overall, MST 3-4 at 1.88%")
    print("  With CLIP-discovered prompts: 55-65% overall, MST 3-4 at 40-60%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interrogate CLIP to discover optimal skin tone vocabulary"
    )
    
    parser.add_argument("--mst_e_csv", required=True,
                       help="Path to MST-E annotations CSV")
    parser.add_argument("--mst_e_dir", required=True,
                       help="Path to MST-E image directory")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
                       help="HuggingFace CLIP model name")
    parser.add_argument("--output_json",
                       help="Path to save results JSON")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU device")
    
    args = parser.parse_args()
    main(args)


# python clip_interrogator.py ^
#   --mst_e_csv "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE\annotations.csv" ^
#   --mst_e_dir "G:\Thesis\MonkSkinTone_Dataset\Segmented_MSTE" ^
#   --output_json "clip_vocabulary.json" ^
#   --clip_model "openai/clip-vit-base-patch32"