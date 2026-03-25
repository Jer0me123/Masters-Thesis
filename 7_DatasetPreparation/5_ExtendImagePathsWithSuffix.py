import argparse
import json
from pathlib import Path

def add_suffix(path: str, suffix: str) -> Path:
    """
    Insert suffix before file extension.
    """
    p = Path(path)
    return p.with_name(p.stem + suffix + p.suffix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_json", required=True, type=Path)
    parser.add_argument("--out_json", required=True, type=Path)
    parser.add_argument("--suffix", required=True,
                        help="Suffix to append before extension, e.g. _depth")
    parser.add_argument(
        "--base_path",
        required=True,
        type=Path,
        help="Base directory to prepend to image paths"
    )

    args = parser.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for split in ["train", "val", "test"]:
        for record in data[split]:
            # 1) append suffix
            p = add_suffix(record["image"], args.suffix)

            # 2) prepend base path
            full_path = args.base_path / p

            # store as string (JSON-friendly)
            record["image"] = str(full_path)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", args.out_json)

if __name__ == "__main__":
    main()

##################################################################################### Stable Diffusion #####################################################################################

######## GENDER SPLITS ########

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Base\splits_gender_face_stratified.json" ^
# --base_path "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Depth\splits_gender_face_stratified_depth.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Depth" ^
# --suffix "_depth"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\EdgeDetection\edges_canny\splits_gender_face_stratified_edge.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\EdgeDetection\edges_canny" ^
# --suffix "_edges"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\High&LowPassFilter\ideal\radius_40\high_pass\splits_gender_face_stratified_high.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\High&LowPassFilter\ideal\radius_40\high_pass" ^
# --suffix "_high"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\High&LowPassFilter\ideal\radius_40\low_pass\splits_gender_face_stratified_low.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\High&LowPassFilter\ideal\radius_40\low_pass" ^
# --suffix "_low"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\Full_NoBg\splits_gender_face_stratified_full_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\Full_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskSegm_NoBg\splits_gender_face_stratified_mask_segm_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskSegm_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskSegm\splits_gender_face_stratified_mask_segm.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskSegm" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskRect_NoBg\splits_gender_face_stratified_mask_rect_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskRect_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskRect\splits_gender_face_stratified_mask_rect.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskRect" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\SemanticSegmentation\splits_gender_face_stratified_segmentation.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\SemanticSegmentation" ^
# --suffix "_seg"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\pixel_shuffle\splits_gender_face_stratified_pixel_shuffle.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\pixel_shuffle" ^
# --suffix "_pixel"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps2\splits_gender_face_stratified_patch_shuffle_ps2.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps2" ^
# --suffix "_ps2"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps4\splits_gender_face_stratified_patch_shuffle_ps4.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps4" ^
# --suffix "_ps4"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps8\splits_gender_face_stratified_patch_shuffle_ps8.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps8" ^
# --suffix "_ps8"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps16\splits_gender_face_stratified_patch_shuffle_ps16.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps16" ^
# --suffix "_ps16"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\VAE\splits_gender_face_stratified_vae.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\VAE" ^
# --suffix "_vae"

######## SKINTONE SPLITS ########

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Base\splits_10mst_face_stratified.json" ^
# --base_path "E:\ImageRetrieval\StableDiffusionGeneratedImages\valid" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Depth\splits_10mst_face_stratified_depth.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Depth" ^
# --suffix "_depth"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\EdgeDetection\edges_canny\splits_10mst_face_stratified_edge.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\EdgeDetection\edges_canny" ^
# --suffix "_edges"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\High&LowPassFilter\ideal\radius_40\high_pass\splits_10mst_face_stratified_high.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\High&LowPassFilter\ideal\radius_40\high_pass" ^
# --suffix "_high"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\High&LowPassFilter\ideal\radius_40\low_pass\splits_10mst_face_stratified_low.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\High&LowPassFilter\ideal\radius_40\low_pass" ^
# --suffix "_low"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\Full_NoBg\splits_10mst_face_stratified_full_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\Full_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskSegm_NoBg\splits_10mst_face_stratified_mask_segm_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskSegm_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskSegm\splits_10mst_face_stratified_mask_segm.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskSegm" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskRect_NoBg\splits_10mst_face_stratified_mask_rect_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskRect_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Occlusion\MaskRect\splits_10mst_face_stratified_mask_rect.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Occlusion\MaskRect" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\SemanticSegmentation\splits_10mst_face_stratified_segmentation.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\SemanticSegmentation" ^
# --suffix "_seg"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\pixel_shuffle\splits_10mst_face_stratified_pixel_shuffle.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\pixel_shuffle" ^
# --suffix "_pixel"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps2\splits_10mst_face_stratified_patch_shuffle_ps2.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps2" ^
# --suffix "_ps2"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps4\splits_10mst_face_stratified_patch_shuffle_ps4.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps4" ^
# --suffix "_ps4"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps8\splits_10mst_face_stratified_patch_shuffle_ps8.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps8" ^
# --suffix "_ps8"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps16\splits_10mst_face_stratified_patch_shuffle_ps16.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\Shuffling&Colour\patch_shuffle_ps16" ^
# --suffix "_ps16"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\StableDiffusion\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\_SPLITS\VAE\splits_10mst_face_stratified_vae.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\StableDiffusionImages\VAE" ^
# --suffix "_vae"

##################################################################################### Professions 125k ISCO Aligned 1k Subset #####################################################################################

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Base\splits_gender_face_stratified.json" ^
# --base_path "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Depth\splits_gender_face_stratified_depth.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Depth" ^
# --suffix "_depth"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\EdgeDetection\edges_canny\splits_gender_face_stratified_edge.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\EdgeDetection\edges_canny" ^
# --suffix "_edges"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\High&LowPassFilter\ideal\radius_40\high_pass\splits_gender_face_stratified_high.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\High&LowPassFilter\ideal\radius_40\high_pass" ^
# --suffix "_high"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\High&LowPassFilter\ideal\radius_40\low_pass\splits_gender_face_stratified_low.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\High&LowPassFilter\ideal\radius_40\low_pass" ^
# --suffix "_low"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\Full_NoBg\splits_gender_face_stratified_full_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\Full_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskSegm_NoBg\splits_gender_face_stratified_mask_segm_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskSegm_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskSegm\splits_gender_face_stratified_mask_segm.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskSegm" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskRect_NoBg\splits_gender_face_stratified_mask_rect_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskRect_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskRect\splits_gender_face_stratified_mask_rect.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskRect" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\SemanticSegmentation\splits_gender_face_stratified_segmentation.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\SemanticSegmentation" ^
# --suffix "_seg"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\pixel_shuffle\splits_gender_face_stratified_pixel_shuffle.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\pixel_shuffle" ^
# --suffix "_pixel"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps2\splits_gender_face_stratified_patch_shuffle_ps2.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps2" ^
# --suffix "_ps2"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps4\splits_gender_face_stratified_patch_shuffle_ps4.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps4" ^
# --suffix "_ps4"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps8\splits_gender_face_stratified_patch_shuffle_ps8.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps8" ^
# --suffix "_ps8"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps16\splits_gender_face_stratified_patch_shuffle_ps16.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps16" ^
# --suffix "_ps16"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\VAE\splits_gender_face_stratified_vae.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\VAE" ^
# --suffix "_vae"

######## SKINTONE SPLITS ########

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Base\splits_10mst_face_stratified.json" ^
# --base_path "F:\ImageRetrieval\Professions_125k_ISCO_Aligned_1k_Subset" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Depth\splits_10mst_face_stratified_depth.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Depth" ^
# --suffix "_depth"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\EdgeDetection\edges_canny\splits_10mst_face_stratified_edge.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\EdgeDetection\edges_canny" ^
# --suffix "_edges"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\High&LowPassFilter\ideal\radius_40\high_pass\splits_10mst_face_stratified_high.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\High&LowPassFilter\ideal\radius_40\high_pass" ^
# --suffix "_high"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\High&LowPassFilter\ideal\radius_40\low_pass\splits_10mst_face_stratified_low.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\High&LowPassFilter\ideal\radius_40\low_pass" ^
# --suffix "_low"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\Full_NoBg\splits_10mst_face_stratified_full_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\Full_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskSegm_NoBg\splits_10mst_face_stratified_mask_segm_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskSegm_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskSegm\splits_10mst_face_stratified_mask_segm.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskSegm" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskRect_NoBg\splits_10mst_face_stratified_mask_rect_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskRect_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Occlusion\MaskRect\splits_10mst_face_stratified_mask_rect.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Occlusion\MaskRect" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\SemanticSegmentation\splits_10mst_face_stratified_segmentation.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\SemanticSegmentation" ^
# --suffix "_seg"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\pixel_shuffle\splits_10mst_face_stratified_pixel_shuffle.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\pixel_shuffle" ^
# --suffix "_pixel"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps2\splits_10mst_face_stratified_patch_shuffle_ps2.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps2" ^
# --suffix "_ps2"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps4\splits_10mst_face_stratified_patch_shuffle_ps4.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps4" ^
# --suffix "_ps4"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps8\splits_10mst_face_stratified_patch_shuffle_ps8.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps8" ^
# --suffix "_ps8"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\Shuffling&Colour\patch_shuffle_ps16\splits_10mst_face_stratified_patch_shuffle_ps16.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\Shuffling&Colour\patch_shuffle_ps16" ^
# --suffix "_ps16"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\Professions_125k_ISCO_Aligned_1k_Subset\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\_SPLITS\VAE\splits_10mst_face_stratified_vae.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\Professions_125k_ISCO_Aligned_1k_Subset\VAE" ^
# --suffix "_vae"


##################################################################################### Debiased Images #####################################################################################

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Base\splits_gender_face_stratified.json" ^
# --base_path "F:\ImageRetrieval\DebiasedImages" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Depth\splits_gender_face_stratified_depth.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Depth" ^
# --suffix "_depth"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\EdgeDetection\edges_canny\splits_gender_face_stratified_edge.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\EdgeDetection\edges_canny" ^
# --suffix "_edges"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\High&LowPassFilter\ideal\radius_40\high_pass\splits_gender_face_stratified_high.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\High&LowPassFilter\ideal\radius_40\high_pass" ^
# --suffix "_high"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\High&LowPassFilter\ideal\radius_40\low_pass\splits_gender_face_stratified_low.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\High&LowPassFilter\ideal\radius_40\low_pass" ^
# --suffix "_low"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\Full_NoBg\splits_gender_face_stratified_full_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\Full_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskSegm_NoBg\splits_gender_face_stratified_mask_segm_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskSegm_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskSegm\splits_gender_face_stratified_mask_segm.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskSegm" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskRect_NoBg\splits_gender_face_stratified_mask_rect_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskRect_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskRect\splits_gender_face_stratified_mask_rect.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskRect" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\SemanticSegmentation\splits_gender_face_stratified_segmentation.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\SemanticSegmentation" ^
# --suffix "_seg"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\pixel_shuffle\splits_gender_face_stratified_pixel_shuffle.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\pixel_shuffle" ^
# --suffix "_pixel"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps2\splits_gender_face_stratified_patch_shuffle_ps2.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps2" ^
# --suffix "_ps2"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps4\splits_gender_face_stratified_patch_shuffle_ps4.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps4" ^
# --suffix "_ps4"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps8\splits_gender_face_stratified_patch_shuffle_ps8.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps8" ^
# --suffix "_ps8"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps16\splits_gender_face_stratified_patch_shuffle_ps16.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps16" ^
# --suffix "_ps16"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_gender_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\VAE\splits_gender_face_stratified_vae.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\VAE" ^
# --suffix "_vae"

######## SKINTONE SPLITS ########

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Base\splits_10mst_face_stratified.json" ^
# --base_path "F:\ImageRetrieval\DebiasedImages" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Depth\splits_10mst_face_stratified_depth.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Depth" ^
# --suffix "_depth"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\EdgeDetection\edges_canny\splits_10mst_face_stratified_edge.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\EdgeDetection\edges_canny" ^
# --suffix "_edges"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\High&LowPassFilter\ideal\radius_40\high_pass\splits_10mst_face_stratified_high.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\High&LowPassFilter\ideal\radius_40\high_pass" ^
# --suffix "_high"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\High&LowPassFilter\ideal\radius_40\low_pass\splits_10mst_face_stratified_low.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\High&LowPassFilter\ideal\radius_40\low_pass" ^
# --suffix "_low"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\Full_NoBg\splits_10mst_face_stratified_full_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\Full_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskSegm_NoBg\splits_10mst_face_stratified_mask_segm_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskSegm_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskSegm\splits_10mst_face_stratified_mask_segm.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskSegm" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskRect_NoBg\splits_10mst_face_stratified_mask_rect_noBg.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskRect_NoBg" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Occlusion\MaskRect\splits_10mst_face_stratified_mask_rect.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Occlusion\MaskRect" ^
# --suffix ""

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\SemanticSegmentation\splits_10mst_face_stratified_segmentation.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\SemanticSegmentation" ^
# --suffix "_seg"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\pixel_shuffle\splits_10mst_face_stratified_pixel_shuffle.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\pixel_shuffle" ^
# --suffix "_pixel"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps2\splits_10mst_face_stratified_patch_shuffle_ps2.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps2" ^
# --suffix "_ps2"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps4\splits_10mst_face_stratified_patch_shuffle_ps4.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps4" ^
# --suffix "_ps4"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps8\splits_10mst_face_stratified_patch_shuffle_ps8.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps8" ^
# --suffix "_ps8"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\Shuffling&Colour\patch_shuffle_ps16\splits_10mst_face_stratified_patch_shuffle_ps16.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\Shuffling&Colour\patch_shuffle_ps16" ^
# --suffix "_ps16"

# python 5_ExtendImagePathsWithSuffix.py ^
# --in_json "UniversalSplits\Base\DebiasedImages\splits_10mst_face_stratified.json" ^
# --out_json "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\_SPLITS\VAE\splits_10mst_face_stratified_vae.json" ^
# --base_path "F:\ImageRetrieval\SpuriousFeatureImages\DebiasedImages\VAE" ^
# --suffix "_vae"