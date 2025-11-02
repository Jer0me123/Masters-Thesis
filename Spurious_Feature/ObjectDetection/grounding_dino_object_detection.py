import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from typing import List, Tuple, Dict, Any
import torch
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# --- Global Configuration (Moved inside detect_objects, but keeping constants here) ---
MAX_CLASSES_PER_CHUNK = 80
BOX_THRESHOLD = 0.25      
TEXT_THRESHOLD = 0.5

# Define the full Open Images class list (as provided in the prompt)
# NOTE: This list is passed to detect_objects via a simplified mechanism now
OPEN_IMAGES_CLASSES = [
    "Accordion", "Adhesive tape", "Aircraft", "Airplane", "Alarm clock", "Alpaca", "Ambulance", 
    "Animal", "Ant", "Antelope", "Apple", "Armadillo", "Artichoke", "Auto part", "Axe", 
    "Backpack", "Bagel", "Baked goods", "Balance beam", "Ball", "Balloon", "Banana", 
    "Band-aid", "Banjo", "Barge", "Barrel", "Baseball bat", "Baseball glove", "Bat (Animal)", 
    "Bathroom accessory", "Bathroom cabinet", "Bathtub", "Beaker", "Bear", "Bed", "Bee", 
    "Beehive", "Beer", "Beetle", "Bell pepper", "Belt", "Bench", "Bicycle", "Bicycle helmet", 
    "Bicycle wheel", "Bidet", "Billboard", "Billiard table", "Binoculars", "Bird", "Blender", 
    "Blue jay", "Boat", "Bomb", "Book", "Bookcase", "Boot", "Bottle", "Bottle opener", 
    "Bow and arrow", "Bowl", "Bowling equipment", "Box", "Boy", "Brassiere", "Bread", 
    "Briefcase", "Broccoli", "Bronze sculpture", "Brown bear", "Building", "Bull", "Burrito", 
    "Bus", "Bust", "Butterfly", "Cabbage", "Cabinetry", "Cake", "Cake stand", "Calculator", 
    "Camel", "Camera", "Can opener", "Canary", "Candle", "Candy", "Cannon", "Canoe", 
    "Cantaloupe", "Car", "Carnivore", "Carrot", "Cart", "Cassette deck", "Castle", "Cat", 
    "Cat furniture", "Caterpillar", "Cattle", "Ceiling fan", "Cello", "Centipede", "Chainsaw", 
    "Chair", "Cheese", "Cheetah", "Chest of drawers", "Chicken", "Chime", "Chisel", 
    "Chopsticks", "Christmas tree", "Clock", "Closet", "Clothing", "Coat", "Cocktail", 
    "Cocktail shaker", "Coconut", "Coffee", "Coffee cup", "Coffee table", "Coffeemaker", 
    "Coin", "Common fig", "Common sunflower", "Computer keyboard", "Computer monitor", 
    "Computer mouse", "Container", "Convenience store", "Cookie", "Cooking spray", 
    "Corded phone", "Cosmetics", "Couch", "Countertop", "Cowboy hat", "Crab", "Cream", 
    "Cricket ball", "Crocodile", "Croissant", "Crown", "Crutch", "Cucumber", "Cupboard", 
    "Curtain", "Cutting board", "Dagger", "Dairy Product", "Deer", "Desk", "Dessert", 
    "Diaper", "Dice", "Digital clock", "Dinosaur", "Dishwasher", "Dog", "Dog bed", "Doll", 
    "Dolphin", "Door", "Door handle", "Donut", "Dragonfly", "Drawer", "Dress", "Drill (Tool)", 
    "Drink", "Drinking straw", "Drum", "Duck", "Dumbbell", "Eagle", "Earrings", "Egg (Food)", 
    "Elephant", "Envelope", "Eraser", "Face powder", "Facial tissue holder", "Falcon", 
    "Fashion accessory", "Fast food", "Fax", "Fedora", "Filing cabinet", "Fire hydrant", 
    "Fireplace", "Fish", "Flag", "Flashlight", "Flower", "Flowerpot", "Flute", "Flying disc", 
    "Food", "Food processor", "Football", "Football helmet", "Footwear", "Fork", "Fountain", 
    "Fox", "French fries", "French horn", "Frog", "Fruit", "Frying pan", "Furniture", 
    "Garden Asparagus", "Gas stove", "Giraffe", "Girl", "Glasses", "Glove", "Goat", 
    "Goggles", "Goldfish", "Golf ball", "Golf cart", "Gondola", "Goose", "Grape", 
    "Grapefruit", "Grinder", "Guacamole", "Guitar", "Hair dryer", "Hair spray", "Hamburger", 
    "Hammer", "Hamster", "Hand dryer", "Handbag", "Handgun", "Harbor seal", "Harmonica", 
    "Harp", "Harpsichord", "Hat", "Headphones", "Heater", "Hedgehog", "Helicopter", 
    "Helmet", "High heels", "Hiking equipment", "Hippopotamus", "Home appliance", 
    "Honeycomb", "Horizontal bar", "Horse", "Hot dog", "House", "Houseplant", "Human arm", 
    "Human beard", "Human body", "Human ear", "Human eye", "Human face", "Human foot", 
    "Human hair", "Human hand", "Human head", "Human leg", "Human mouth", "Human nose", 
    "Humidifier", "Ice cream", "Indoor rower", "Infant bed", "Insect", "Invertebrate", 
    "Ipod", "Isopod", "Jacket", "Jacuzzi", "Jaguar (Animal)", "Jeans", "Jellyfish", 
    "Jet ski", "Jug", "Juice", "Kangaroo", "Kettle", "Kitchen & dining room table", 
    "Kitchen appliance", "Kitchen knife", "Kitchen utensil", "Kitchenware", "Kite", 
    "Knife", "Koala", "Ladder", "Ladle", "Ladybug", "Lamp", "Land vehicle", "Lantern", 
    "Laptop", "Lavender (Plant)", "Lemon", "Leopard", "Light bulb", "Light switch", 
    "Lighthouse", "Lily", "Limousine", "Lion", "Lipstick", "Lizard", "Lobster", "Loveseat", 
    "Luggage and bags", "Lynx", "Magpie", "Mammal", "Man", "Mango", "Maple", "Maracas", 
    "Marine invertebrates", "Marine mammal", "Measuring cup", "Mechanical fan", 
    "Medical equipment", "Microphone", "Microwave oven", "Milk", "Miniskirt", "Mirror", 
    "Missile", "Mixer", "Mixing bowl", "Mobile phone", "Monkey", "Moths and butterflies", 
    "Motorcycle", "Mouse", "Muffin", "Mug", "Mule", "Mushroom", "Musical instrument", 
    "Musical keyboard", "Nail (Construction)", "Necklace", "Nightstand", "Oboe", 
    "Office building", "Office supplies", "Orange", "Organ (Musical Instrument)", "Ostrich", 
    "Otter", "Oven", "Owl", "Oyster", "Paddle", "Palm tree", "Pancake", "Panda", 
    "Paper cutter", "Paper towel", "Parachute", "Parking meter", "Parrot", "Pasta", 
    "Pastry", "Peach", "Pear", "Pen", "Pencil case", "Pencil sharpener", "Penguin", 
    "Perfume", "Person", "Personal care", "Personal flotation device", "Piano", 
    "Picnic basket", "Picture frame", "Pig", "Pillow", "Pineapple", "Pitcher (Container)", 
    "Pizza", "Pizza cutter", "Plant", "Plastic bag", "Plate", "Platter", "Plumbing fixture", 
    "Polar bear", "Pomegranate", "Popcorn", "Porch", "Porcupine", "Poster", "Potato", 
    "Power plugs and sockets", "Pressure cooker", "Pretzel", "Printer", "Pumpkin", 
    "Punching bag", "Rabbit", "Raccoon", "Racket", "Radish", "Ratchet (Device)", "Raven", 
    "Rays and skates", "Red panda", "Refrigerator", "Remote control", "Reptile", 
    "Rhinoceros", "Rifle", "Ring binder", "Rocket", "Roller skates", "Rose", "Rugby ball", 
    "Ruler", "Salad", "Salt and pepper shakers", "Sandal", "Sandwich", "Saucer", 
    "Saxophone", "Scale", "Scarf", "Scissors", "Scoreboard", "Scorpion", "Screwdriver", 
    "Sculpture", "Sea lion", "Sea turtle", "Seafood", "Seahorse", "Seat belt", "Segway", 
    "Serving tray", "Sewing machine", "Shark", "Sheep", "Shelf", "Shellfish", "Shirt", 
    "Shorts", "Shotgun", "Shower", "Shrimp", "Sink", "Skateboard", "Ski", "Skirt", "Skull", 
    "Skunk", "Skyscraper", "Slow cooker", "Snack", "Snail", "Snake", "Snowboard", 
    "Snowman", "Snowmobile", "Snowplow", "Soap dispenser", "Sock", "Sofa bed", "Sombrero", 
    "Sparrow", "Spatula", "Spice rack", "Spider", "Spoon", "Sports equipment", 
    "Sports uniform", "Squash (Plant)", "Squid", "Squirrel", "Stairs", "Stapler", 
    "Starfish", "Stationary bicycle", "Stethoscope", "Stool", "Stop sign", "Strawberry", 
    "Street light", "Stretcher", "Studio couch", "Submarine", "Submarine sandwich", 
    "Suit", "Suitcase", "Sun hat", "Sunglasses", "Surfboard", "Sushi", "Swan", "Swim cap", 
    "Swimming pool", "Swimwear", "Sword", "Syringe", "Table", "Table tennis racket", 
    "Tablet computer", "Tableware", "Taco", "Tank", "Tap", "Tart", "Taxi", "Tea", 
    "Teapot", "Teddy bear", "Telephone", "Television", "Tennis ball", "Tennis racket", 
    "Tent", "Tiara", "Tick", "Tie", "Tiger", "Tin can", "Tire", "Toaster", "Toilet", 
    "Toilet paper", "Tomato", "Tool", "Toothbrush", "Torch", "Tortoise", "Towel", 
    "Tower", "Toy", "Traffic light", "Traffic sign", "Train", "Training bench", 
    "Treadmill", "Tree", "Tree house", "Tripod", "Trombone", "Trousers", "Truck", 
    "Trumpet", "Turkey", "Turtle", "Umbrella", "Unicycle", "Van", "Vase", "Vegetable", 
    "Vehicle", "Vehicle registration plate", "Violin", "Volleyball (Ball)", "Waffle", 
    "Waffle iron", "Wall clock", "Wardrobe", "Washing machine", "Waste container", 
    "Watch", "Watercraft", "Watermelon", "Weapon", "Whale", "Wheel", "Wheelchair", 
    "Whisk", "Whiteboard", "Willow", "Window", "Window blind", "Wine", "Wine glass", 
    "Wine rack", "Winter melon", "Wok", "Woman", "Wood-burning stove", "Woodpecker", 
    "Worm", "Wrench", "Zebra", "Zucchini"
]


# ---------------- Worker Function ----------------
def draw_and_save_image(
    image: Image.Image, 
    instances_info: List[Dict[str, Any]], 
    save_path: str, 
    color_map: Dict[str, Tuple[int, int, int]]
) -> str:
    """
    Worker function to perform visualization and saving using PIL/OpenCV.
    """
    # Convert PIL Image to BGR for OpenCV drawing
    img_np = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    
    for instance in instances_info:
        # Box format: [xmin, ymin, xmax, ymax]
        box = instance["box"]
        label = instance["label"]
        score = instance["score"]

        # Get color based on label or use a default if not found
        color = color_map.get(label, (0, 255, 0))  # Default to green (BGR)

        # Draw box
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(img_np, pt1, pt2, color, 2)

        # Draw label
        text = f"{label}: {score:.2f}"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(img_np, (pt1[0], pt1[1] - h - 10), (pt1[0] + w, pt1[1]), color, -1)
        
        # Draw text
        cv2.putText(img_np, text, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite(save_path, img_np)
    return save_path

# ---------------- Dataset ----------------
class ImageDataset(Dataset):
    def __init__(self, image_dir, extensions=(".jpg", ".jpeg", ".png")):
        image_dir = Path(image_dir)
        # Use glob to find all images
        self.image_paths = sorted([str(p) for ext in extensions for p in image_dir.glob(f"*{ext}")])
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        # Load image as PIL Image for Grounding DINO
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        # The Dataloader returns the PIL Image, path, and size
        return img, path, (orig_h, orig_w) # (H, W)

def collate_images(batch):
    # PIL Images are collected, no numpy conversion or transformation here
    imgs, paths, orig_sizes = zip(*batch)
    return list(imgs), list(paths), list(orig_sizes)

# ---------------- Grounding DINO Helper Functions ----------------
def structure_labels_for_grounding_dino(labels: List[str]) -> str:
    """Formats a list of labels into the text prompt required by Grounding DINO."""
    # Convert to lowercase and strip to ensure consistency
    cleaned = [lbl.strip().strip(".").lower() for lbl in labels]
    # Join with dot-space and ensure final dot
    prompt = ". ".join(cleaned)
    if not prompt.endswith("."):
        prompt += "."
    return prompt

def get_class_chunks(classes: List[str], max_chunk_size: int) -> List[List[str]]:
    """Splits the full class list into smaller chunks for inference."""
    return [classes[i:i + max_chunk_size] for i in range(0, len(classes), max_chunk_size)]

# ---------------- Main Object Detection Function ----------------
def detect_objects(
    image_dir: str,
    output_dir: str,
    model_id: str,      
    color_json: str,
    batch_size: int = 4,
    num_workers: int = 4,
    conf_thresh: float = 0.5,
    resume: bool = True
):
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. This will be slow.")

    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    # --- 1. Load Model and Processor ---
    print(f"Loading Grounding DINO model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    # --- 2. Prepare Classes and Chunks ---
    # In a real scenario, color_json might be used, but here we hardcode OPEN_IMAGES_CLASSES
    # For compatibility, we load the color map from the provided JSON
    with open(color_json, "r") as f:
        color_map = {k.title(): tuple(v) for k, v in json.load(f).items()} # Ensure title-case matches post-processing
    
    full_class_list = OPEN_IMAGES_CLASSES
    class_chunks = get_class_chunks(full_class_list, MAX_CLASSES_PER_CHUNK)
    print(f"Loaded {len(full_class_list)} classes in {len(class_chunks)} chunks for Grounding DINO inference.")

    # --- 3. Prepare Dataset ---
    dataset = ImageDataset(image_dir)
    
    # Apply resume logic
    if resume:
        processed = {p.stem for p in output_dir.glob("*.png")}
        dataset.image_paths = [p for p in dataset.image_paths if Path(p).stem not in processed]
        print(f"Resuming: {len(dataset.image_paths)} images left to process")
    
    total_images_to_process = len(dataset.image_paths)
    if not dataset.image_paths:
        print("All images already processed. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, collate_fn=collate_images)

    all_futures: List[Future] = []
    
    # --- 4. Batch Inference Loop (GPU) ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
         tqdm(total=len(loader), desc="G-DINO Inference (GPU) & Submission (CPU)") as pbar:

        for batch_idx, (batch_images, batch_paths, batch_orig_sizes) in enumerate(loader):
            
            # Use original PIL images for the model input
            # Original sizes are (H, W) but batch_target_sizes needs (W, H) for post-processing
            batch_target_sizes = [img.size[::-1] for img in batch_images]
            
            # --- 5. Chunked Inference ---
            batch_detections: List[List[Dict[str, Any]]] = [[] for _ in range(len(batch_images))]

            for chunk_idx, class_chunk in enumerate(class_chunks):
                # 5a. Create prompt for the chunk
                prompt = structure_labels_for_grounding_dino(class_chunk)
                prompts = [prompt] * len(batch_images)

                # 5b. Prepare batch input and move to device
                # The processor handles image resizing and normalization internally
                inputs = processor(images=batch_images, text=prompts, return_tensors="pt").to(device)

                # 5c. Run inference
                with torch.no_grad():
                    # Use float16 on CUDA for speed/memory efficiency
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                        outputs = model(**inputs)

                # 5d. Post-process outputs
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=TEXT_THRESHOLD, target_sizes=batch_target_sizes
                )

                # 5e. Collect detections for the batch
                for img_idx, res in enumerate(results):
                    # Filter by box score and collect all valid detections for this image across all chunks
                    for score, label, box in zip(res["scores"], res["text_labels"], res["boxes"]):
                        if score.item() >= conf_thresh:
                            # Convert box to standard list of ints [xmin, ymin, xmax, ymax]
                            box_int = [int(round(v.item())) for v in box] 
                            
                            # Standardize label casing for lookup (Grounding DINO outputs lowercase/space-separated)
                            standard_label = " ".join(label.split()).title() 
                            
                            batch_detections[img_idx].append({
                                "score": score.item(),
                                "label": standard_label,
                                "box": box_int,
                                "image_path": batch_paths[img_idx]
                            })
            
            # --- 6. Submit Save Tasks (CPU) ---
            for img_idx, image in enumerate(batch_images):
                save_path = output_dir / (Path(batch_paths[img_idx]).stem + ".png")
                
                # Filter for resume logic
                if save_path.exists() and resume:
                    continue
                
                # Submit save task to the process pool
                future = executor.submit(
                    draw_and_save_image, 
                    image, 
                    batch_detections[img_idx], 
                    str(save_path),
                    color_map
                )
                all_futures.append(future)

            pbar.update(1)

        # Wait for remaining saves
        if all_futures:
            # We use the length of all futures to track remaining saves
            for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Saving final results (CPU)"):
                future.result()
                
    torch.cuda.empty_cache()
    print("Object detection completed successfully!")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Open-Vocabulary Object Detection using Grounding DINO")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detected images.")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny", help="The Grounding DINO model ID.")
    parser.add_argument("--color_json", type=str, required=True, help="Path to the LVIS/Open Images color map JSON.")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images per batch for GPU inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Bounding Box score threshold.")
    args = parser.parse_args()
    
    detect_objects(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        color_json=args.color_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_thresh=args.conf_thresh
    )

# ---------------- Example Usage ----------------
# GDino Tiny model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\grounding_dino_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-GDINO-detected" --model_id "IDEA-Research/grounding-dino-tiny" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\LVIS_color_map.json" --batch_size 4 --num_workers 8 --conf_thresh 0.5

# Around 80hrs for 10k images using GDino Tiny model on RTX 3060ti, due to having to process each image multiple times for all class chunks.