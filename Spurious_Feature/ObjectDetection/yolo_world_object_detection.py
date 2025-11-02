import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from typing import List, Dict, Any, Tuple, Set
import torch
import cv2
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import argparse
import gc
from ultralytics import YOLO
from ultralytics.engine.results import Results

OPEN_IMAGES_CLASSES = [
    "Accordion",
    "Adhesive tape",
    "Aircraft",
    "Airplane",
    "Alarm clock",
    "Alpaca",
    "Ambulance",
    "Animal",
    "Ant",
    "Antelope",
    "Apple",
    "Armadillo",
    "Artichoke",
    "Auto part",
    "Axe",
    "Backpack",
    "Bagel",
    "Baked goods",
    "Balance beam",
    "Ball",
    "Balloon",
    "Banana",
    "Band-aid",
    "Banjo",
    "Barge",
    "Barrel",
    "Baseball bat",
    "Baseball glove",
    "Bat (Animal)",
    "Bathroom accessory",
    "Bathroom cabinet",
    "Bathtub",
    "Beaker",
    "Bear",
    "Bed",
    "Bee",
    "Beehive",
    "Beer",
    "Beetle",
    "Bell pepper",
    "Belt",
    "Bench",
    "Bicycle",
    "Bicycle helmet",
    "Bicycle wheel",
    "Bidet",
    "Billboard",
    "Billiard table",
    "Binoculars",
    "Bird",
    "Blender",
    "Blue jay",
    "Boat",
    "Bomb",
    "Book",
    "Bookcase",
    "Boot",
    "Bottle",
    "Bottle opener",
    "Bow and arrow",
    "Bowl",
    "Bowling equipment",
    "Box",
    "Boy",
    "Brassiere",
    "Bread",
    "Briefcase",
    "Broccoli",
    "Bronze sculpture",
    "Brown bear",
    "Building",
    "Bull",
    "Burrito",
    "Bus",
    "Bust",
    "Butterfly",
    "Cabbage",
    "Cabinetry",
    "Cake",
    "Cake stand",
    "Calculator",
    "Camel",
    "Camera",
    "Can opener",
    "Canary",
    "Candle",
    "Candy",
    "Cannon",
    "Canoe",
    "Cantaloupe",
    "Car",
    "Carnivore",
    "Carrot",
    "Cart",
    "Cassette deck",
    "Castle",
    "Cat",
    "Cat furniture",
    "Caterpillar",
    "Cattle",
    "Ceiling fan",
    "Cello",
    "Centipede",
    "Chainsaw",
    "Chair",
    "Cheese",
    "Cheetah",
    "Chest of drawers",
    "Chicken",
    "Chime",
    "Chisel",
    "Chopsticks",
    "Christmas tree",
    "Clock",
    "Closet",
    "Clothing",
    "Coat",
    "Cocktail",
    "Cocktail shaker",
    "Coconut",
    "Coffee",
    "Coffee cup",
    "Coffee table",
    "Coffeemaker",
    "Coin",
    "Common fig",
    "Common sunflower",
    "Computer keyboard",
    "Computer monitor",
    "Computer mouse",
    "Container",
    "Convenience store",
    "Cookie",
    "Cooking spray",
    "Corded phone",
    "Cosmetics",
    "Couch",
    "Countertop",
    "Cowboy hat",
    "Crab",
    "Cream",
    "Cricket ball",
    "Crocodile",
    "Croissant",
    "Crown",
    "Crutch",
    "Cucumber",
    "Cupboard",
    "Curtain",
    "Cutting board",
    "Dagger",
    "Dairy Product",
    "Deer",
    "Desk",
    "Dessert",
    "Diaper",
    "Dice",
    "Digital clock",
    "Dinosaur",
    "Dishwasher",
    "Dog",
    "Dog bed",
    "Doll",
    "Dolphin",
    "Door",
    "Door handle",
    "Donut",
    "Dragonfly",
    "Drawer",
    "Dress",
    "Drill (Tool)",
    "Drink",
    "Drinking straw",
    "Drum",
    "Duck",
    "Dumbbell",
    "Eagle",
    "Earrings",
    "Egg (Food)",
    "Elephant",
    "Envelope",
    "Eraser",
    "Face powder",
    "Facial tissue holder",
    "Falcon",
    "Fashion accessory",
    "Fast food",
    "Fax",
    "Fedora",
    "Filing cabinet",
    "Fire hydrant",
    "Fireplace",
    "Fish",
    "Flag",
    "Flashlight",
    "Flower",
    "Flowerpot",
    "Flute",
    "Flying disc",
    "Food",
    "Food processor",
    "Football",
    "Football helmet",
    "Footwear",
    "Fork",
    "Fountain",
    "Fox",
    "French fries",
    "French horn",
    "Frog",
    "Fruit",
    "Frying pan",
    "Furniture",
    "Garden Asparagus",
    "Gas stove",
    "Giraffe",
    "Girl",
    "Glasses",
    "Glove",
    "Goat",
    "Goggles",
    "Goldfish",
    "Golf ball",
    "Golf cart",
    "Gondola",
    "Goose",
    "Grape",
    "Grapefruit",
    "Grinder",
    "Guacamole",
    "Guitar",
    "Hair dryer",
    "Hair spray",
    "Hamburger",
    "Hammer",
    "Hamster",
    "Hand dryer",
    "Handbag",
    "Handgun",
    "Harbor seal",
    "Harmonica",
    "Harp",
    "Harpsichord",
    "Hat",
    "Headphones",
    "Heater",
    "Hedgehog",
    "Helicopter",
    "Helmet",
    "High heels",
    "Hiking equipment",
    "Hippopotamus",
    "Home appliance",
    "Honeycomb",
    "Horizontal bar",
    "Horse",
    "Hot dog",
    "House",
    "Houseplant",
    "Human arm",
    "Human beard",
    "Human body",
    "Human ear",
    "Human eye",
    "Human face",
    "Human foot",
    "Human hair",
    "Human hand",
    "Human head",
    "Human leg",
    "Human mouth",
    "Human nose",
    "Humidifier",
    "Ice cream",
    "Indoor rower",
    "Infant bed",
    "Insect",
    "Invertebrate",
    "Ipod",
    "Isopod",
    "Jacket",
    "Jacuzzi",
    "Jaguar (Animal)",
    "Jeans",
    "Jellyfish",
    "Jet ski",
    "Jug",
    "Juice",
    "Kangaroo",
    "Kettle",
    "Kitchen & dining room table",
    "Kitchen appliance",
    "Kitchen knife",
    "Kitchen utensil",
    "Kitchenware",
    "Kite",
    "Knife",
    "Koala",
    "Ladder",
    "Ladle",
    "Ladybug",
    "Lamp",
    "Land vehicle",
    "Lantern",
    "Laptop",
    "Lavender (Plant)",
    "Lemon",
    "Leopard",
    "Light bulb",
    "Light switch",
    "Lighthouse",
    "Lily",
    "Limousine",
    "Lion",
    "Lipstick",
    "Lizard",
    "Lobster",
    "Loveseat",
    "Luggage and bags",
    "Lynx",
    "Magpie",
    "Mammal",
    "Man",
    "Mango",
    "Maple",
    "Maracas",
    "Marine invertebrates",
    "Marine mammal",
    "Measuring cup",
    "Mechanical fan",
    "Medical equipment",
    "Microphone",
    "Microwave oven",
    "Milk",
    "Miniskirt",
    "Mirror",
    "Missile",
    "Mixer",
    "Mixing bowl",
    "Mobile phone",
    "Monkey",
    "Moths and butterflies",
    "Motorcycle",
    "Mouse",
    "Muffin",
    "Mug",
    "Mule",
    "Mushroom",
    "Musical instrument",
    "Musical keyboard",
    "Nail (Construction)",
    "Necklace",
    "Nightstand",
    "Oboe",
    "Office building",
    "Office supplies",
    "Orange",
    "Organ (Musical Instrument)",
    "Ostrich",
    "Otter",
    "Oven",
    "Owl",
    "Oyster",
    "Paddle",
    "Palm tree",
    "Pancake",
    "Panda",
    "Paper cutter",
    "Paper towel",
    "Parachute",
    "Parking meter",
    "Parrot",
    "Pasta",
    "Pastry",
    "Peach",
    "Pear",
    "Pen",
    "Pencil case",
    "Pencil sharpener",
    "Penguin",
    "Perfume",
    "Person",
    "Personal care",
    "Personal flotation device",
    "Piano",
    "Picnic basket",
    "Picture frame",
    "Pig",
    "Pillow",
    "Pineapple",
    "Pitcher (Container)",
    "Pizza",
    "Pizza cutter",
    "Plant",
    "Plastic bag",
    "Plate",
    "Platter",
    "Plumbing fixture",
    "Polar bear",
    "Pomegranate",
    "Popcorn",
    "Porch",
    "Porcupine",
    "Poster",
    "Potato",
    "Power plugs and sockets",
    "Pressure cooker",
    "Pretzel",
    "Printer",
    "Pumpkin",
    "Punching bag",
    "Rabbit",
    "Raccoon",
    "Racket",
    "Radish",
    "Ratchet (Device)",
    "Raven",
    "Rays and skates",
    "Red panda",
    "Refrigerator",
    "Remote control",
    "Reptile",
    "Rhinoceros",
    "Rifle",
    "Ring binder",
    "Rocket",
    "Roller skates",
    "Rose",
    "Rugby ball",
    "Ruler",
    "Salad",
    "Salt and pepper shakers",
    "Sandal",
    "Sandwich",
    "Saucer",
    "Saxophone",
    "Scale",
    "Scarf",
    "Scissors",
    "Scoreboard",
    "Scorpion",
    "Screwdriver",
    "Sculpture",
    "Sea lion",
    "Sea turtle",
    "Seafood",
    "Seahorse",
    "Seat belt",
    "Segway",
    "Serving tray",
    "Sewing machine",
    "Shark",
    "Sheep",
    "Shelf",
    "Shellfish",
    "Shirt",
    "Shorts",
    "Shotgun",
    "Shower",
    "Shrimp",
    "Sink",
    "Skateboard",
    "Ski",
    "Skirt",
    "Skull",
    "Skunk",
    "Skyscraper",
    "Slow cooker",
    "Snack",
    "Snail",
    "Snake",
    "Snowboard",
    "Snowman",
    "Snowmobile",
    "Snowplow",
    "Soap dispenser",
    "Sock",
    "Sofa bed",
    "Sombrero",
    "Sparrow",
    "Spatula",
    "Spice rack",
    "Spider",
    "Spoon",
    "Sports equipment",
    "Sports uniform",
    "Squash (Plant)",
    "Squid",
    "Squirrel",
    "Stairs",
    "Stapler",
    "Starfish",
    "Stationary bicycle",
    "Stethoscope",
    "Stool",
    "Stop sign",
    "Strawberry",
    "Street light",
    "Stretcher",
    "Studio couch",
    "Submarine",
    "Submarine sandwich",
    "Suit",
    "Suitcase",
    "Sun hat",
    "Sunglasses",
    "Surfboard",
    "Sushi",
    "Swan",
    "Swim cap",
    "Swimming pool",
    "Swimwear",
    "Sword",
    "Syringe",
    "Table",
    "Table tennis racket",
    "Tablet computer",
    "Tableware",
    "Taco",
    "Tank",
    "Tap",
    "Tart",
    "Taxi",
    "Tea",
    "Teapot",
    "Teddy bear",
    "Telephone",
    "Television",
    "Tennis ball",
    "Tennis racket",
    "Tent",
    "Tiara",
    "Tick",
    "Tie",
    "Tiger",
    "Tin can",
    "Tire",
    "Toaster",
    "Toilet",
    "Toilet paper",
    "Tomato",
    "Tool",
    "Toothbrush",
    "Torch",
    "Tortoise",
    "Towel",
    "Tower",
    "Toy",
    "Traffic light",
    "Traffic sign",
    "Train",
    "Training bench",
    "Treadmill",
    "Tree",
    "Tree house",
    "Tripod",
    "Trombone",
    "Trousers",
    "Truck",
    "Trumpet",
    "Turkey",
    "Turtle",
    "Umbrella",
    "Unicycle",
    "Van",
    "Vase",
    "Vegetable",
    "Vehicle",
    "Vehicle registration plate",
    "Violin",
    "Volleyball (Ball)",
    "Waffle",
    "Waffle iron",
    "Wall clock",
    "Wardrobe",
    "Washing machine",
    "Waste container",
    "Watch",
    "Watercraft",
    "Watermelon",
    "Weapon",
    "Whale",
    "Wheel",
    "Wheelchair",
    "Whisk",
    "Whiteboard",
    "Willow",
    "Window",
    "Window blind",
    "Wine",
    "Wine glass",
    "Wine rack",
    "Winter melon",
    "Wok",
    "Woman",
    "Wood-burning stove",
    "Woodpecker",
    "Worm",
    "Wrench",
    "Zebra",
    "Zucchini"]

# ----------------- Helper Functions -----------------

# --- Worker Function (CPU-intensive: Visualization and JSON Save) ---
def draw_and_save_image(
    image: Image.Image, 
    instances_info: List[Dict[str, Any]], 
    save_path_img: str, 
    save_path_json: str,
    color_map: Dict[str, Tuple[int, int, int]],
    white_background: bool = False
) -> Tuple[str, str]:
    """
    Core function to perform visualization (CV2/Numpy) and save detections to JSON.
    This function is run inside the ProcessPoolExecutor.
    """
    # Convert PIL Image to BGR for OpenCV drawing
    if white_background:
        width, height = image.size
        # np.full creates a new array filled with the specified value (255 for white)
        # BGR format is (Height, Width, Channels)
        img_np = np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        # Original logic
        img_np = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    
    # Define new text parameters
    FONT_SCALE = 0.4    # Reduced font size for smaller labels
    FONT_THICKNESS = 1
    BOX_THICKNESS = 2
    TEXT_PADDING = 2
    
    # ----------------- Data for JSON -----------------
    json_data = {}
    
    for instance in instances_info:
        # Box format: [xmin, ymin, xmax, ymax]
        box = instance["box"]
        label = instance["label"]
        score = instance["score"]

        # Get color based on label or use a default if not found
        formatted_label = label.title()
        color = color_map.get(formatted_label, (0, 255, 0))  # Default to green (BGR)

        # Draw box
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        cv2.rectangle(img_np, pt1, pt2, color, BOX_THICKNESS)

        # --- Label Placement Logic ---
        text = f"{label}: {score:.2f}"
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        
        # Determine the starting coordinates for the background rectangle
        x_start = pt1[0]
        y_start = pt1[1]

        # Calculate the top-left (pt_bg1) and bottom-right (pt_bg2) corners of the background
        pt_bg1 = (x_start, y_start) 
        pt_bg2 = (x_start + w + 2 * TEXT_PADDING, y_start + h + baseline + 2 * TEXT_PADDING)

        # Draw background rectangle for text.
        cv2.rectangle(
            img_np, 
            pt_bg1, 
            pt_bg2, 
            color, 
            -1 # Filled rectangle
        )
        
        # Determine the text position
        text_x = x_start + TEXT_PADDING
        text_y = y_start + h + TEXT_PADDING # y_start (top line) + h (text height) + padding

        # Draw text (Black text)
        cv2.putText(
            img_np, 
            text, 
            (text_x, text_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            FONT_SCALE, 
            (0, 0, 0), # Black text color (BGR)
            FONT_THICKNESS
        )
        
        # Aggregate detection data for JSON
        if label not in json_data:
             json_data[label] = []
        json_data[label].append({
            'box_xyxy': box, 
            'confidence': score
        })

    # Save annotated image
    cv2.imwrite(save_path_img, img_np)
    
    # Save detection data to a dedicated JSON file for this image
    with open(save_path_json, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    return save_path_img, save_path_json

# --- Intermediate Worker Function (Handles opening the file for the executor) ---
def process_and_save_single_image(
    result: Results,
    output_dir: Path,
    json_output_dir: Path,
    class_names: Dict[int, str],
    color_map: Dict[str, Tuple[int, int, int]],
    conf_thresh: float,
    executor: ProcessPoolExecutor,
    white_background: bool = False,
    excluded_classes_set: Set[str] = None,
    included_classes_set: Set[str] = None
) -> Tuple[Future, int] or None:
    """
    Prepares data from a single YOLO result and submits the visualization/save task 
    to the ProcessPoolExecutor. Returns (Future, detection_count) or None on failure.
    """
    image_path = Path(result.path)
    base_name = image_path.stem
    
    # Load the original image file for visualization
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path} for visualization: {e}")
        return None 
    
    # Handle defaults
    if excluded_classes_set is None:
        excluded_classes_set = set()
    if included_classes_set is None:
        included_classes_set = set()
        
    instances_info: List[Dict[str, Any]] = []
    total_detections = 0
    
    if result.boxes is not None and result.boxes.xyxy.shape[0] > 0:
        # Get results on CPU as numpy for easy iteration
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Extract, filter, and structure the data
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf >= conf_thresh: # Apply confidence threshold
                
                label = class_names[class_id]
                # Normalize the label (Title Case) for robust comparison
                formatted_label = label.replace('_', ' ').title() 

                # Class Inclusion Check (Highest Precedence)
                if included_classes_set and formatted_label not in included_classes_set:
                    continue # Skip if this label is not in the required list

                # Class Exclusion Check (Fallback)
                if formatted_label in excluded_classes_set:
                    continue # Skip this instance if it is explicitly excluded

                total_detections += 1
                instances_info.append({
                    'box': box.tolist(),
                    'confidence': float(conf),
                    'score': float(conf),
                    'label': label,
                })
    
    save_path_img = output_dir / f"{base_name}.png"
    save_path_json = json_output_dir / f"{base_name}.json"

    # Submit save task to the process pool
    future = executor.submit(
        draw_and_save_image, 
        image_pil, 
        instances_info, 
        str(save_path_img),
        str(save_path_json),
        color_map,
        white_background
    )
    return future, total_detections


# ---------------- Main Object Detection Function ----------------
def detect_objects(
    image_dir: str,
    output_dir: str,
    model_path: str,
    color_json: str,
    batch_size: int = 4,
    num_workers: int = 4,
    conf_thresh: float = 0.5,
    imgsz: int = 640,
    resume: bool = True,
    white_background: bool = False,
    exclude_classes: str = "",
    include_classes: str = ""
):
    # Setup device and directories
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU. This will be very slow.")

    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    json_output_dir = output_dir / "json_detections"
    os.makedirs(json_output_dir, exist_ok=True)
    
    # --- 1. Load Model and Colors ---
    print(f"Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    model.set_classes(OPEN_IMAGES_CLASSES)
    model.to(device)
    model.eval()
    class_names = model.names
    
    with open(color_json, "r") as f:
        color_map = {k.title(): tuple(v) for k, v in json.load(f).items()} 

    # Process included classes
    included_classes_set = set()
    if include_classes:
        # Split by comma, strip spaces, and normalize to Title Case set for fast lookup
        included_classes_set = {c.strip().title() for c in include_classes.split(',')}
        print(f"Only including the following normalized classes: {included_classes_set}")

    # Process excluded classes (only if include is NOT used)
    excluded_classes_set = set()
    if not included_classes_set and exclude_classes:
        # Split by comma, strip spaces, and normalize to Title Case set for fast lookup
        excluded_classes_set = {c.strip().title() for c in exclude_classes.split(',')}
        print(f"Excluding the following normalized classes: {excluded_classes_set}")
    
    # --- 2. Filter Image Paths (Resume Logic) ---
    image_dir_path = Path(image_dir)
    all_image_paths = sorted(
        [str(p) for ext in (".jpg", ".jpeg", ".png") for p in image_dir_path.glob(f"*{ext}")]
    )
    
    # Identify processed images by checking for their JSON files
    if resume:
        processed_json_stems = {p.stem for p in json_output_dir.glob("*.json")}
        
        # Filter the list to only include paths that haven't been processed
        paths_to_process = [
            path for path in all_image_paths if Path(path).stem not in processed_json_stems
        ]
        
        processed_count = len(all_image_paths) - len(paths_to_process)
        print(f"Resuming: Found {processed_count} processed files. {len(paths_to_process)} images left to process.")
    else:
        paths_to_process = all_image_paths
    
    if not paths_to_process:
        print("All images already processed. Exiting.")
        return
    
    # --- 3. Chunk the Paths to Limit File Handles and GPU Memory ---
    CHUNK_SIZE = 32
    total_images_to_process = len(paths_to_process)
    
    # Split the list into chunks of CHUNK_SIZE
    chunks = [paths_to_process[i:i + CHUNK_SIZE] 
              for i in range(0, total_images_to_process, CHUNK_SIZE)]
    
    print(f"Starting batched YOLO inference on {total_images_to_process} images in {len(chunks)} chunks (Chunk Size: {CHUNK_SIZE}).")

    all_futures: List[Future] = []
    total_detections = 0
    
    # --- 4. Loop Through Chunks, Run YOLO, and Offload Saving ---
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor, \
          tqdm(total=total_images_to_process, desc="YOLOv8 Inference (GPU) & Submission (CPU)") as pbar:

        for chunk_idx, chunk in enumerate(chunks):
            with torch.no_grad():
                results_generator = model.predict(
                    source=chunk,
                    conf=conf_thresh,
                    imgsz=imgsz,
                    device=device,
                    batch=batch_size,
                    stream=True,
                    save=False, 
                    verbose=False
                )

                for r in results_generator:
                    
                    # Offload visualization and saving to the CPU process pool
                    future_result = process_and_save_single_image(
                        r, output_dir, json_output_dir, class_names, color_map, conf_thresh, executor, white_background,
                        excluded_classes_set,
                        included_classes_set
                    )
                    
                    if future_result:
                        future, det_count = future_result
                        all_futures.append(future)
                        total_detections += det_count
                    
                    pbar.update(1) # Update the progress bar for each processed image

            del results_generator
            torch.cuda.empty_cache()
            gc.collect()

        # Wait for all visualization/save tasks to complete
        if all_futures:
            for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Saving final results (CPU)"):
                future.result()
                
    torch.cuda.empty_cache()
    print(f"\nYOLOv8-World detection completed successfully!")
    print(f"Total detections found (above {conf_thresh}): {total_detections}")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Object Detection using YOLOv8-World Model")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save detected images and JSON.")
    parser.add_argument("--model_path", type=str, default="yolov8x-worldv2.pt", help="Path/ID to the YOLOv8-World model weights.")
    parser.add_argument("--color_json", type=str, required=True, help="Path to the LVIS/Open Images color map JSON.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images per batch for GPU inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading (now unused).")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="Bounding Box score threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference (YOLOv8-World default is 640).")
    parser.add_argument("--white_background", action="store_true", help="If set, bounding boxes will be drawn on a pure white background instead of the original image.")
    parser.add_argument("--exclude_classes", type=str, default="", help="Comma-separated list of class names (e.g., 'Human face,Dog') to exclude from visualization and JSON output. Ignored if --include_classes is used.")
    parser.add_argument("--include_classes", type=str, default="", help="Comma-separated list of class names (e.g., 'Man,Woman') to ONLY include in visualization and JSON output. Takes precedence over --exclude_classes.")
    args = parser.parse_args()
    
    detect_objects(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        color_json=args.color_json,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_thresh=args.conf_thresh if args.conf_thresh is not None else 0.25,
        imgsz=args.imgsz,
        white_background=args.white_background,
        exclude_classes=args.exclude_classes,
        include_classes=args.include_classes
    )

# ---------------- Example Usage ----------------
# Yolov8x-World model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolo_world_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-worldv2-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-worldv2.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640

# Yolov8x-World White Background model command:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolo_world_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-worldv2-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-worldv2.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640 --white_background

# Example command to exclude specific classes:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolo_world_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-worldv2-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-worldv2.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640 --exclude_classes "Man, Woman, Human face"

# Example command to include specific classes only:
# "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\.venv-Copy-Copy\Scripts\python.exe" "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolo_world_object_detection.py" --image_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ImageCaptioningEvaluationDatasets\LAION-5B-10k\LAION-5B-10k-images" --output_dir "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\LAION-5B-10k-yolov8x-worldv2-detected" --model_path "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\yolov8x-worldv2.pt" --color_json "C:\MastersRepos\ARI5902-Research-Topics-in-AI\LAION-5B Testing\Spurious_Feature\ObjectDetection\openImagesv7_color_map.json" --batch_size 16 --num_workers 8 --imgsz 640 --include_classes "Man,Woman"