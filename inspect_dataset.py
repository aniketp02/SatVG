import pickle
import numpy as np

# Load the train data
with open('/home/pokle/Trans-VG/visual_grounding/dior-rsvg/train-data.pth', 'rb') as f:
    train_data = pickle.load(f)

# Extract all bounding boxes
all_boxes = [item[1] for item in train_data]

# Check for uniqueness
unique_boxes = np.unique(all_boxes, axis=0)
print(f"Total boxes: {len(all_boxes)}, Unique boxes: {len(unique_boxes)}")

# Print distribution of boxes
if len(unique_boxes) < 10:
    print("Unique boxes:", unique_boxes)
else:
    print("First 5 boxes:", all_boxes[:5])
    # Check if all boxes are identical
    all_identical = all(np.array_equal(box, all_boxes[0]) for box in all_boxes)
    print(f"All boxes identical: {all_identical}")
