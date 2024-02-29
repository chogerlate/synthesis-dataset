import os 
from pathlib import Path
import cv2 
import numpy as np 
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import albumentations as A


class cfg:
    # original dataset variables
    data = Path('./original_dataset')
    bg = data / 'bg'
    obj = data / 'obj'

    # generated dataaset variables
    dataset =  data/ 'dataset'
    num_images = 20
    

def get_img_and_mask(img_path, mask_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) 
    mask = mask.astype(np.uint8) # This is binary mask
    
    return img, mask

def resize_img(img, desired_max, desired_min=None):
   
    h, w = img.shape[0], img.shape[1]
    
    longest, shortest = max(h, w), min(h, w)
    longest_new = desired_max
    if desired_min:
        shortest_new = desired_min
    else:
        shortest_new = int(shortest * (longest_new / longest))
    
    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
        
    transform_resize = A.Compose([
        A.Sequential([
        A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)
        ], p=1)
    ])

    transformed = transform_resize(image=img)
    img_r = transformed["image"]
        
    return img_r


def resize_transform_obj(img, mask, h_new, w_new, perspective=False):
    """resize and trasnform object image and mask for object image

    Args:
        img (numpy array): object image
        mask (numpy array): mask of object image
        h_new (int): _description_
        w_new (int): _description_
        perspective (bool, optional): additional image transform such perspective. Defaults to False.

    Returns:
        tuple(numpy array, numpy array): transformed object image and mask
    """        
    transform_resize = A.Resize(h_new, w_new, interpolation=1, always_apply=False, p=1)

    transformed_resized = transform_resize(image=img, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]

    transforms = A.Compose([
    A.Perspective(p=1,fit_output=True),
    ])
    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]
        
    return img_t, mask_t

def add_obj(img_comp, mask_comp, img, mask, x, y, idx):
    '''
    img_comp - composition of objects
    mask_comp - composition of objects` masks
    img - image of object
    mask - binary mask of object
    x, y - coordinates where center of img is placed
    Function returns img_comp in CV2 RGB format + mask_comp
    '''
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]
    
    h, w = img.shape[0], img.shape[1]
    
    x = x - int(w/2)
    y = y - int(h/2)
    
    mask_b = mask == 255
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)
    
    if x >= 0 and y >= 0:
    
        h_part = h - max(0, y+h-h_comp) # h_part - part of the image which gets into the frame of img_comp along y-axis
        w_part = w - max(0, x+w-w_comp) # w_part - part of the image which gets into the frame of img_comp along x-axis

        img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * ~mask_rgb_b[0:h_part, 0:w_part, :] + (img * mask_rgb_b)[0:h_part, 0:w_part, :]
        mask_comp[y:y+h_part, x:x+w_part] = mask_comp[y:y+h_part, x:x+w_part] * ~mask_b[0:h_part, 0:w_part] + (idx * mask_b)[0:h_part, 0:w_part]
        mask_added = mask[0:h_part, 0:w_part]
        
    elif x < 0 and y < 0:
        
        h_part = h + y
        w_part = w + x
        
        img_comp[0:0+h_part, 0:0+w_part, :] = img_comp[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_b[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_b)[h-h_part:h, w-w_part:w, :]
        mask_comp[0:0+h_part, 0:0+w_part] = mask_comp[0:0+h_part, 0:0+w_part] * ~mask_b[h-h_part:h, w-w_part:w] + (idx * mask_b)[h-h_part:h, w-w_part:w]
        mask_added = mask[h-h_part:h, w-w_part:w]
        
    elif x < 0 and y >= 0:
        
        h_part = h - max(0, y+h-h_comp)
        w_part = w + x
        
        img_comp[y:y+h_part, 0:0+w_part, :] = img_comp[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_b[0:h_part, w-w_part:w, :] + (img * mask_rgb_b)[0:h_part, w-w_part:w, :]
        mask_comp[y:y+h_part, 0:0+w_part] = mask_comp[y:y+h_part, 0:0+w_part] * ~mask_b[0:h_part, w-w_part:w] + (idx * mask_b)[0:h_part, w-w_part:w]
        mask_added = mask[0:h_part, w-w_part:w]
        
    elif x >= 0 and y < 0:
        
        h_part = h + y
        w_part = w - max(0, x+w-w_comp)
        
        img_comp[0:0+h_part, x:x+w_part, :] = img_comp[0:0+h_part, x:x+w_part, :] * ~mask_rgb_b[h-h_part:h, 0:w_part, :] + (img * mask_rgb_b)[h-h_part:h, 0:w_part, :]
        mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * ~mask_b[h-h_part:h, 0:w_part] + (idx * mask_b)[h-h_part:h, 0:w_part]
        mask_added = mask[h-h_part:h, 0:w_part]
    
    return img_comp, mask_comp, mask_added

def generated_dataset(object_path:str, 
                      bg_paths:str,
                      save_dir:str,
                      split_ratios=(0.8, 0.1, 0.1),
                      num_images=10,
                      object_range=(1,5)):
    """generate dataset from object and background images

    Args:
        object_path (str): object image path
        bg_paths (str): background image path
        save_dir (str): save directory
        split_ratios (tuple, optional): ratios for train, valid and test. Defaults to (0.8, 0.1, 0.1).
        num_images (int, optional): number of images we want to generate. Defaults to 10.
        object_range (tuple, optional): range of number of objects in image . Defaults to (1,5).
    """
    
    bg_paths = sorted(list(Path(bg_paths).glob('*.jpg')))
    obj_paths = sorted(list(Path(object_path).glob('*.jpg')))
    obj_mask_paths = sorted(list(Path(object_path).glob('*_mask.png')))
    obj_dict = {
        "object_path": obj_paths,
        "object_mask_path": obj_mask_paths
    }
    id2object = {i: str(os.path.basename(obj))[:-4] for i, obj in enumerate(obj_paths)}
    print(id2object)
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Split data into train, valid, and test sets
    train_ratio, valid_ratio, test_ratio = split_ratios
    num_train_images = int(num_images * train_ratio)
    num_valid_images = int(num_images * valid_ratio)
    num_test_images = num_images - num_train_images - num_valid_images

    # Create separate directories for train, valid, and test sets
    train_dir = os.path.join(save_dir, "train")
    valid_dir = os.path.join(save_dir, "valid")
    test_dir = os.path.join(save_dir, "test")

    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

    # Generate the dataset
    for i in range(num_images):
        # Randomly select a background image
        bg_path = np.random.choice(bg_paths)
        bg = cv2.imread(str(bg_path))
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg = resize_img(bg, desired_max=1920, desired_min=1920)
        mask_comp = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)

        round_range = np.random.randint(object_range[0], object_range[1]+1)
        classes = []
        for round in range(round_range):

            # Randomly select an object image and its mask
            obj_idx = np.random.choice(len(obj_dict["object_path"]))
            classes.append(obj_idx) # add the class to the list
            obj_path = obj_dict["object_path"][obj_idx]
            mask_path = obj_dict["object_mask_path"][obj_idx]
            obj, mask = get_img_and_mask(str(obj_path), str(mask_path))

            # Resize and transform the object image and its mask
            # Generate random height and width within a range
            h_new = np.random.randint(300, 800)
            w_new = np.random.randint(300, 800)
            obj, mask = resize_transform_obj(obj, mask, h_new=h_new, w_new=w_new, perspective=True)
            
            # Add the object to the background image
            x = np.random.randint(0,bg.shape[1]) 
            y = np.random.randint(0,bg.shape[0])
            img_comp, mask_comp, _ = add_obj(bg, mask_comp, obj, mask, x=x, y=y, idx=round+1)
            
        # Save the composition and its label
        if i < num_train_images:
            save_path = os.path.join(train_dir, "images", f"Image{i}.jpg")
            label_save_path = os.path.join(train_dir, "labels", f"Label{i}.txt")
        elif i < num_train_images + num_valid_images:
            save_path = os.path.join(valid_dir, "images", f"Image{i}.jpg")
            label_save_path = os.path.join(valid_dir, "labels", f"Label{i}.txt")
        else:
            save_path = os.path.join(test_dir, "images", f"Image{i}.jpg")
            label_save_path = os.path.join(test_dir, "labels", f"Label{i}.txt")

        cv2.imwrite(save_path, cv2.cvtColor(img_comp, cv2.COLOR_RGB2BGR))

        # write the label file
        # Get the unique values in the mask (excluding the background)
        unique_values = np.unique(mask_comp)[1:]
        bounding_boxes = []

        # Iterate over the unique values
        for value in unique_values:
            # Create a binary mask for the current object
            object_mask = np.where(mask_comp == value, 1, 0).astype(np.uint8)
            # Convert the mask to grayscale
            # object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
            # Bounding Box
            bbox = [0, 0, 0, 0]
            # Find where the mask is not zero
            rows = np.any(object_mask, axis=1)
            cols = np.any(object_mask, axis=0)

            # Find the minimum and maximum row and column indices
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            bbox = [x_min, x_max, y_min, y_max]
            bounding_boxes.append(bbox)

        print(save_path,bounding_boxes, classes)
        # Write the bounding boxes to the label file
        with open(label_save_path, 'w') as f:
            for i, box in enumerate(bounding_boxes):
                x1, y1 = box[0], box[2]
                x2, y2 = box[1], box[2]
                x3, y3 = box[1], box[3]
                x4, y4 = box[0], box[3]
                f.write(f"{classes[i]} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")


def plot_image_and_boxes(image_path, bounding_box_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load the bounding box data
    bounding_boxes = []
    with open(bounding_box_path, 'r') as f:
        for line in f:
            data = line.strip().split()
            class_id = int(data[0])
            x1, y1, x2, y2, x3, y3, x4, y4 = map(int, data[1:])
            bounding_boxes.append(((x1, y1), (x2, y2), (x3, y3), (x4, y4)))

    # Plot the image and bounding boxes
    fig, ax = plt.subplots()
    ax.imshow(image)

    for bbox in bounding_boxes:
        x = [bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0], bbox[0][0]]
        y = [bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1], bbox[0][1]]
        ax.plot(x, y, color='red')

    plt.show()

if __name__ == "__main__":
    generated_dataset(object_path=cfg.obj, 
                      bg_paths=cfg.bg,
                      save_dir=cfg.dataset, 
                      num_images=cfg.num_images, 
                      object_range=(1,5))


