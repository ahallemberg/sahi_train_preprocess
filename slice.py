import logging
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

## only accesp bboxes with atleast a threshold, also make sure not too many images are images of just backgound.# resize to nn input size

class Slicer:
    def __init__(self, imgs_path: str, labels_path: str, output_path: str, nn_input_size: int) -> None: 
        assert os.path.exists(imgs_path), "Image path does not exist"
        assert os.path.exists(labels_path), "Label path does not exist"
        
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # also check that output_path/images and output_path/labels exists
        if not os.path.exists(os.path.join(output_path, "images")):
            os.mkdir(os.path.join(output_path, "images"))
        
        if not os.path.exists(os.path.join(output_path, "labels")):
            os.mkdir(os.path.join(output_path, "labels"))

        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.output_path = output_path

        assert nn_input_size > 0, "nn_input_size must be greater than 0"
        self.nn_input_size = nn_input_size

        self._init_logger()

    def _init_logger(self) -> None:
        logging.basicConfig(filename=f"{self.output_path}/log.txt", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def calculate_slice_regions(self, image_height: int, image_width: int, target_height: int, target_width: int, overlap_height_ratio: int, overlap_width_ratio: int) -> list[tuple[tuple[int, int, int, int], int]]:
        # Calculations for moving step in both dimensions
        step_height = target_height - int(target_height * overlap_height_ratio)
        step_width = target_width - int(target_width * overlap_width_ratio)

        # Pre-calculate last slice adjustments
        last_x = image_width - target_width
        last_y = image_height - target_height

        regions: list[tuple[tuple[int, int, int, int], int]] = []
        index = 0

        for y in range(0, image_height, step_height):
            adjusted_y = y if y + target_height <= image_height else last_y
            for x in range(0, image_width, step_width):
                adjusted_x = x if x + target_width <= image_width else last_x
                regions.append((index, (adjusted_x, adjusted_y, target_width, target_height)))
                index += 1

        return regions
    

    def load_yolo_labels(self, filename: str) -> list[tuple[int, float, float, float, float]]:
        with open(filename, 'r') as file:
            # Each line is split into components (class_id, x_center, y_center, width, height),
            # and converted to the appropriate types (int for class_id, float for the rest)
            labels = [tuple(map(float, line.split())) for line in file]
            # If class_id needs to be an int, you can adjust the above line to handle this conversion
            labels = [(int(class_id), *bbox) for class_id, *bbox in labels]
        return labels
    
    def _save_labels(self, sliced_labels: dict[int, list[tuple[int, float, float, float, float]]], name: str) -> None:
        for region_index, label_group in sliced_labels.items(): 
            with open(os.path.join(self.output_path, "labels", f"{name}_{region_index}.txt"), 'w') as file:
                for label in label_group:
                    file.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

        
    def resize_slice(self, img_slice: np.array, target_long_side: int) -> np.array:
        # Get current dimensions
        h, w = img_slice.shape[:2]
        # Determine scaling factor
        if h > w:  # If height is the longer side
            scale = target_long_side / float(h)
            new_h, new_w = target_long_side, int(w * scale)
        else:  # Width is the longer side or equal
            scale = target_long_side / float(w)
            new_h, new_w = int(h * scale), target_long_side
        # Resize the slice
        resized_slice = cv2.resize(img_slice, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_slice

    
    def _slice_and_save_imgs(self, regions: list[tuple[int, tuple[int, int, int, int]]], img: np.ndarray, name: str, ext: str) -> None:
        for region_index, region in regions:
            x, y, w, h = region
            sliced_img = img[y:y+h, x:x+w]
            if sliced_img.shape[0] > self.nn_input_size or sliced_img.shape[1] > self.nn_input_size:
                sliced_img = self.resize_slice(sliced_img, self.nn_input_size)
            cv2.imwrite(os.path.join(self.output_path, "images", f"{name}_{region_index}.{ext}"), sliced_img)

    
    def _process(self, img_basename: str, target_height: int, target_width: int, overlap_height_ratio: int, overlap_width_ratio: int) -> None:
        if target_height < self.nn_input_size or target_width < self.nn_input_size:
            logging.error("Target height and width must be greater or equal to nn_input_size")
            return
    
        img: np.array = cv2.imread(os.path.join(self.imgs_path, img_basename))

        labels = self.load_yolo_labels(os.path.join(self.labels_path, img_basename.replace(img_basename.split('.')[-1], "txt")))
        image_height, image_width, _ = img.shape

        if image_height < target_height or image_width < target_width:
            logging.warning(f"Image dimensions are smaller than target dimensions for {img_basename}")
            return
        
        regions = self.calculate_slice_regions(image_height, image_width, target_height, target_width, overlap_height_ratio, overlap_width_ratio)
        sliced_labels = self._calc_slice_labels(image_height, image_width, regions, labels)

        ## determin which slices to use
        # update regions var for that, and sliced_labels

        # write label to disk
        self._save_labels(sliced_labels, img_basename.removesuffix(f".{img_basename.split('.')[-1]}"))
        # slice image
        self._slice_and_save_imgs(regions, img, img_basename.removesuffix(f".{img_basename.split('.')[-1]}"), img_basename.split('.')[-1])

        #self._slice_img_and_label(regions, img, label)
        

    def slice(self, target_height: int, target_width: int, overlap_height_ratio: float, overlap_width_ratio: float, workers: int = os.cpu_count()):
        img_basenames = os.listdir(self.imgs_path)
        args = [(img_basename, target_height, target_width, overlap_height_ratio, overlap_width_ratio) for img_basename in img_basenames]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._process, *arg) for arg in args]
            
            for _ in tqdm(as_completed(futures), total=len(args), desc="Processing Images"):
                pass

        self.clean
    def single_slice(self, img_basename: str,  target_height: int, target_width: int, overlap_height_ratio: float, overlap_width_ratio: float):
        self._process(img_basename, target_height, target_width, overlap_height_ratio, overlap_width_ratio)
    
    def _calc_slice_labels(self, image_height: int, image_width: int, regions: list[tuple[int, tuple[int, int, int, int]]], labels: list[tuple[int, float, float, float, float]]) -> dict[int, list[tuple[int, float, float, float, float]]]: 
        new_labels: dict[int, list[tuple[int, float, float, float, float]]] = {region_index: [] for region_index, _ in regions}
        for region_index, (region_x, region_y, region_w, region_h) in regions:
      
            for class_id, x_center_rel, y_center_rel, width_rel, height_rel in labels: # relative to image
                bbox_x = (x_center_rel - width_rel / 2) * image_width
                bbox_y = (y_center_rel - height_rel / 2) * image_height
                bbox_w = width_rel * image_width
                bbox_h = height_rel * image_height

                delta_x = bbox_x - region_x
                delta_y = bbox_y - region_y

                if delta_x >= 0 and delta_x <= region_w and delta_y >= 0 and delta_y <= region_h:
                    bbox_region_x_rel = delta_x / region_w
                    bbox_region_y_rel = delta_y / region_h 

                    bbox_region_width_rel = min(bbox_w / region_w, 1-bbox_region_x_rel)
                    bbox_region_height_rel = min(bbox_h / region_h, 1-bbox_region_y_rel)

                    bbox_region_x_center_rel = bbox_region_x_rel + bbox_region_width_rel / 2
                    bbox_region_y_center_rel = bbox_region_y_rel + bbox_region_height_rel / 2
                    

                    new_labels[region_index].append((class_id, bbox_region_x_center_rel, bbox_region_y_center_rel, bbox_region_width_rel, bbox_region_height_rel))

        return new_labels


slicer = Slicer("/home/askhb/ascend/suas2023_detection_dataset/test/resized/images", "/home/askhb/ascend/suas2023_detection_dataset/test/resized/labels", "/home/askhb/ascend/suas2023_detection_dataset/test/slicedandresized", 640)
slicer.slice(640, 640, 0.5, 0.5)
