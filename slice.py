import logging
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time 
import shutil
from enum import Enum
import copy
from random import shuffle 

## only accesp bboxes with atleast a threshold, also make sure not too many images are images of just backgound.# resize to nn input size

class Sliced_BBox_Operation(Enum): 
    THROW = 1 # throw away the bbox
    DELETE = 2 # delete the slice

class Point: 
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)


class RectangleRegion: 
    def __init__(self, top_left: Point, bottom_right: Point) -> None:
        self.top_left = top_left
        self.bottom_right = bottom_right

    def __contains__(self, point: Point) -> bool:
        return self.top_left.x <= point.x <= self.bottom_right.x and self.top_left.y <= point.y <= self.bottom_right.y

# slicer multiple constructors
class Slicer:
    def __init__(self, imgs_path: str, labels_path: str, output_path: str, nn_input_size: int, empty_img_ratio: None|float = None, bbox_size_threshold: None|float = None, sliced_bbox_operation: None|Sliced_BBox_Operation = None) -> None: 
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
        self.empty_img_ratio = empty_img_ratio
        self.bbox_size_threshold = bbox_size_threshold
        self.sliced_bbox_operation = sliced_bbox_operation

        assert nn_input_size > 0, "nn_input_size must be greater than 0"
        self.nn_input_size = nn_input_size

        self._empty_img_count = 0
        self._total_img_count = 0

        self._init_logger()


    def _init_logger(self) -> None:
        logging.basicConfig(filename=f"{self.output_path}/log.txt", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


    def calculate_slice_regions(self, image_height: int, image_width: int, target_height: int, target_width: int, overlap_height_ratio: int, overlap_width_ratio: int) -> dict[int, tuple[int, int, int, int]]:
        # Calculations for moving step in both dimensions
        step_height = target_height - int(target_height * overlap_height_ratio)
        step_width = target_width - int(target_width * overlap_width_ratio)

        # Pre-calculate last slice adjustments
        last_x = image_width - target_width
        last_y = image_height - target_height

        regions: dict[int, tuple[int, int, int, int]] = {}
        index = 0

        for y in range(0, image_height, step_height):
            adjusted_y = y if y + target_height <= image_height else last_y
            for x in range(0, image_width, step_width):
                adjusted_x = x if x + target_width <= image_width else last_x
                regions[index] = (adjusted_x, adjusted_y, target_width, target_height)
                index += 1

        return regions
    

    def load_yolo_labels(self, filename: str) -> list[tuple[int, float, float, float, float]]:
        with open(filename, 'r') as file:
            labels = [tuple(map(float, line.split())) for line in file]
            labels = [(int(class_id), *bbox) for class_id, *bbox in labels]
        return labels
    
    
    def _save_labels(self, sliced_labels: dict[int, list[tuple[int, float, float, float, float]]], name: str) -> None:
        for region_index, label_group in sliced_labels.items(): 
            with open(os.path.join(self.output_path, "labels", f"{name}_{region_index}.txt"), 'w') as file:
                for label in label_group:
                    file.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

        
    def resize_slice(self, img_slice: np.array, target_long_side: int) -> np.array:
        h, w = img_slice.shape[:2]
        if h > w:  
            scale = target_long_side / float(h)
            new_h, new_w = target_long_side, int(w * scale)
        else: 
            scale = target_long_side / float(w)
            new_h, new_w = int(h * scale), target_long_side

        # Resize the slice
        resized_slice = cv2.resize(img_slice, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_slice

    
    def _slice_and_save_imgs(self, regions: dict[int, tuple[int, int, int, int]], img: np.ndarray, name: str, ext: str) -> None:
        for region_index, region in regions.items():
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
        
        regions: dict[int, tuple[int, int, int, int]] = self.calculate_slice_regions(image_height, image_width, target_height, target_width, overlap_height_ratio, overlap_width_ratio)
        sliced_labels: dict[int, list[tuple[int, float, float, float, float]]]

        regions, sliced_labels = self._calc_slice_labels(image_height, image_width, regions, labels) #filter for bbox and empty images
   
   

        # filter labels for bbox size
        if self.empty_img_ratio is not None:
            regions, sliced_labels = self._filter_empty_labels(regions, sliced_labels)
        
        # write label to disk
        self._save_labels(sliced_labels, img_basename.removesuffix(f".{img_basename.split('.')[-1]}"))
        # slice image
        self._slice_and_save_imgs(regions, img, img_basename.removesuffix(f".{img_basename.split('.')[-1]}"), img_basename.split('.')[-1])

    def _compress(self) -> None:
        shutil.make_archive(os.path.basename(self.output_path), 'zip', self.output_path)

    def _post_process(self) -> None:
        # print out log and delete file
        with open(f"{self.output_path}/log.txt", 'r') as file:
            print(file.read())
            os.remove(f"{self.output_path}/log.txt")


    def slice(self, target_height: int, target_width: int, overlap_height_ratio: float, overlap_width_ratio: float, archive: bool = False, workers: int = os.cpu_count()):
        start = time.time()
        img_basenames = os.listdir(self.imgs_path)
        args = [(img_basename, target_height, target_width, overlap_height_ratio, overlap_width_ratio) for img_basename in img_basenames]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._process, *arg) for arg in args]
            
            for _ in tqdm(as_completed(futures), total=len(args), desc="Processing Images"):
                pass
        
        self._post_process()

        if archive: 
            self._compress()

        print(f"Time taken: {time.time() - start:.2f} seconds")

    def single_slice(self, img_basename: str,  target_height: int, target_width: int, overlap_height_ratio: float, overlap_width_ratio: float) -> None:
        self._process(img_basename, target_height, target_width, overlap_height_ratio, overlap_width_ratio)
        self._post_process()

    
    def _calc_slice_labels(self, image_height: int, image_width: int, regions: dict[int, tuple[int, int, int, int]], labels: list[tuple[int, float, float, float, float]]) -> dict[int, list[tuple[int, float, float, float, float]]]: 
        new_regions = copy.deepcopy(regions)
        new_labels: dict[int, list[tuple[int, float, float, float, float]]] = {region_index: [] for region_index in regions}
     
        for region_index, (region_x, region_y, region_w, region_h) in regions.items():
            region_topl = Point(region_x, region_y)
            region_botr = Point(region_x + region_w, region_y + region_h)
            rectangleRegion = RectangleRegion(region_topl, region_botr)
      
            for class_id, x_center_rel, y_center_rel, width_rel, height_rel in labels: # relative to original image
                bbox_x = (x_center_rel - width_rel / 2) * image_width
                bbox_y = (y_center_rel - height_rel / 2) * image_height
                bbox_w = width_rel * image_width
                bbox_h = height_rel * image_height

                bbox_topl = (bbox_x, bbox_y)
                bbox_topr = (bbox_x + bbox_w, bbox_y)
                bbox_botl = (bbox_x, bbox_y + bbox_h)
                bbox_botr = (bbox_x + bbox_w, bbox_y + bbox_h)

                print(f"bbox {bbox_topl} {bbox_topr} {bbox_botl} {bbox_botr}")
                print((region_x, region_y, region_w, region_h))
                
                









            
                #check for absolute delta. Do not skip/
                delta_x = bbox_x - region_x
                delta_y = bbox_y - region_y

                if abs(delta_x) < region_w and abs(delta_y) < region_h: # check if distance is correct, doesnt overshoot 
                    if delta_x < 0: # if delta is negative, remove from width
                        bbox_w = bbox_w + delta_x
                        bbox_region_x_rel = 0
                        bbox_region_width_rel = min(bbox_w / region_w, 1) 
                    else: 
                        bbox_w = ((region_x+region_w) -delta_x) / region_w
                        bbox_region_x_rel = delta_x / region_w
                        bbox_region_width_rel = bbox_w / region_w

                    
                    if delta_y < 0: # if delta is negative, remove from height
                        bbox_h = bbox_h + delta_y
                        bbox_region_y_rel = 0
                        bbox_region_height_rel = min(bbox_h / region_h, 1) # if object is bigger than slice, set to 1
                    else:
                        bbox_h = ((region_y+region_h) -delta_y) / region_h
                        bbox_region_y_rel = delta_y / region_h
                        bbox_region_height_rel = bbox_h / region_h
               

                    if bbox_region_width_rel < 0 or bbox_region_height_rel < 0: # check if bbox is outside of region
                        continue

                    # filter bbox if it is too small
                    if self.bbox_size_threshold is not None and (bbox_region_width_rel*bbox_region_height_rel)/((bbox_w/region_w)*(bbox_h/region_h)) < self.bbox_size_threshold:
                        match self.sliced_bbox_operation: 
                            case Sliced_BBox_Operation.THROW: # remove bbox from label
                                print(f"Throwing away bbox {class_id} in region {region_index}")
                                continue

                            case Sliced_BBox_Operation.DELETE | None: # delete label and corresponding region 
                                print(f"Deleting bbox {class_id} in region {region_index}")
                                del new_labels[region_index]
                                del new_regions[region_index] 
                                break
                          
                            case _: 
                                raise ValueError("Invalid Sliced_BBox_Operation") 

                    bbox_region_x_center_rel = bbox_region_x_rel + (bbox_region_width_rel / 2)
                    bbox_region_y_center_rel = bbox_region_y_rel + (bbox_region_height_rel / 2)
                    

                    new_labels[region_index].append((class_id, bbox_region_x_center_rel, bbox_region_y_center_rel, bbox_region_width_rel, bbox_region_height_rel))
        
        # print all keys in new_regions

        
 
        return new_regions, new_labels
    
    def _filter_empty_labels(self, regions: dict[int, tuple[int, int, int, int]], labels: dict[int, list[tuple[int, float, float, float, float]]]) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, list[tuple[int, float, float, float, float]]]]: # takes in pointers and deletes items from the list/dict
         
        bbox_labels = {region_index: label for region_index, label in labels.items() if len(label) > 0}
    
        empty_labels = {region_index: label for region_index, label in labels.items() if len(label) == 0}


        new_regions = {}
        new_labels = copy.deepcopy(bbox_labels) # need to do deepcopy

        for key, item in bbox_labels.items(): # loop through bboxes
            self._total_img_count += 1

            # add region to new region for this key
            new_regions[key] = regions[key]

            print(key,item)
            if key in [2, 12 ,43]:
                print("bbox loop " + str(key))

        # shuffle empty labels to get random backgrounds 
        l = list(empty_labels.items())
        shuffle(l)
        for index, empty_label in l: # loop tough empty images
            if self._empty_img_count / max(self._total_img_count,1) < self.empty_img_ratio: # check if ratio is less than threshold
                self._total_img_count += 1
                self._empty_img_count += 1
                # add both labels and empty image 

                new_regions[index] = regions[index]
                new_labels[index] = empty_label # add empty label to

                print(index, empty_label)
                if item in [2, 12 ,43]:
                    print("empty loop " + str(index))

        print("LABELS")
        print(new_labels)
        print("REGIONS")
        print(new_regions)
        return new_regions, new_labels

        """
    _save_labels
        print(labels)
        # first add bboxes, then add the empty images. 
        print(regions)

        new_regions = copy.deepcopy(regions)
        new_labels = copy.deepcopy(labels)

        for region_index, label in labels.items(): 
            if len(label) == 0:
                if self._empty_img_count / max(self._total_img_count,1) > self.empty_img_ratio:
                    del new_labels[region_index]
                    del new_regions[region_index]
                else: 
                    self._empty_img_count += 1
                    self._total_img_count += 1
            else:
                self._total_img_count += 1

        return new_regions, new_labels

        """

slicer = Slicer(
    "/home/askhb/ascend/suas2023_detection_dataset/test/resized/images", 
    "/home/askhb/ascend/suas2023_detection_dataset/test/resized/labels", 
    "/home/askhb/ascend/suas2023_detection_dataset/test/custom_sliced_test2", 
    640, 

)

slicer.single_slice("Image7830.png", 640, 640, 0.2, 0.2)