import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time 
from enum import Enum
import copy
from random import shuffle 
from typing import Union
import traceback


class Sliced_BBox_Operation(Enum): 
    THROW = 1 # throw away the bbox
    DELETE = 2 # delete the slice


class Point: 
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"


class RectangleRegion: 
    def __init__(self, top_left: Point, bottom_right: Point) -> None:
        self.top_left = top_left
        self.bottom_right = bottom_right

        assert self.top_left.x < self.bottom_right.x, "Top left x must be less than bottom right x"
        assert self.top_left.y < self.bottom_right.y, "Top left y must be less than bottom right y"

    def __contains__(self, point: Point) -> bool:
        return self.top_left.x <= point.x <= self.bottom_right.x and self.top_left.y <= point.y <= self.bottom_right.y
    
    def __repr__(self) -> str:
        return f"RectangleRegion({self.top_left}, {self.bottom_right})"
    
    def __truediv__(self, other: 'RectangleRegion') -> 'RectangleRegion':
        self.top_left.x /= other.width()
        self.top_left.y /= other.height()
        self.bottom_right.x /= other.width()
        self.bottom_right.y /= other.height()
        return self

    def width(self) -> float:
        return self.bottom_right.x - self.top_left.x
    
    def height(self) -> float:
        return self.bottom_right.y - self.top_left.y
    
    def x(self) -> float:
        return self.top_left.x
    
    def y(self) -> float:
        return self.top_left.y
    
    def x_center(self) -> float:
        return self.top_left.x + self.width() / 2
    
    def y_center(self) -> float:
        return self.top_left.y + self.height() / 2

    def area(self) -> float:
        return self.width() * self.height()
    
    def transform_to_local(self, other: 'RectangleRegion') -> None: # transform self to local coordinates of other
        self.top_left -= other.top_left
        self.bottom_right -= other.top_left
    
    def intersection(self, other: 'RectangleRegion') -> Union['RectangleRegion',None]: # returns the intersection of two rectangles
        top_left = Point(max(self.top_left.x, other.top_left.x), max(self.top_left.y, other.top_left.y))
        bottom_right = Point(min(self.bottom_right.x, other.bottom_right.x), min(self.bottom_right.y, other.bottom_right.y))
        # check if intersection is valid
        if top_left.x >= bottom_right.x or top_left.y >= bottom_right.y:
            return None
        return RectangleRegion(top_left, bottom_right)


class Slicer:
    def __init__(self, imgs_path: str, labels_path: str, output_path: str, nn_input_size: int, empty_img_ratio: Union[None,float] = None, bbox_size_threshold: Union[None,float] = None, sliced_bbox_operation: Sliced_BBox_Operation = Sliced_BBox_Operation.DELETE, shuffle_empty: bool = False) -> None: 
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
        self.shuffle_empty = shuffle_empty

        assert nn_input_size > 0, "nn_input_size must be greater than 0"
        self.nn_input_size = nn_input_size

        self._empty_img_count = 0
        self._total_img_count = 0

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
            raise ValueError("Target height and width must be greater or equal to nn_input_size")
            
        img: np.array = cv2.imread(os.path.join(self.imgs_path, img_basename))

        labels = self.load_yolo_labels(os.path.join(self.labels_path, img_basename.replace(img_basename.split('.')[-1], "txt")))
        image_height, image_width, _ = img.shape

        if image_height < target_height or image_width < target_width:
            raise ValueError(f"Image dimensions are smaller than target dimensions for {img_basename}")
        
        regions: dict[int, tuple[int, int, int, int]] = self.calculate_slice_regions(image_height, image_width, target_height, target_width, overlap_height_ratio, overlap_width_ratio)
        sliced_labels: dict[int, list[tuple[int, float, float, float, float]]]

        regions, sliced_labels = self._calc_slice_labels(image_height, image_width, regions, labels) 
        del labels

        # filter labels for bbox size
        if self.empty_img_ratio is not None:
            regions, sliced_labels = self._filter_empty_labels(regions, sliced_labels)
        
        # write labels to disk
        self._save_labels(sliced_labels, img_basename.removesuffix(f".{img_basename.split('.')[-1]}"))

        # slice the actual image and save to disk
        self._slice_and_save_imgs(regions, img, img_basename.removesuffix(f".{img_basename.split('.')[-1]}"), img_basename.split('.')[-1])


    def slice(self, target_height: int, target_width: int, overlap_height_ratio: float, overlap_width_ratio: float, workers: int = os.cpu_count()):
        start = time.time()
        img_basenames = os.listdir(self.imgs_path)
        args = [(img_basename, target_height, target_width, overlap_height_ratio, overlap_width_ratio) for img_basename in img_basenames]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._process, *arg) for arg in args]
            
            for future in tqdm(as_completed(futures), total=len(args), desc="Processing Images"):
                try: 
                    future.result()  
        
                except Exception as e:
                    print(f"An error occurred: {e}")
                    traceback.print_exc()
                    for f in futures:
                        f.cancel()
                    exit(1)

        print(f"Time taken: {time.time() - start:.2f} seconds")


    def single_slice(self, img_basename: str,  target_height: int, target_width: int, overlap_height_ratio: float, overlap_width_ratio: float) -> None:
        self._process(img_basename, target_height, target_width, overlap_height_ratio, overlap_width_ratio)

    
    def _calc_slice_labels(self, image_height: int, image_width: int, regions: dict[int, tuple[int, int, int, int]], labels: list[tuple[int, float, float, float, float]]) -> dict[int, list[tuple[int, float, float, float, float]]]: 
        new_regions = copy.copy(regions) # doesnt need to use deepcopy since del only deletes references
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

                bbox_topl = Point(bbox_x, bbox_y)
                bbox_botr = Point(bbox_x + bbox_w, bbox_y + bbox_h)

                intersection = rectangleRegion.intersection(RectangleRegion(bbox_topl, bbox_botr))
                if intersection is None:
                    continue
                
                intersection.transform_to_local(rectangleRegion)
                intersection /= rectangleRegion


                # filter bbox if it is too small
                if self.bbox_size_threshold is not None and intersection.area() < self.bbox_size_threshold:
                    if self.sliced_bbox_operation == Sliced_BBox_Operation.THROW: 
                        continue
                    
                    elif self.sliced_bbox_operation == Sliced_BBox_Operation.DELETE: # delete label and corresponding region 
                        del new_labels[region_index]
                        del new_regions[region_index] 
                        break
                        
                    else: 
                        raise ValueError("Invalid Sliced_BBox_Operation") 
                         

                new_labels[region_index].append((class_id, intersection.x_center(), intersection.y_center(), intersection.width(), intersection.height()))
        
        return new_regions, new_labels
    

    def _filter_empty_labels(self, regions: dict[int, tuple[int, int, int, int]], labels: dict[int, list[tuple[int, float, float, float, float]]]) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, list[tuple[int, float, float, float, float]]]]: # takes in pointers and deletes items from the list/dict
        bbox_labels = {region_index: label for region_index, label in labels.items() if len(label) > 0}
        empty_labels = {region_index: label for region_index, label in labels.items() if len(label) == 0}

        new_regions = {}
        new_labels = copy.copy(bbox_labels) 

        for index in bbox_labels.keys(): # loop through bboxes
            self._total_img_count += 1

            # add region to new region for this index
            new_regions[index] = regions[index]


        l = list(empty_labels.items())
        if self.shuffle_empty:
            shuffle(l)
            
        for index, empty_label in l: # loop tough empty images
            if self._empty_img_count / max(self._total_img_count,1) < self.empty_img_ratio: # check if ratio is less than threshold
                self._total_img_count += 1
                self._empty_img_count += 1
                # add both labels and empty image 

                new_regions[index] = regions[index]
                new_labels[index] = empty_label # add empty label to

     
        return new_regions, new_labels