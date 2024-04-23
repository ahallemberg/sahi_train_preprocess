
    def _filter_labels_bbox(self, regions: dict[int, tuple[int, int, int, int]], labels: dict[int, list[tuple[int, float, float, float, float]]]) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, list[tuple[int, float, float, float, float]]]]: # takes in pointers and deletes items from the list/dict
        new_regions = copy.copy(regions)
        new_labels = copy.copy(labels)
      
        _local_empty_img_count = 0

        for region_index, label in labels.items(): 
            print(new_regions)
            print(new_labels)
            print("-"*20)
            print(label)
            input()    
     
            if len(label) == 0: # empty label, determin what to do
                if self._empty_img_count / max(self._total_img_count,1) > self.empty_img_ratio: # too many empty images, delete slice
                    print("deleting img")
                    print(self._total_img_count, self._empty_img_count)
                    del new_labels[region_index]
                    del new_regions[region_index]

                else: # keep slice
                    print("keepign empty slice")
                    self._total_img_count += 1
                    self._empty_img_count += 1

            else: # non empty label, check size of bboxes 
                for bbox in label: 
                    if bbox[3]*bbox[4] < self.bbox_size_threshold: # area size of bbox is smaller than the threshold, determin what to do by the self.sliced_bbox_operation
                        match self.sliced_bbox_operation: 
                            case Sliced_BBox_Operation.THROW: # remove bbox from label
                                new_labels[region_index].remove(bbox)

                                # check if label is empty, if so delete slice

                            case Sliced_BBox_Operation.DELETE: # delete label and corresponding region 
                                del new_labels[region_index]
                                del new_regions[region_index] 
                          
                            case _: 
                                raise ValueError("Invalid Sliced_BBox_Operation") 
                             
                try: 
                    if len(new_labels[region_index]) == 0: # no bboxes, remove label from labels, and region
                        del new_labels[region_index]
                        del new_regions[region_index]

                    else: # non empty
                        self._total_img_count += 1

                except KeyError: 
                    pass 

        # should be updated on the fly 
        self._total_img_count += len(regions)
        self._empty_img_count += _local_empty_img_count

        print(new_labels)
    
        return new_regions, new_labels  
        