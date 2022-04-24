# System for action recognition

The system is based on the integration of information from two separate modules. The first module (PAN_for_YOLACT) is responsible for motion detection and categorisation. The second module (yolact) is an instance segmentation module that recognises objects and their position on the
scene. The information from both modules is passed to a classifier that makes the final prediction.

### To train PAN:
0) Activate pan env with conda
1) Prepare a dataset as described in [readme](https://gitlab.ciirc.cvut.cz/ostapana/bachelor_thesis_code/-/blob/main/PAN_for_YOLACT/README.md)
2) Configure PAN_for_YOLACT/ops/dataset_config.py (return_somethingv2 and ROOT_DATASET)
3) Run ```scripts/train/sthv2/Lite.sh```

### To test PAN:
1) Configure PAN_for_YOLACT/test_models.py (Choose correct weights, categories, etc.)
2) Run ```scripts/test/sthv2/Lite.sh```

### To prepare YOLACT module:
0) Activate yolact-env with conda
1) Run ```python.py --annotate --path path_to_videos --move```

### To train the final classifier (on 8 classes)
1) Run ```yolact-camera/scripts/train/train_8.sh ```

### To test the final classifier (on 8 classes)
1) Run ```yolact-camera/scripts/test/test_8.sh ```

Configure scripts if needed
