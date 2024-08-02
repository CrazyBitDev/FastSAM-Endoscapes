# FastSAM with Endoscapes2023
## Matteo Ingusci - VR506254

- `download.ipynb`: Jupyter notebook to download the dataset, weights and dependecies.
- `endoscapes_dataset.ipynb`: Jupyter notebook to load and visualize the dataset.
- `endoscapes_dataset_converter.ipynb`: Jupyter notebook to convert the dataset to a format compatible with FastSAM.
- `yolo_train.py`: Python script to fine-tune the YOLOv8 model on the Endoscapes dataset and evaluate it.
- `clip_train.py`: Python script to fine-tune the CLIP model on the Endoscapes dataset and evaluate it.
- `FastSAM_test.py`: Python script to test the FastSAM model on the Endoscapes dataset with the YOLOv8 and CLIP models.

# Edits

- FastSAM repository:
    - `fastsam/prompt.py` now allows to set the model and preprocess function for the CLIP model. It also allows to provide a list of prompts to use and it provides the matrix of similarities between the images and the prompts.

- Ultralytics:
    - Fix the evaluation of the YOLO model, the criterion was not correctly calculated.