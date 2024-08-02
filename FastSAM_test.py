import sys
sys.path.append('./FastSAM')

from fastsam import FastSAM, FastSAMPrompt
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import clip
import cv2
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay

def parseFile(file: str, img_size: tuple) -> list[dict]:
    """
    function to parse the labels file

    Args:
    file: str -> the path to the labels file
    img_size: tuple -> the size of the image

    Returns:
    list[dict] -> a list of dictionaries containing the instances
    """

    # read the labels file lines
    with open(file) as f:
        labels_str = f.readlines()

    instances = []
    # for each line in the labels file
    for i in range(len(labels_str)):
        # split the line by spaces
        label = labels_str[i].split(' ')
        
        # if the image size is provided, convert the segment to the image size
        if img_size:
            segment = np.array([
                np.array(label[1::2], dtype=np.float64) * float(img_size[1]),
                np.array(label[2::2], dtype=np.float64) * float(img_size[0])
            ], dtype=np.int64).T

            # define the mask
            mask = np.zeros((int(img_size[0]), int(img_size[1])), dtype=np.uint8)
            cv2.fillPoly(mask, [segment], 1)
        else:
            # if the image size is not provided, set the segment and mask to None
            segment = None
            mask = None

        # append the instance to the instances list
        instances.append({
            'segment': segment,
            'class': int(label[0]),
            'mask': mask,
            'best_score': 0,
            'best_result_mask': None
        })

    return instances


def combine(instances: list[dict], results: object) -> list[dict]:
    """
    Combine the instances with the results

    Args:
    instances: list[dict] -> a list of dictionaries containing the instances
    results: object -> the results object

    Returns:
    list[dict] -> a list of dictionaries containing the instances with the best result mask
    """

    # get the masks from the results
    result_masks = results[0].masks.data

    # for each instance
    for i in range(len(instances)):
        # get the mask
        mask = np.array(instances[i]['mask'])
        # for each mask in the results
        for j in range(len(result_masks)):
            # get the result mask
            result_mask = result_masks[j].cpu().numpy()
            # compute the intersection and union
            intersection = np.logical_and(mask, result_mask)
            union = np.logical_or(mask, result_mask)
            # compute the Intersection over Union (IoU)
            iou = np.sum(intersection) / np.sum(union)
            # if the IoU is greater than the best score, update the best score and the best result mask
            if iou > instances[i]['best_score']:
                instances[i]['best_score'] = iou
                instances[i]['best_result_mask'] = j

    return instances

# set the variable device to "cuda:0" if a GPU is available, otherwise set it to "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load the CLIP model and the pre-process function
# load the fine-tuned CLIP model weights
clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
checkpoint = torch.load("./weights/CLIP-finetuned.pt")
clip_model.load_state_dict(checkpoint['model_state_dict'])

# load the FastSAM model
fastSAM_model = FastSAM('./weights/YOLO8-finetuned.pt')

# define the class names
CLASS_NAMES = [
    'background',
    'cystic plate',
    'calot triangle',
    'cystic artery',
    'cystic duct',
    'gallbladder',
    'tool'
]

ground_truth_classes = []
predicted_classes = []
probabilities = []
images_test = os.listdir('./dataset/images/test/')
# for each image in the test images
for image in tqdm(images_test):
    # get the results from the FastSAM (YOLO) model
    everything_results = fastSAM_model(f'./dataset/images/test/{image}',
                                       device=device, retina_masks=True,
                                       conf=0.4, iou=0.9,
                                       verbose=False)
    
    # if there are no results, set the shape to None, otherwise set it to the original shape of the results
    if everything_results != None:
        shape = everything_results[0].orig_shape
    else:
        shape = None

    # load and parse the label file
    label_file = image.replace('.jpg', '.txt')
    instances = parseFile(f'./dataset/labels/test/{label_file}', shape)

    # if there are no results, set the ground truth classes to the classes of the instances
    # and the predicted classes and probabilities to 0
    if everything_results == None:
        for instance in instances:
            ground_truth_classes.append(instance['class'])
            predicted_classes.append(0)
            probabilities.append(0.0)
        continue

    # combine the instances with the results
    instances = combine(instances, everything_results)

    # define the FastSAMPrompt object
    prompt_process = FastSAMPrompt(f'./dataset/images/test/{image}',
                                   everything_results, device=device,
                                   clip_model=clip_model,
                                   preprocess=preprocess
                                   )
    
    # get the prompt text and the probabilities
    _, probs = prompt_process.text_prompt(
        text=[f'a photo of a {CLASS_NAMES[class_id]}' for class_id in range(1, len(CLASS_NAMES))]
    )
    # convert the probabilities to softmax, then get the top 3 values and indexes
    probs = probs.softmax(dim=1)
    values, indexes = probs.topk(3)

    # for each instance
    for instance in instances:
        # get the ground truth class
        gt_class = instance['class']
        ground_truth_classes.append(gt_class)
        # get the predicted instance index
        predicted_instance_idx = instance['best_result_mask']
        # if the instance has no predicted instance, set the predicted class and probability to 0
        if predicted_instance_idx == None:
            predicted_classes.append(0)
            probabilities.append(0.0)
            continue
        # get the predicted class and probability
        predicted_class = indexes[predicted_instance_idx][0].item() + 1
        probabilities.append(probs[predicted_instance_idx][gt_class-1].item())
        predicted_classes.append(predicted_class)

# compute and plot the confusion matrix
conf_matrix = confusion_matrix(y_true=ground_truth_classes, y_pred=predicted_classes)
conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                          display_labels=CLASS_NAMES)
conf_matrix_disp.plot(xticks_rotation=45)
plt.tight_layout()
# save the confusion matrix
plt.savefig('confusion_matrix.png')

# define the plot for the Precision-Recall curve
_, ax = plt.subplots(figsize=(7, 8))
stats = []
# for each class compute the precision, recall, accuracy, F1-score and plot the Precision-Recall curve
for class_id in range(1, len(CLASS_NAMES)):
    class_name = CLASS_NAMES[class_id]
    class_ground_truth = np.array(ground_truth_classes) == class_id
    predicted_class_correct = np.array(predicted_classes) == class_id
    n_instances = np.sum(class_ground_truth)
    true_positives = np.sum(np.logical_and(class_ground_truth, predicted_class_correct))
    false_positives = np.sum(np.logical_and(~class_ground_truth, predicted_class_correct))
    false_negatives = np.sum(np.logical_and(class_ground_truth, ~predicted_class_correct))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    accuracy = true_positives / n_instances if n_instances > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    confidences = np.array(probabilities)[class_ground_truth]
    precision_curve, recall_curve, _ = precision_recall_curve(predicted_class_correct[class_ground_truth], confidences)

    pr_curve = PrecisionRecallDisplay(precision=precision_curve, recall=recall_curve)
    pr_curve.plot(ax=ax, name=f"Class {class_name}")
    stats.append({
        'class': class_name,
        'n_instances': n_instances,
        'precision': precision,
        'accuracy': accuracy,
        'recall': recall,
        'f1-score': f1_score,
    })

# set the labels and save the plot
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, fancybox=True, shadow=True)
plt.savefig('PR_curve.png')

# compute the metrics for all the classes
class_name = "all"
class_ground_truth = np.array(ground_truth_classes) > 0
n_instances = np.sum(class_ground_truth)
# get precision, recall and accuracy by Weighted-Averaging, so by averaging the metrics for each class and weighted by the number of instances
precision = np.sum([stat['precision'] * stat['n_instances'] for stat in stats]) / n_instances if n_instances > 0 else 0
recall = np.sum([stat['recall'] * stat['n_instances'] for stat in stats]) / n_instances if n_instances > 0 else 0
accuracy = np.sum([stat['accuracy'] * stat['n_instances'] for stat in stats]) / n_instances if n_instances > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

stats.append({
    'class': class_name,
    'n_instances': n_instances,
    'precision': precision,
    'accuracy': accuracy,
    'recall': recall,
    'f1-score': f1_score,
})

# save the metrics to a CSV file
stats = pd.concat([pd.DataFrame([stat]) for stat in stats], ignore_index=True)
stats.to_csv('stats.csv', index=False)