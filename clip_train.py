from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import clip
import os
import cv2

import wandb
from tqdm import tqdm
import random

enable_wandb = True

# Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)

EPOCH = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)

optimizer = optim.Adam(clip_model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

if enable_wandb:
    wandb.init(
        project="CLIP training",
    )

CLASS_NAMES = [
    'background',
    'cystic plate',
    'calot Triangle',
    'cystic artery',
    'cystic duct',
    'gallbladder',
    'tool'
]

def segment_image(image, bbox):
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    segmented_image_array = np.zeros_like(image_array)
    x1, y1, x2, y2 = bbox
    segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new('RGB', image.size, (255, 255, 255))

    transparency_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)
    transparency_mask[y1:y2, x1:x2] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def parseFile(file, img, flip_h=False, flip_v=False):

    # read the labels file lines
    with open(file) as f:
        labels_str = f.readlines()

    classes = []
    cropped_imgs = []
    for i in range(len(labels_str)):
        label = labels_str[i].split(' ')
        segment = np.array([
            np.array(label[1::2], dtype=np.float64) * float(img.size[0]),
            np.array(label[2::2], dtype=np.float64) * float(img.size[1])
        ], dtype=np.int64).T

        if flip_h:
            segment[:, 0] = img.size[0] - segment[:, 0]

        if flip_v:
            segment[:, 1] = img.size[1] - segment[:, 1]

        # find bbox
        x_min, y_min = np.min(segment, axis=0)
        x_max, y_max = np.max(segment, axis=0)

        cropped_image = segment_image(img, (x_min, y_min, x_max, y_max))

        classes.append(int(label[0]))
        cropped_imgs.append(cropped_image)

    return classes, cropped_imgs

# for each image in ./dataset/images/train/
images_train = os.listdir('./dataset/images/train/')
images_val = os.listdir('./dataset/images/val/')
for ep in range(EPOCH):
    
    losses = []
    random.shuffle(images_train)

    clip_model.train()
    for image in tqdm(images_train):
        label = image.replace('.jpg', '.txt')

        try:
            image_input = Image.open(f'./dataset/images/train/{image}')
            image_input = image_input.convert("RGB")
        except:
            continue
        
        flip_h = random.choice([True, False])
        if flip_h:
            image_input = image_input.transpose(Image.FLIP_LEFT_RIGHT)

        flip_v = random.choice([True, False])
        if flip_v:
            image_input = image_input.transpose(Image.FLIP_TOP_BOTTOM)
        
        try:
            classes, cropped_imgs = parseFile(f'./dataset/labels/train/{label}', image_input,
                                    flip_h=flip_h, flip_v=flip_v)
        except:
            continue
        
        if len(classes) == 0:
            continue

        optimizer.zero_grad()

        preprocessed_images = [preprocess(image).to(device) for image in cropped_imgs]
        stacked_images = torch.stack(preprocessed_images)

        tokenized_text = clip.tokenize([f"a photo of a {CLASS_NAMES[class_id]}" for class_id in classes]).to(device)

        logits_per_image, logits_per_text = clip_model(stacked_images, tokenized_text)

        ground_truth = torch.arange(len(stacked_images)).to(device)
        
        loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        
        loss.backward()

        losses.append(loss.item())

        optimizer.step()


    mean_loss = np.mean(losses)

    print("Epoch:", ep, "Mean loss:", mean_loss)
    if enable_wandb:
        wandb.log({'loss': mean_loss}, commit=False)




    clip_model.eval()

    tokenized_texts = clip.tokenize([f"a photo of a {CLASS_NAMES[class_id]}" for class_id in range(1, len(CLASS_NAMES))]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized_texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    total_examples = {}
    top_1_correct = {}
    top_3_correct = {}
    total_predictions = {}
    true_binary = {}
    pred_probs = {}
    for class_id in range(1, len(CLASS_NAMES)):
        total_examples[class_id] = 0
        top_1_correct[class_id] = 0
        top_3_correct[class_id] = 0
        total_predictions[class_id] = 0
        true_binary[class_id] = []
        pred_probs[class_id] = []

    for image in tqdm(images_val):
        label = image.replace('.jpg', '.txt')

        try:
            image_input = Image.open(f'./dataset/images/val/{image}')
            image_input = image_input.convert("RGB")
        except:
            continue
        
        try:
            classes, cropped_imgs = parseFile(f'./dataset/labels/val/{label}', image_input,
                                    flip_h=False, flip_v=False)
        except:
            continue
        
        if len(classes) == 0:
            continue

        with torch.no_grad():
            preprocessed_images = [preprocess(image).to(device) for image in cropped_imgs]
            stacked_images = torch.stack(preprocessed_images)
            image_features = clip_model.encode_image(stacked_images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values_3, indices_3 = probs.topk(3)
            indices_3 = indices_3 + 1

            for i in range(len(classes)):
                total_examples[classes[i]] += 1
                total_predictions[indices_3[i][0].item()] += 1
                if classes[i] in indices_3[i]:
                    top_3_correct[classes[i]] += 1
                if classes[i] == indices_3[i][0]:
                    top_1_correct[classes[i]] += 1
                    true_binary[classes[i]].append(1)
                    pred_probs[classes[i]].append(values_3[i][0].item())
                else:
                    true_binary[classes[i]].append(0)
                    pred_probs[classes[i]].append(
                        # find the probability of the true class using indices_3
                        probs[i][classes[i]-1].item()
                    )

    top_1_accuracy = sum(top_1_correct.values()) / sum(total_examples.values())
    top_3_accuracy = sum(top_3_correct.values()) / sum(total_examples.values())
    
    if wandb:
        wandb.log({
            'top_1_accuracy': top_1_accuracy,
            'top_3_accuracy': top_3_accuracy
        }, commit=True)
    print("Top 1 Accuracy:", top_1_accuracy, "Top 3 Accuracy:", top_3_accuracy)

    torch.save({
        'epoch': ep,
        'model_state_dict': clip_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': mean_loss,
        }, f"clip_checkpoint/{ep}.pt")

if enable_wandb:
    wandb.finish()