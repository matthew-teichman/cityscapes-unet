import time
import copy
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# evaluation metrics
def evaluation_score(predictions, targets, eps=1e-7, threshold=0.6):
    predictions = predictions.cpu()
    targets = targets.cpu()
    # calculating dice score
    dice_intersection = (predictions * targets).sum(dim=(2, 3))
    dice = (2. * dice_intersection + eps) / (predictions.pow(2).sum(dim=(2, 3)) + targets.pow(2).sum(dim=(2, 3)) + eps)
    dice = dice.mean()
    # calculating iou score
    inp = copy.copy(predictions.view(-1))
    target = copy.copy(targets.view(-1))
    iou_intersection = (inp * target).sum()
    iou_total = (inp + target).sum()
    iou_union = iou_total - iou_intersection
    iou = iou_intersection / (iou_union + eps)

    # calculate the true positives, false positives, and false negatives
    target = target.int()
    predictions = predictions.int()
    mask_pred = predictions.flatten()
    mask_true = target.flatten()
    mask_pred = mask_pred.numpy()
    mask_true = mask_true.numpy()

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(mask_true, mask_pred, labels=[0, 1]).ravel()

    # Calculate precision and recall
    precision = tp / (tp + fp + 1)
    recall = tp / (tp + fn + 1)
    return dice, iou, precision, recall, predictions


# the function plots training loss or error
def plot_data(training_data, validation_data, title, x_label, y_label, legend1, legend2):
    plt.title(title)
    n = len(training_data)  # number of epoch
    plt.plot(range(1, n + 1), training_data, label=legend1)
    plt.plot(range(1, n + 1), validation_data, label=legend2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.show()


# validation test
def validation(model, device, valid_dataloader, loss_function):
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    for i, (img, mask) in enumerate(valid_dataloader):
        img, mask = img.to(device), mask.to(device)
        outputs = model(img)
        loss = loss_function(outputs, mask.float())
        dice, iou, precision, recall, _ = evaluation_score(outputs, mask.float())
        total_loss += loss.data.item()
        total_dice += dice
        total_iou += iou
        total_precision += precision
        total_recall += recall
    total_loss = float(total_loss / (i + 1))
    total_dice = float(total_dice / (i + 1))
    total_iou = float(total_iou / (i + 1))
    total_precision = float(total_precision / (i + 1))
    total_recall = float(total_recall / (i + 1))
    return total_loss, total_dice, total_iou, total_precision, total_recall


def visual(img, mask, threshold):
    img = img.cpu().numpy()
    mask = mask.cpu().numpy()
    mask = np.where(mask > threshold, 1, 0).astype(np.uint8)

    img = np.squeeze(img)
    mask = np.squeeze(mask)

    img = np.transpose(img, [1, 2, 0])

    temp = np.stack((mask, mask, mask), axis=-1)
    temp = temp.astype(np.float32)
    image = cv2.addWeighted(img, 0.8, temp, 0.2, 0.0)

    plt.imshow(img)
    plt.show()
    plt.imshow(mask)
    plt.show()
    plt.imshow(image)
    plt.show()


def testing(model, device, test_dataloader, threshold=0.7):
    for i, (img, mask) in enumerate(test_dataloader):
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            output = model(img)
            duration = time.time() - start_time
            # if output probability is greater than threshold it becomes 1
            gen_mask = (output > threshold)
            dice, iou, precision, recall, mask = evaluation_score(gen_mask, mask)
        print("Image: " + str(i) + " - Dice: " + str(float(dice)) + " IoU: " + str(float(iou))
              + " Precision: " + str(float(precision)) + " - Recall: " + str(float(recall)) + "Time: " + str(
            duration * 1000) + "ms")
        visual(img, gen_mask, threshold)
    print("Done!")



