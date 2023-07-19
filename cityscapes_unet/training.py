import os
import time
import torch
from tqdm import tqdm
from cityscapes_unet.evaluate import evaluation_score


# training loop
def train(path, model, train_loader, valid_loader, loss_function, optimizer, device, seed, num_epochs=30):
    # reproducible results
    torch.manual_seed(seed)

    # send model to device
    model.to(device)
    model.train()

    start_time = time.time()

    # training loop
    train_loss = []
    valid_loss = []
    for epoch in range(num_epochs):
        # enable model to train
        model.train()
        # training loop per epoch
        losses_train = 0.0
        total_dice_train, total_iou_train = 0.0, 0.0
        total_precision_train, total_recall_train = 0.0, 0.0
        for i, (imgs, masks) in tqdm(enumerate(train_loader), unit="batch", total=len(train_loader)):
            imgs, masks = imgs.to(device), masks.to(device)
            # reset backprogation values
            optimizer.zero_grad()
            # forward pass through model
            outputs = model(imgs)
            # calculate loss
            loss = loss_function(outputs, masks.float())
            # backpropagate loss
            loss.backward()
            # update optimizer
            optimizer.step()
            # collect loss values
            losses_train += loss.detach().data.item()
            dice, iou, precision, recall = evaluation_score(outputs.detach(), masks.detach())
            total_dice_train += dice
            total_iou_train += iou
            total_precision_train += precision
            total_recall_train += recall
        # enable model evaluation
        model.eval()

        train_loss.append(losses_train / (i + 1e-7))
        total_dice_train = total_dice_train / (i + 1e-7)
        total_iou_train = total_iou_train / (i + 1e-7)
        total_precision_train = total_precision_train / (i + 1e-7) 
        total_recall_train = total_recall_train / (i + 1e-7)

        print(
            "Epoch: {} - Training Losses: {:.4f} - Dice Score: {:.4f} - IoU Score: {:.4f} - Precision: {:.4f} - Recall: {:.4f}".format(
                epoch + 1,
                train_loss[epoch],
                total_dice_train,
                total_iou_train,
                total_precision_train,
                total_recall_train
        ))
    
        losses_valid = 0.0
        total_dice_valid, total_iou_valid = 0.0, 0.0
        total_precision_valid, total_recall_valid = 0.0, 0.0
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(valid_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                # forward pass through model
                outputs = model(imgs)
                # calculate loss
                loss = loss_function(outputs, masks.float())
                losses_valid += loss.detach().data.item()
                dice, iou, precision, recall = evaluation_score(outputs.detach(), masks.detach())
                total_dice_valid += dice
                total_iou_valid += iou
                total_precision_valid += precision
                total_recall_valid += recall
            valid_loss.append(losses_valid / (i + 1e-7))
            total_dice_valid = total_dice_valid / (i + 1e-7)
            total_iou_valid = total_iou_valid / (i + 1e-7)
            total_precision_valid = total_precision_valid / (i + 1e-7) 
            total_recall_valid = total_recall_valid / (i + 1e-7)

        print(
            "Epoch: {} - Validation Losses: {:.4f} - Dice Score: {:.4f} - IoU Score: {:.4f} - Precision: {:.4f} - Recall: {:.4f}".format(
                epoch + 1,
                valid_loss[epoch],
                total_dice_valid,
                total_iou_valid,
                total_precision_valid,
                total_recall_valid
        ))

        # Saving Model
        torch.save(model.state_dict(), os.path.join(path, "saved_model_epoch" + str(epoch) + ".pth"))

    elapsed_time = time.time() - start_time
    elapsed_time = elapsed_time / (60.0 * 60.0)
    print("Total training time elapsed: {:.2f} hours".format(elapsed_time))
    print("Done!")
    return model
