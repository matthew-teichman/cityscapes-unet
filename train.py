import os
import time
import torch
from evaluate import evaluation_score


# training loop
def train(path, model, train_loader, valid_loader, loss_function, optimizer, device, seed, num_epochs=30):
    # reproducible results
    torch.manual_seed(seed)

    # send model to device
    model.to(device)
    model.train()

    start_time = time.time()

    # training loop
    train_loss, val_loss, val_dice, val_iou, val_precision, val_recall = [], [], [], [], [], []
    for epoch in range(num_epochs):
        # enable model to train
        model.train()
        # training loop per epoch
        losses = 0.0
        total_dice, total_iou = 0.0, 0.0
        total_precision, total_recall = 0.0, 0.0
        for i, (imgs, masks) in enumerate(train_loader):
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
            losses += loss.detach().data.item()
            dice, iou, precision, recall, _ = evaluation_score(outputs.detach(), masks.detach())
            total_dice += dice
            total_iou += iou
            total_precision += precision
            total_recall += recall
        train_loss.append(losses / (i + 1))
        total_dice = total_dice / (i + 1)
        total_iou = total_iou / (i + 1)
        total_precision = total_precision / (i + 1)
        total_recall = total_recall / (i + 1)

        # enable model evaluation
        model.eval()

        print(
            "Epoch: {} - Training Losses: {:.4f} - Dice Score: {:.4f} - IoU Score: {:.4f}"
            " - Precision: {:.4f} - Recall: {:.4f}".format(
                epoch + 1,
                train_loss[epoch],
                total_dice,
                total_iou,
                total_precision,
                total_recall
            ))

        # Saving Model
        torch.save(model.state_dict(), os.path.join(path, "saved_model_epoch" + str(epoch) + ".pth"))

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = elapsed_time / (60.0 * 60.0)
    print("Total training time elapsed: {:.2f} seconds".format(elapsed_time))
    print("Done!")
    return model
