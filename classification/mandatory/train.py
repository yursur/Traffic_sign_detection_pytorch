import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import matplotlib.pyplot as plt

from models import create_model
from datasets import train_loader, test_loader
from config import CLASS_NAME, NUM_EPOCHS, SAVE_MODEL_PATH, SAVE_PLOTS_PATH, CLASSES, DEVICE

print(f"Device: {DEVICE}")

# function for running training iterations
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    """
        Function for training classification model.
    """

    print("Training")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(num_epochs):
        start_epoch = time.time()
        print('-' * 10)
        print(f'Epoch [{epoch+1}/{num_epochs}]')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_loader

            running_loss = 0.0
            running_corrects = 0


            # Iterate over data.
            for batch_id, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics for the batch
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            # statistics for the epoch
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # update loss and accuracy lists for plots
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.cpu().numpy())
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.cpu().numpy())

            # deep copy and save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"{SAVE_MODEL_PATH}/{CLASS_NAME}_classification_model.pth")
                print("MODEL SAVED!")

        # create two subplots: for loss and for accuracy
        fig_1, loss_ax = plt.subplots()
        loss_ax.plot(train_loss, color='blue', label='train', marker='o')
        loss_ax.plot(val_loss, color='red', label='val', marker='o')
        loss_ax.set_xlabel('epochs')
        loss_ax.set_ylabel('loss')
        loss_ax.legend()

        fig_2, acc_ax = plt.subplots()
        acc_ax.plot(train_acc, color='blue', label='train', marker='o')
        acc_ax.plot(val_acc, color='red', label='val', marker='o')
        acc_ax.set_xlabel('epochs')
        acc_ax.set_ylabel('accuracy')
        acc_ax.legend()

        fig_1.savefig(f"{SAVE_PLOTS_PATH}/loss_plot.png")
        fig_2.savefig(f"{SAVE_PLOTS_PATH}/acc_plot.png")
        print(f"PLOTS SAVED!")

        time_epoch = time.time() - start_epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] completed in {time_epoch // 60:.0f}m {time_epoch % 60:.0f}s")
        plt.close('all')

    time_training = time.time() - since
    print(f'\nTraining complete in {time_training // 60:.0f}m {time_training % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # initialize the model and move to the device
    model = create_model(num_classes=len(CLASSES))
    model = model.to(DEVICE)
    # get the models parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define CRITERION
    criterion = nn.CrossEntropyLoss()
    # define the OPTIMIZER
    optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    # decay LR by a factor of 'gamma' every 'step_size' epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # train the model
    trained_model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

