from classification.blue_border.datasets import train_dataloader_dict, test_dataloader_dict
from models import create_model
from tqdm.auto import tqdm
import torch
import time
import matplotlib.pyplot as plt
from classification.config import DEVICE, SUBCLASSES_DICT, NUM_EPOCHS, SAVE_MODEL_PATH, SAVE_PLOTS_PATH, EPOCHS_SAVE_MODEL, EPOCHS_SAVE_PLOTS
from utils import Averager
import copy

print(f"Device: {DEVICE}")

## function for running training iterations
def train_model(model, subclass, criterion, optimizer, scheduler, num_epochs):
    """
        Function for training classification model.
    """

    print("Training")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_dataloader_dict[subclass]
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = test_dataloader_dict[subclass]

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

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(outputs)
            epoch_acc = running_corrects.double() / len(outputs)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train(train_loader, model):
    """
    Function for training classification model.
    """

    print("Training")

    ## initialize tqdm progress bar
    prog_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, labels = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        ## Casts operations to mixed precision
        # with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        #     loss_dict = models(images, targets)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        ## Scales the loss, and calls backward()
        ## to create scaled gradients for MIXED PRECISION training
        # scaler.scale(losses).backward()
        losses.backward()
        ## Unscales gradients and calls optimizer.step()
        # scaler.step(optimizer)
        optimizer.step()
        ## Updates the scale for next iteration
        # scaler.update()
        train_itr += 1

        ## update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


## function for running validation iterations
def validate(valid_data_loader, model):
    """
    Function for validating detection model.
    """

    print("Validating")
    global val_itr
    global val_loss_list

    ## initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        ## update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == '__main__':
    ## initialize the models and move to the device
    for subclass in SUBCLASSES_DICT:
        model = create_model(subclass=subclass)
        model = model.to(DEVICE)
        ## get the models parameters
        params = [p for p in model.parameters() if p.requires_grad]
        ## define the OPTIMIZER
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
        ## initialize the Averager class
        train_loss_hist = Averager()
        val_loss_hist = Averager()
        train_itr = 1
        val_itr = 1
        ## train and validation loss lists to store loss values and plots graphs for all iterations
        train_loss_list = []
        val_loss_list = []
        ## start the training epochs
        for epoch in range(NUM_EPOCHS):
            print(f"\nEPOCH [{epoch + 1} / {NUM_EPOCHS}]")
            ## reset the training and validation loss histories for the current epoch
            train_loss_hist.reset()
            val_loss_hist.reset()
            ## create two subplots: for training and for validation
            fig_1, train_ax = plt.subplots()
            fig_2, valid_ax = plt.subplots()
            ## start timer and carry out training and validation
            start = time.time()
            train_loss = train(train_loader, model)
            val_loss = validate(test_loader, model)
            print(f"Train loss: {train_loss_hist.value:.3f} \n Validation loss: {val_loss_hist.value:.3f}")
            end = time.time()
            print(f"Took {((end - start) / 60):.2f} minutes")
            if (epoch + 1) % EPOCHS_SAVE_MODEL == 0:
                ## save models after every n epochs
                torch.save(model.state_dict(), f"{SAVE_MODEL_PATH}/detection_model{epoch + 1}.pth")
                print("MODEL SAVED...")

            if (epoch + 1) % EPOCHS_SAVE_PLOTS == 0:
                ## save loss plots after n epochs
                train_ax.plot(train_loss, color='blue')
                train_ax.set_xlabel('iterations')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('iterations')
                valid_ax.set_ylabel('validation loss')
                fig_1.savefig(f"{SAVE_PLOTS_PATH}/train_loss_{epoch + 1}.png")
                fig_2.savefig(f"{SAVE_PLOTS_PATH}/valid_loss_{epoch + 1}.png")
                print("PLOTS SAVED...")

            if (epoch + 1) == NUM_EPOCHS:
                ## save loss plots and models once at the end
                train_ax.plot(train_loss, color='blue')
                train_ax.set_xlabel('iterations')
                train_ax.set_ylabel('train loss')
                valid_ax.plot(val_loss, color='red')
                valid_ax.set_xlabel('iterations')
                valid_ax.set_ylabel('validation loss')
                fig_1.savefig(f"{SAVE_PLOTS_PATH}/train_loss_{epoch + 1}.png")
                fig_2.savefig(f"{SAVE_PLOTS_PATH}/valid_loss_{epoch + 1}.png")
                torch.save(model.state_dict(), f"{SAVE_MODEL_PATH}/models{epoch + 1}.pth")

            plt.close('all')