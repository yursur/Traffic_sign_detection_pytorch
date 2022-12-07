from datasets import train_loader, test_loader
from models import create_model
from tqdm.auto import tqdm
import torch
import time
import matplotlib.pyplot as plt
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, SAVE_MODEL_ROOT, SAVE_PLOTS_ROOT, EPOCHS_SAVE_MODEL, EPOCHS_SAVE_PLOTS
from utils import Averager

print(f"Device: {DEVICE}")

# function for running training iterations
def train(train_loader, model):
    print("Training")
    # Creates once at the beginning of training for MIXED PRECISION training
    # scaler = torch.cuda.amp.GradScaler()

    global train_itr
    global train_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(train_loader, total=len(train_loader))

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # Casts operations to mixed precision
        # with torch.amp.autocast(device_type='cpu', dtype=torch.bfloat16):
        #     loss_dict = models(images, targets)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        # Scales the loss, and calls backward()
        # to create scaled gradients for MIXED PRECISION training
        # scaler.scale(losses).backward()
        losses.backward()
        # Unscales gradients and calls optimizer.step()
        # scaler.step(optimizer)
        optimizer.step()
        # Updates the scale for next iteration
        # scaler.update()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model):
    print("Validating")
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
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
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


if __name__ == '__main__':
    # initialize the models and move to the device
    model = create_model(num_classes=NUM_CLASSES, six_class_detection=True)
    model = model.to(DEVICE)
    # get the models parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the OPTIMIZER
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values and plots graphs for all iterations
    train_loss_list = []
    val_loss_list = []
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH [{epoch + 1} / {NUM_EPOCHS}]")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # create two subplots: for training and for validation
        fig_1, train_ax = plt.subplots()
        fig_2, valid_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(test_loader, model)
        print(f"Train loss: {train_loss_hist.value:.3f} \n Validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.2f} minutes")
        if (epoch + 1) % EPOCHS_SAVE_MODEL == 0:  # save models after every n epochs
            torch.save(model.state_dict(), f"{SAVE_MODEL_ROOT}/detection_model{epoch + 1}.pth")
            print("MODEL SAVED...")

        if (epoch + 1) % EPOCHS_SAVE_PLOTS == 0:  # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            fig_1.savefig(f"{SAVE_PLOTS_ROOT}/train_loss_{epoch + 1}.png")
            fig_2.savefig(f"{SAVE_PLOTS_ROOT}/valid_loss_{epoch + 1}.png")
            print("PLOTS SAVED...")

        if (epoch + 1) == NUM_EPOCHS:  # save loss plots and models once at the end
            train_ax.plot(train_loss, color='blue')
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('train loss')
            valid_ax.plot(val_loss, color='red')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('validation loss')
            fig_1.savefig(f"{SAVE_PLOTS_ROOT}/train_loss_{epoch + 1}.png")
            fig_2.savefig(f"{SAVE_PLOTS_ROOT}/valid_loss_{epoch + 1}.png")
            torch.save(model.state_dict(), f"{SAVE_MODEL_ROOT}/models{epoch + 1}.pth")

        plt.close('all')