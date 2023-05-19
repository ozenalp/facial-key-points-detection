import torch
from tqdm import tqdm

def train_model(model, dataloader, optimizer, criterion, device):
    # Set the model to training mode
    model.train()

    # Initialize the total loss for this epoch
    epoch_loss = 0

    # Loop over all batches of data provided by the dataloader
    for i, (images, keypoints) in tqdm(enumerate(dataloader), desc="Training", total=len(dataloader), leave=False):
        # Move the images and keypoints to the device that will perform the computation (e.g. a GPU)
        images = images.float().to(device)
        keypoints = keypoints.float().to(device)

        # Clear the gradients from the previous iteration
        optimizer.zero_grad()

        # Feed the images into the model and obtain the predictions
        outputs = model(images)

        # Calculate the loss between the model's predictions and the true keypoints
        loss = criterion(outputs, keypoints)

        # Backpropagate the loss through the model
        loss.backward()
        
        # Update the model's parameters based on the gradients
        optimizer.step()

        # Add to the total loss for this epoch; multiply by batch size because loss is averaged over batch
        epoch_loss += loss.item() * images.size(0)

    # Return the average loss over all samples
    return epoch_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    # Set the model to evaluation mode
    model.eval()

    # Initialize the total loss for this evaluation
    epoch_loss = 0

    # Deactivate autograd engine to disable backpropagation, which reduces memory usage and speeds up computation
    with torch.no_grad():
        # Loop over all batches of data provided by the dataloader
        for i, (images, keypoints) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader), leave=False):
            # Move the images and keypoints to the device that will perform the computation (e.g. a GPU)
            images = images.float().to(device)
            keypoints = keypoints.float().to(device)

            # Feed the images into the model and obtain the predictions
            outputs = model(images)

            # Calculate the loss between the model's predictions and the true keypoints
            loss = criterion(outputs, keypoints)

            # Add to the total loss for this evaluation; multiply by batch size because loss is averaged over batch
            epoch_loss += loss.item() * images.size(0)

    # Return the average loss over all samples
    return epoch_loss / len(dataloader.dataset)