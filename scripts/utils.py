import pandas as pd
import matplotlib.pyplot as plt

def plot_facial_points(facialpoints_df_copy):
    # Create a new matplotlib figure with a size of 20x20
    fig = plt.figure(figsize=(20, 20))

    # Loop over the first 16 images in the DataFrame
    for i in range(16):
        # Add a subplot to the figure in a 4x4 grid at position i + 1
        ax = fig.add_subplot(4, 4, i + 1)
        
        # Display the i-th image in grayscale
        image = plt.imshow(facialpoints_df_copy['Image'][i], cmap='gray')

        # For every pair of x, y coordinates in the row, plot them on the image
        # We use 'rx' to denote that the points should be marked with red x's
        for j in range(1, 31, 2):
            plt.plot(facialpoints_df_copy.loc[i][j-1], facialpoints_df_copy.loc[i][j], 'rx')

    # Display the figure with all subplots
    plt.show()



def count_parameters(model):
    # Sum up the number of elements (parameters) in each tensor of the model's parameters
    # Only consider parameters that require gradients (i.e., trainable parameters)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def df_results(epochs, train_losses, valid_losses, name):
    # Create a Pandas DataFrame to store the training and validation losses for each epoch
    df = pd.DataFrame({
        # The 'epoch' column records the epoch number, starting from 1 and going up to 'epochs' inclusive
        'epoch': range(1, epochs+1),
        # The 'train_loss' column records the training loss for each epoch
        'train_loss': train_losses,
        # The 'valid_loss' column records the validation loss for each epoch
        'valid_loss': valid_losses,
    })
    
    # Save the DataFrame to a CSV file. The file is saved in the '../results/' directory
    # and its name is given by the 'name' parameter followed by '.csv'
    df.to_csv('../results/' + name + '.csv', index=False)
    
    # Return the DataFrame
    return df