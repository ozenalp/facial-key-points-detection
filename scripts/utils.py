import matplotlib.pyplot as plt

def plot_facial_points(facialpoints_df_copy):
    fig = plt.figure(figsize=(20, 20))

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1)
        image = plt.imshow(facialpoints_df_copy['Image'][i], cmap='gray')
        for j in range(1, 31, 2):
            plt.plot(facialpoints_df_copy.loc[i][j-1], facialpoints_df_copy.loc[i][j], 'rx')

    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)