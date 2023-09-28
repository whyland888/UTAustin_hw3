import torch
from utils import SuperTuxDataset, DenseSuperTuxDataset, load_data, load_dense_data, ToTensor


def get_statistics(dataset_name):

    # SuperTux
    if dataset_name == 'SuperTux':
        path = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\UTAustin_hw3\data\train"
        dataset = SuperTuxDataset(path)
        loader = load_data(path, batch_size=256, transform=ToTensor())

    # DenseSuperTux
    elif dataset_name == 'DenseSuperTux':
        path = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\UTAustin_hw3\dense_data\train"
        dataset = DenseSuperTuxDataset(path)
        loader = load_dense_data(path, batch_size=256)

    else:
        print("Please provide valid dataset name")

    # Sum tensors
    channel_sum = torch.zeros(3)
    channel_sum_of_squares = torch.zeros(3)

    for batch in loader:
        data, _ = batch  # Not interested in labels
        data = data.view(data.size(0), data.size(1), -1)  # Flatten spatial dimensions
        batch_sum = torch.sum(data, dim=2)
        batch_sum_of_squares = torch.sum(data ** 2, dim=2)

        channel_sum += torch.sum(batch_sum, dim=0)
        channel_sum_of_squares += torch.sum(batch_sum_of_squares, dim=0)

    # Calculate mean and std
    num_samples = len(dataset)
    mean = channel_sum / (num_samples * data.size(2))
    std = torch.sqrt(channel_sum_of_squares / (num_samples * data.size(2)))

    print(f"{dataset_name} dataset mean per channel:", mean)
    print(f"{dataset_name} dataset standard deviation per channel:", std)


get_statistics('DenseSuperTux')


