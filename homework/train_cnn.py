from models import CNNClassifier, save_model
from utils import load_data, ToTensor, Transform
import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from torchvision import transforms
import argparse


def train(args):
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.n_epochs
    layers = args.layers[0]
    layers = [2**(i+5) for i in range(layers)]
    print(layers)


    # Paths to data
    #local_train_path = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\UTAustin_hw3\data\train"
    #local_valid_path = r"C:\Users\Will\OneDrive\Desktop\State Farm\UT Austin Deep Learning\UTAustin_hw3\data\valid"
    ubuntu_train_path = r"/home/william/Desktop/UT_Austin_Computer_Vision/UTAustin_hw3/data/train"
    ubuntu_valid_path = r"/home/william/Desktop/UT_Austin_Computer_Vision/UTAustin_hw3/data/valid"
    #colab_train_path = r"/content/UTAustin_hw3/data/train"
    #colab_valid_path = r"/content/UTAustin_hw3/data/valid"


    # Data loading
    train_loader = load_data(dataset_path=ubuntu_train_path, num_workers=4, batch_size=batch_size,
                             args=args, transform=Transform())

    valid_loader = load_data(dataset_path=ubuntu_valid_path, num_workers=4, batch_size=batch_size,
                             args=args, transform=None)

    model = CNNClassifier(layers=layers, normalize=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.90)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    global_steps = 0
    epoch_loss = [100]
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_total_samples = 0.0
        train_total_correct = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_logger.add_scalar('train', loss, global_steps)
            loss.backward()
            optimizer.step()
            global_steps += 1
            train_loss += loss.item()

            # Get accuracy
            _, predicted = torch.max(outputs, 1)
            train_total_samples += labels.size(0)
            train_total_correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}")
        print(f"Train Accuracy: {train_total_correct / train_total_samples}")

        # Validation
        model.eval()
        val_loss = 0.0
        valid_total_correct = 0.0
        valid_total_samples = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

                # Get accuracy
                _, predicted = torch.max(outputs, 1)
                valid_total_samples += labels.size(0)
                valid_total_correct += (predicted == labels).sum().item()


        valid_logger.add_scalar('valid', val_loss/len(valid_loader), epoch)
        print(f"Validation Loss: {val_loss / len(valid_loader):.4f}")
        print(f"Validation Accuracy: {valid_total_correct/valid_total_samples}")

        # Save if better than previous models
        if val_loss/len(valid_loader) < sorted(epoch_loss)[0]:
            save_model(model)
        epoch_loss.append(val_loss/len(valid_loader))



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # torch_directml.device()
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--rand_crop', type=int, required=True)
    parser.add_argument('--h_flip', type=float, required=True)
    parser.add_argument('--v_flip', type=float, required=True)
    parser.add_argument('--rand_rotate', type=int, required=True)
    parser.add_argument('--brightness', type=float, required=True)
    parser.add_argument('--contrast', type=float, required=True)
    parser.add_argument('--saturation', type=float, required=True)
    parser.add_argument('--hue', type=float, required=True)
    parser.add_argument(
        '--optim',
        choices=['sgd', 'adamw'],
        help='Choose one of the available options: sgd, adamw'
    )

    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        help='A list of integer values'
    )
    args = parser.parse_args()
    train(args)