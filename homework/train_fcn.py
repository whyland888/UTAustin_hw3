import torch
import numpy as np

from models import FCN, save_model
from utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
import dense_transforms
import torch.utils.tensorboard as tb


def train(args):

    # Logging
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Hyperparameters
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.n_epochs

    # Paths to data
    ubuntu_train_path = r"/home/william/Desktop/UT_Austin_Computer_Vision/UTAustin_hw3/dense_data/train"
    ubuntu_valid_path = r"/home/william/Desktop/UT_Austin_Computer_Vision/UTAustin_hw3/dense_data/valid"
    colab_train_path = r"/content/UTAustin_hw3/dense_data/train"
    colab_valid_path = r"/content/UTAustin_hw3/dense_data/valid"

    # Data loading
    # train_loader = load_dense_data(dataset_path=ubuntu_train_path)
    # valid_loader = load_dense_data(dataset_path=ubuntu_valid_path)
    train_loader = load_dense_data(dataset_path=colab_train_path)
    valid_loader = load_dense_data(dataset_path=colab_valid_path)

    # Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FCN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training Loop
    global_steps = 0
    epoch_loss = [100]

    for epoch in range(num_epochs):
        print(epoch)
        confusion_matrix = ConfusionMatrix()
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.squeeze()
            # print(np.shape(labels[0]))
            # preds = outputs.argmax(1)
            loss = criterion(outputs, labels)
            train_logger.add_scalar('train', loss, global_steps)
            loss.backward()
            optimizer.step()
            global_steps += 1
            train_loss += loss.item()

            # Get accuracy
            confusion_matrix.add(preds=outputs.argmax(1), labels=labels)

        print(f"Training performance for epoch {epoch}: {confusion_matrix.iou}")

        # Validation
        model.eval()
        confusion_matrix = ConfusionMatrix()
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                # Get accuracy
                confusion_matrix.add(preds=outputs.argmax(1), labels=labels)

        print(f"Validation performance for epoch {epoch}: {confusion_matrix.iou}")

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--n_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    args = parser.parse_args()
    train(args)
