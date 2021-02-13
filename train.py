
import config
import numpy as np

def train_ssd(model, optimizer, n_epoch, train_data, valid_data=None, show_log=False):

    n_batch = len(train_data)
    if valid_data is not None:
        n_batch_v = len(valid_data)
    log = np.zeros([n_epoch, 6])

    for epoch in range(n_epoch):

        batch_cl_pos_loss = 0.
        batch_cl_neg_loss = 0.
        batch_loc_loss = 0.

        for images, labels, boxes in train_data:

            optimizer.zero_grad()
            images = images.to(model.device)
            labels = labels.to(model.device)
            boxes = boxes.to(model.device)
            model(images)
            cl_pos_loss, cl_neg_loss, loc_loss = model.backward(labels, boxes)
            batch_cl_pos_loss += cl_pos_loss
            batch_cl_neg_loss += cl_neg_loss
            batch_loc_loss += loc_loss
            optimizer.step()

        log[epoch, :3] = [batch_cl_pos_loss / float(n_batch),
                          batch_cl_neg_loss / float(n_batch),
                          batch_loc_loss / float(n_batch)]

        if valid_data is not None:

            batch_cl_pos_loss = 0.
            batch_cl_neg_loss = 0.
            batch_loc_loss = 0.

            for images, labels, boxes in valid_data:
                with torch.no_grad():
                    images = images.to(model.device)
                    labels = labels.to(model.device)
                    boxes = boxes.to(model.device)
                    model(images)
                    cl_pos_loss, cl_neg_loss, loc_loss = model.backward(labels, boxes)
                    batch_cl_pos_loss += cl_pos_loss
                    batch_cl_neg_loss += cl_neg_loss
                    batch_loc_loss += loc_loss

            log[epoch, 3:6] = [batch_cl_pos_loss / float(n_batch_v),
                               batch_cl_neg_loss / float(n_batch_v),
                               batch_loc_loss / float(n_batch_v)]


        if show_log:
            print('Batch #{:d}'.format(epoch+1))
            print('Positive Classification: {:.2f}'.format(log[epoch, 0]))
            print('Negative Classification: {:.2f}'.format(log[epoch, 1]))
            print('Location Loss {:.2f}'.format(log[epoch, 2]))

            if valid_data is not None:
                print('Validation')
                print('Positive Classification: {:.2f}'.format(log[epoch, 3]))
                print('Negative Classification: {:.2f}'.format(log[epoch, 4]))
                print('Location Loss {:.2f}'.format(log[epoch, 5]))

    return log

if __name__ == '__main__':

    import argparse
    import data
    import model
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    import torch
    import os

    parser = argparse.ArgumentParser(description='SSD training')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--data_dir', type=str, help='path to the dataset', required=True)
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=config.EPOCHS)
    parser.add_argument('--batch_size', type=int, help='batch size', default=config.BATCH_SIZE)
    parser.add_argument('--export', type=str, help='path to export model.pth', required=True)
    parser.add_argument('--verbose', help='verbose output', action='store_false')

    args = parser.parse_args()

    dataset = data.TrashNetDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    net = model.MobileNetV2_SSD().to(config.DEVICE)
    if args.weights is not None:
        net.load_state_dict(torch.load(args.model, map_location=config.DEVICE))
    net.train()
    optimizer = Adam(net.parameters(), lr=config.LEARNING_RATE)
    train_ssd(net, optimizer, args.n_epoch, loader, show_log=args.verbose)

    parsed_path = args.export.split('/')
    path = '.'
    for dir in parsed_path[:-1]:
        path = os.path.join(path, dir)
        if not os.path.exists(path):
            os.makedirs(path)

    torch.save(net.state_dict(), args.export)
