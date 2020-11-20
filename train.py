
from os.path import join
import argparse
import numpy as np
import pandas as pd
import os
import torch

from pathlib2 import Path
import tqdm
from CNNpred2D import CNNpred
from processing_data import costruct_data_warehouse, cnn_data_sequence_pre_train, cnn_data_sequence, transforming_data_warehouse
from dataset import WholeDataset, generate_batches
from sklearn.metrics import accuracy_score as accuracy, f1_score
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(args, model, dataset):
    model.eval()
    loss_fcn = torch.nn.BCELoss()

    data_dataloader = generate_batches(dataset, args.batch_size, n_workers=args.num_workers)


    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for batch_data, batch_label in data_dataloader:
            batch_logit = model(batch_data).view(-1)

            loss = loss_fcn(batch_logit, batch_label)

            pred = (batch_logit > 0.5).int()

            pred_list.extend(pred)
            label_list.extend(batch_label)

            loss_list.append(loss.item())

        loss_data = np.array(loss_list).mean()
        acc = accuracy(pred_list, label_list)
        f1 = f1_score(pred_list, label_list, average='macro')

    return loss_data, acc, f1,



def train(args, train_dataset, val_dataset, test_dataset, i):

    my_file = Path(join(args.Base_dir,
                        '2D-models/best-{}-{}-{}-{}-{}.pt'.format(args.epochs, args.seq_len, args.num_filter, args.drop, i)))
    filepath = join(args.Base_dir, '2D-models/best-{}-{}-{}-{}-{}.pt'.format(args.epochs, args.seq_len, args.num_filter, args.drop, i))
    if my_file.is_file() and args.override == False:
        print('loading model')

    else:

        model = CNNpred(args.number_feature, args.num_filter, args.drop)
        cur_step = 0
        best_f1 = -1

        loss_fcn = torch.nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=20,
                                                               threshold=0.001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0.00001,
                                                               eps=1e-08,
                                                               verbose=True)

        start_epoch = 0

        for epoch in range(start_epoch, args.epochs):
            model.eval()
            loss_list = []
            pred_list = []
            label_list = []
            train_dataloader = generate_batches(train_dataset, args.batch_size, n_workers=args.num_workers)

            for batch_data, batch_label in train_dataloader:
                batch_logit = model(batch_data).view(-1)

                loss = loss_fcn(batch_logit, batch_label)

                pred = (batch_logit > 0.5).int()

                pred_list.extend(pred)
                label_list.extend(batch_label)

                optimizer.zero_grad()
                loss.backward()
                parameters = list(model.parameters())
                optimizer.step()
                loss_list.append(loss.item())

            loss_data = np.array(loss_list).mean()
            train_acc = accuracy(pred_list, label_list)
            train_f1 = f1_score(pred_list, label_list, average='macro')

            print("Epoch {:05d}\n"
                  "Train: loss: {:.4f} | accuracy: {:.4f} | f-acore: {:.4f}"
                  .format(epoch + 1, loss_data, train_acc, train_f1))


            scheduler.step(loss_data)

            if wandb.run is not None:  # is initialized
                wandb.log(
                    {'train/loss': loss_data,
                     'train/accuracy': train_acc,
                     'train/macro_f1': train_f1,
                     },
                    step=epoch + 1
                )

            if (epoch + 1) % 1 == 0:
                val_loss, val_acc, val_f1 = validate(args, model, val_dataset)
                print("Validation:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
                      .format(val_loss, val_acc, val_f1))

                if wandb.run is not None:  # is initialized
                    wandb.log(
                        {'validation/loss': val_loss,
                         'validation/accuracy': val_acc,
                         'validation/macro_f1': val_f1,
                         },
                        step=epoch + 1
                    )

                # choosing best model according to best validation accuracy
                if best_f1 < val_f1:
                    best_f1 = val_f1
                    cur_step = 0
                    torch.save(model, filepath)

                else:
                    cur_step += 1
                    if cur_step == args.patience:
                        break

    model = torch.load(filepath)
    model = model.to(device)

    print('results of best model')

    train_loss, train_acc, train_f1 = validate(args, model, train_dataset)
    print("Train:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
          .format(train_loss, train_acc, train_f1))

    val_loss, val_acc, val_f1 = validate(args, model, val_dataset)
    print("Validation:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
          .format(val_loss, val_acc, val_f1))

    test_loss, test_acc, test_f1 = validate(args, model, test_dataset)
    print("Test:  loss: {:.4f} | accuracy: {:.4f} | f1: {:.4f}"
          .format(test_loss, test_acc, test_f1))

    print('---------------')

    return model


def prediction(args, data_loaders_warehouse, model, order_stocks, cnn_results):
    for name in order_stocks:
        value = data_loaders_warehouse[name]
        test_data = value[1]

        cnn_results.append(validate(args, model, test_data, 'test')[2])

    return cnn_results

def saving_results(args, cnn_results, order_stocks):
    cnn_results = np.array(cnn_results)
    cnn_results = cnn_results.reshape(args.num_iter - 1, len(order_stocks))
    cnn_results = pd.DataFrame(cnn_results, columns=order_stocks)
    cnn_results = cnn_results.append([cnn_results.mean(), cnn_results.max(), cnn_results.std()], ignore_index=True)
    cnn_results.to_csv(join(args.Base_dir, '2D-models/new results.csv'), index=False)


def main(args):
    if args.use_wandb:
        wandb.init(project='CNNpred', config=args)

    TRAIN_ROOT_PATH = join(args.Base_dir, 'Dataset')
    train_file_names = os.listdir(join(args.Base_dir, 'Dataset'))

    print('Loading train data ...')
    data_warehouse, number_of_stocks, args.number_feature, samples_in_each_stock = \
        costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names, args.predict_day, args.seq_len)

    print('number of stocks = {}'.format(number_of_stocks))
    print('number of features = {}'.format(args.number_feature))
    print('number of samples in each stock = {}'.format(samples_in_each_stock))

    order_stocks = data_warehouse.keys()
    transformed_data_loader_warehouse = transforming_data_warehouse(data_warehouse, order_stocks, args.seq_len)

    cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
        data_warehouse, args.seq_len)

    train_data = WholeDataset(cnn_train_data, cnn_train_target)
    val_data = WholeDataset(cnn_valid_data, cnn_valid_target)
    test_data = WholeDataset(cnn_test_data, cnn_test_target)

    cnn_results = []

    for i in range(1, args.num_iter):

        print('Iteration {}'.format(i))
        model = train(args, train_data, val_data, test_data, i)
        cnn_results = prediction(args, transformed_data_loader_warehouse, model, order_stocks, cnn_results)

    saving_results(args, cnn_results, order_stocks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNNpred')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--Base-dir", type=str, default='',
                        help="Location of Base Directory")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--seq-len", type=int, default=60,
                        help="History of each sample")
    parser.add_argument("--predict-day", type=int, default=1,
                        help="Day ahead prediction")
    parser.add_argument("--num-iter", type=int, default=2,
                        help="number of repeating algorithm")
    parser.add_argument("--num-filter", type=int, default=8,
                        help="number filters in conv layer")
    parser.add_argument("--drop", type=float, default=0.1,
                        help="Fully connected dropout")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, #5e-4,
                        help="weight decay")
    parser.add_argument('--batch-size', type=int, default=128,
                        help="batch size used for training, validation and test")
    parser.add_argument('--patience', type=int, default=200,
                        help="used for early stop")
    parser.add_argument('--use-wandb', default=False, type=bool)
    parser.add_argument('--override', default=False, type=bool,
                        help='overrride the existing models')
    parser.add_argument('--num-workers', default=0, type=int)
    args = parser.parse_args()

    main(args)
