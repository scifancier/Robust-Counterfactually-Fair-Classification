# python3 main.py --n 5 --r 0.0 --bar 0.48 --dataset bank --p 0 --d u2w_z2u --gpu 0
# python3 cfidln.py --n 5 --r 0.4 --bar 0.08 --dataset adult --p 1
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys, os
import tools
import dataloader
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau
from vae import BaseVAE, UciNet, uVAE, yVAE
from baseline import full_train, full_evaluate, vaz_train, vaz_evaluate, mpz_train, mpz_evaluate
from tabulate import tabulate


parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float, default=0.2, help="The start noise rate")
parser.add_argument('--bar', type=float, default=0.08,
                    help="running experiment at noise rate [args.r, args.r + args.bar]")
parser.add_argument('--n', type=int, default=5, help="the number of runs with random seeds")
parser.add_argument('--d', type=str, default='output', help="description for the output dir")
parser.add_argument('--p', type=int, default=1, help="0 for printing in terminal, 1 for printing in file")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--epochs', type=int, default=20, help="End epoch")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='adult', help="adult, bank")
parser.add_argument('--num_workers', type=int, default=16, help="#Thread for dataloader")
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'])
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--gpu', type=str, default='n')
parser.add_argument('--noise_type', type=str, default='i', help="s or i")

args = parser.parse_args()
args.data_dir = './dataset/'


if args.gpu != 'n':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(os.environ["CUDA_VISIBLE_DEVICES"])


def loaddata(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_data = dataloader.Dataset(args.dataset, True, False, False, dir=args.data_dir, noise_rate=args.noise_rate,
                                    random_seed=args.seed, noise_type=args.noise_type)
    val_data = dataloader.Dataset(args.dataset, False, True, False, dir=args.data_dir, noise_rate=args.noise_rate,
                                    random_seed=args.seed, noise_type=args.noise_type)
    test_data = dataloader.Dataset(args.dataset, False, False, True, dir=args.data_dir, noise_rate=args.noise_rate,
                                    random_seed=args.seed, noise_type=args.noise_type)

    return train_data, val_data, test_data


def get_loader(train_data, val_data, test_data):
    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=False)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=False)

    eval_loader = DataLoader(dataset=test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             drop_last=False)

    return train_loader, val_loader, eval_loader


def evaluate(eval_loader, model, type):
    if torch.cuda.is_available():
        model = model.cuda()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    model.eval()

    for _, (az, z, w, by) in enumerate(eval_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        u, y_pred, az_pred, w_pred, by_pred, mu, logvar = model(az, z, w, by)
        prediction = torch.argmax(y_pred, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    acc = (correct / total).detach().item()
    if type == 'val':
        print('\n---- Validation ----')
        print('[Validation] ACC: ', acc)
    if type == 'test':
        print('\n---- Evaluation ----')
        print('[Test] ACC: ', acc)

    return acc


criterion = nn.CrossEntropyLoss()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return 10 * (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b))


def train(epoch, vae_model, train_loader, optimizer, scheduler):
    if torch.cuda.is_available():
        vae_model = vae_model.cuda()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    print('\n==== Epoch:{0} ===='.format(epoch))
    vae_model.train()

    for _, (az, z, w, by) in enumerate(train_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        u, y_pred, z_pred, w_pred, by_pred, mu, logvar, targets_a, targets_b, lam = vae_model(az, z, w, by,
                                                                                              mix_up=True)
        l_y = mixup_criterion(criterion, y_pred, targets_a, targets_b, lam)

        l_z = F.mse_loss(z, z_pred, reduction="mean")
        l_w = 1 * F.mse_loss(w, w_pred, reduction="mean")
        l_by = 1 * F.cross_entropy(by_pred, by, reduction="mean")
        l_u = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = l_z + l_w + l_by + l_u + l_y

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        prediction = torch.argmax(y_pred, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    loss = loss.detach().item()
    acc = (correct / total).detach().item()

    print('[Train] Loss: ', loss)
    print('[Train] ACC: ', acc)

    return loss, acc


def main(args):
    if args.dataset == 'adult':
        args.a_dim = 1
        args.z_dim = 3
        args.u_dim = 2
        args.y_dim = 2
        args.w_dim = 37
        args.by_dim = 2
        args.full = 41

    if args.dataset == 'bank':
        args.a_dim = 1
        args.z_dim = 28
        args.u_dim = 2
        args.y_dim = 2
        args.w_dim = 5
        args.by_dim = 2
        args.full = 34

    print('\nParameter:', args)

    print('\nLoading data ...')

    train_data, val_data, test_data = loaddata(args)
    train_loader, val_loader, eval_loader = get_loader(train_data, val_data, test_data)

    # vae
    print('\nTraining vae ')
    vae_model = BaseVAE(a_dim=args.a_dim, z_dim=args.z_dim, u_dim=args.u_dim, y_dim=args.y_dim, w_dim=args.w_dim,
                        by_dim=args.by_dim)

    # vae_model = yVAE(a_dim=args.a_dim, z_dim=args.z_dim, u_dim=args.u_dim, y_dim=args.y_dim, w_dim=args.w_dim,
    #                     by_dim=args.by_dim)

    # vae_model = uVAE(a_dim=args.a_dim, z_dim=args.z_dim, u_dim=args.u_dim, y_dim=args.y_dim, w_dim=args.w_dim,
    #                     by_dim=args.by_dim)

    optimizer = torch.optim.SGD(vae_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train_acc_list = []
    val_acc_list = []
    acc_list = []

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train(epoch, vae_model, train_loader, optimizer, scheduler)
        val_acc = evaluate(val_loader, vae_model, 'val')
        accuracy = evaluate(eval_loader, vae_model, 'test')

        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)
        acc_list.append(accuracy)

    index = np.argmax(np.array(val_acc_list))
    max_index = np.argmax(np.array(acc_list))
    print('\nvalidation acc max epoch', index, 'acc:', acc_list[index])
    print('final epoch acc:', acc_list[-1], '\nmax acc', acc_list[max_index])

    # vanilla_z
    print('\nTraining vanilla ')
    vaz_model = UciNet(input_size=args.z_dim)
    vaz_optimizer = torch.optim.SGD(vaz_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    vaz_scheduler = ReduceLROnPlateau(vaz_optimizer, 'min')

    vaz_train_acc_list = []
    vaz_val_acc_list = []
    vaz_acc_list = []

    for epoch in range(args.epochs):
        vaz_train(epoch, vaz_model, train_loader, vaz_optimizer, vaz_scheduler)
        val_acc = vaz_evaluate(val_loader, vaz_model, 'val')
        accuracy = vaz_evaluate(eval_loader, vaz_model, 'test')

        vaz_train_acc_list.append(0)
        vaz_val_acc_list.append(val_acc)
        vaz_acc_list.append(accuracy)

    vaz_index = np.argmax(np.array(vaz_val_acc_list))
    max_index = np.argmax(np.array(vaz_acc_list))
    print('\nvalidation acc max epoch', index, 'acc:', vaz_acc_list[vaz_index])
    print('final epoch acc:', vaz_acc_list[-1], '\nmax acc', vaz_acc_list[max_index])

    # mixup_z
    print('\nTraining mixup ')
    mpz_model = UciNet(input_size=args.z_dim)
    mpz_optimizer = torch.optim.SGD(mpz_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    mpz_scheduler = ReduceLROnPlateau(mpz_optimizer, 'min')

    mpz_train_acc_list = []
    mpz_val_acc_list = []
    mpz_acc_list = []

    for epoch in range(args.epochs):
        mpz_train(epoch, mpz_model, train_loader, mpz_optimizer, mpz_scheduler)
        val_acc = mpz_evaluate(val_loader, mpz_model, 'val')
        accuracy = mpz_evaluate(eval_loader, mpz_model, 'test')

        mpz_train_acc_list.append(tr_acc)
        mpz_val_acc_list.append(val_acc)
        mpz_acc_list.append(accuracy)

    mpz_index = np.argmax(np.array(mpz_val_acc_list))
    max_index = np.argmax(np.array(mpz_acc_list))
    print('\nvalidation acc max epoch', mpz_index, 'acc:', mpz_acc_list[mpz_index])
    print('final epoch acc:', mpz_acc_list[-1], '\nmax acc', mpz_acc_list[max_index])

    return acc_list[index], full_acc_list[full_index], vaz_acc_list[vaz_index], mpz_acc_list[mpz_index]


if __name__ == "__main__":
    # Run from noise rate [args.r, args.r + args.bar] with 0.1 interval
    whole_result = [['FULL'], ['VANILLA'], ['MIXUP'], ['CFIDLN']]
    for m in np.arange(args.r, args.r + args.bar, 0.1):
        args.noise_rate = round(m, 2)

        seed_list = []
        full_seed_list = []
        vaz_seed_list = []
        mpz_seed_list = []

        args.output_dir = 'result/' + args.dataset + '/'
        if not os.path.exists(args.output_dir):
            os.system('mkdir -p %s' % args.output_dir)

        # Print in terminal or file
        if args.p == 0:
            f = open(args.output_dir + args.d + '_' + str(args.noise_type) + '_' + str(args.noise_rate) + '.txt', 'a')
            sys.stdout = f
            sys.stderr = f

        # Run args.n seeds
        for i in range(args.n):
            args.seed = 1 + i
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)

            acc, full_acc, vaz_acc, mpz_acc = main(args)
            seed_list.append(acc)
            full_seed_list.append(full_acc)
            vaz_seed_list.append(vaz_acc)
            mpz_seed_list.append(mpz_acc)

        print('\nacc:', seed_list)
        print('\nacc', 'mean:', np.array(seed_list).mean(), 'std:',
              np.array(seed_list).std(ddof=1))
        result = str(round(np.array(seed_list).mean() * 100, 2)) + '$\pm$' + str(
            round(100 * np.std(np.array(seed_list), ddof=1), 2))
        full_result = str(round(np.array(full_seed_list).mean() * 100, 2)) + '$\pm$' + str(
            round(100 * np.std(np.array(full_seed_list), ddof=1), 2))
        vaz_result = str(round(np.array(vaz_seed_list).mean() * 100, 2)) + '$\pm$' + str(
            round(100 * np.std(np.array(vaz_seed_list), ddof=1), 2))
        mpz_result = str(round(np.array(mpz_seed_list).mean() * 100, 2)) + '$\pm$' + str(
            round(100 * np.std(np.array(mpz_seed_list), ddof=1), 2))

        data = [['FULL', full_result], ['VANILLA', vaz_result], ['MIXUP', mpz_result], ['CFIDLN', result]]
        print(tabulate(data, headers=["Method", str(args.noise_rate)]))

        result_list = [full_result, vaz_result, mpz_result, result]
        for j in range(len(result_list)):
            whole_result[j].append(result_list[j])

    headers = ["Method"]
    for m in np.arange(args.r, args.r + args.bar, 0.1):
        m = round(m, 2)
        headers.append(str(m))
    print(tabulate(whole_result, headers=headers))
