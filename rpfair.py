# python3 forward.py --n 5 --r 0.1 --bar 0.38 --dataset adult --epochs 20 --p 0 --gpu 1 --d 0.02reg_
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
from vae import BaseVAE, UciNet
from baseline import full_train, full_evaluate, vaz_train, vaz_evaluate, mpz_train, mpz_evaluate
from tabulate import tabulate
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--a', type=float, default=0.02, help="The fair constraint weight")
parser.add_argument('--r', type=float, default=0.2, help="The start noise rate")
parser.add_argument('--bar', type=float, default=0.08,
                    help="running experiment at noise rate [args.r, args.r + args.bar]")
parser.add_argument('--n', type=int, default=5, help="the number of runs with random seeds")
parser.add_argument('--d', type=str, default='output', help="description for the output dir")
parser.add_argument('--p', type=int, default=1, help="1 for printing in terminal, 0 for printing in file")
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
args.data_dir = '../CFIDLN/dataset/'


if args.gpu != 'n':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(os.environ["CUDA_VISIBLE_DEVICES"])


def loaddata(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_data = dataloader.st_Dataset(args.dataset, True, False, False, dir=args.data_dir, noise_rate=args.noise_rate,
                                    random_seed=args.seed, noise_type=args.noise_type)
    val_data = dataloader.st_Dataset(args.dataset, False, True, False, dir=args.data_dir, noise_rate=args.noise_rate,
                                    random_seed=args.seed, noise_type=args.noise_type)
    test_data = dataloader.st_Dataset(args.dataset, False, False, True, dir=args.data_dir, noise_rate=args.noise_rate,
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
    cm = np.zeros((2, 2))

    model.eval()

    for _, (a, z, w, by) in enumerate(eval_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                a = a.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        # u, y_pred, az_pred, w_pred, by_pred, mu, logvar = model(az, z, w, by)
        x = torch.cat((z, w), dim=1)
        y_pred = model(x)

        prediction = torch.argmax(y_pred, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

        tem_cm = confusion_matrix(a.cpu(), prediction.detach().cpu())

        cm += tem_cm

    acc = (correct / total).detach().item()
    p11 = cm[1][1] / a.sum()
    p10 = cm[0][1] / a.sum()
    if p11 == 0 or p10 == 0:
        p_rule = 0
    else:
        p_rule = (p11 / p10).detach().item()
        p_rule = min(p_rule, 1 / p_rule)
    if type == 'val':
        print('\n---- Validation ----')
        print('[Validation] ACC: ', acc)
    if type == 'test':
        print('\n---- Evaluation ----')
        print('[Test] ACC: ', acc)
        print('[Test] pFair: ', p_rule)

    return acc, p_rule


criterion = nn.CrossEntropyLoss()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return 10 * (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b))

def rptrain(epoch, model, train_loader, optimizer, scheduler):
    if torch.cuda.is_available():
        model = model.cuda()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    print('\n==== Epoch:{0} ===='.format(epoch))
    model.train()
    nnloss = nn.NLLLoss()

    for _, (a, z, w, by, t) in enumerate(train_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                a = a.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        ins_t = []
        t = t.numpy()
        clean_y = np.argmax(t, 1)
        for i in range(len(by)):
            if clean_y[i] > 0:
                tem_t = np.array([[1.0, 0.], t[i]])
            else:
                tem_t = np.array([t[i], [0., 1.0]])
            ins_t.append(tem_t)
        ins_t = torch.FloatTensor(ins_t).cuda()

        x = torch.cat((z, w), dim=1)
        out = model(x)
        m = nn.Softmax(dim=1)
        prob = m(out)
        n_prob = torch.empty(prob.shape).cuda()
        for i in range(prob.shape[0]):
            n_prob[i] = prob[i].reshape(1, -1).mm(ins_t[i]).squeeze()

        logprob = n_prob.log()
        loss_ce = nnloss(logprob, by)

        d_theta = prob[:, 1]   # P(Y=1|x)
        d_theta = d_theta - 0.5

        gap = a - a.mean()

        reg = args.a * torch.abs(torch.dot(gap, d_theta))

        loss = loss_ce + reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        prediction = torch.argmax(out, 1)
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


    # Training R-p-Fair
    print('\nTraining R-p-Fair ')
    model = UciNet(input_size=args.full - 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train_acc_list = []
    val_acc_list = []
    acc_list = []
    fair_list = []

    for epoch in range(args.epochs):
        tr_loss, tr_acc = rptrain(epoch, model, train_loader, optimizer, scheduler)
        val_acc, _ = evaluate(val_loader, model, 'val')
        accuracy, fair = evaluate(eval_loader, model, 'test')

        train_acc_list.append(tr_acc)
        val_acc_list.append(val_acc)
        acc_list.append(accuracy)
        fair_list.append(fair)

    index = np.argmax(np.array(val_acc_list))
    max_index = np.argmax(np.array(acc_list))
    print('\nvalidation acc max epoch', index, 'acc:', acc_list[index], 'fair:', fair_list[index])
    print('final epoch acc:', acc_list[-1], '\nmax acc', acc_list[max_index])
    rpacc = acc_list[index]
    rpfair = fair_list[index]

    return rpacc, rpfair, 0, 0


if __name__ == "__main__":
    # Run from noise rate [args.r, args.r + args.bar] with 0.1 interval
    whole_result = [['R-p-Fair acc'], ['R-p-Fair fairness']]
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
            args.seed = 41 + i
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

        data = [['R-p-Fair acc', result], ['R-p-Fair fair', full_result]]
        print(tabulate(data, headers=["Method", str(args.noise_rate)]))

        result_list = [result, full_result]
        for j in range(len(result_list)):
            whole_result[j].append(result_list[j])

    headers = ["Method"]
    for m in np.arange(args.r, args.r + args.bar, 0.1):
        m = round(m, 2)
        headers.append(str(m))
    print(tabulate(whole_result, headers=headers))
