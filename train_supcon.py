from __future__ import print_function
from __future__ import division

import os
import sys
import argparse
import configparser
import torch
import pickle
import random
import string
import numpy as np
import torch.utils.data as tud
import torch.optim as optim
from lm import utils
import data as dataset
from torchvision import transforms
import time
import torch.nn.functional as functional
from torch import nn
from model_supcon import SupCon, SupConLoss

random.seed(222)
np.random.seed(222)
torch.manual_seed(222)
torch.cuda.manual_seed(222)


def train_supcon(model, loader, hypers, optimizer, loss_supcon, loss_ce, log_path, model_path, hyper_path, device, interval):
    
    model.train()
    loss_avg = []
    loss_avg_epoch = []
    time_start = time.perf_counter()
    time_used = 0
    
    for i_batch, (imgs, labels) in enumerate(loader):
        
        optimizer.zero_grad()
        imgs = torch.cat([imgs[0], imgs[1]], dim=0)

        imgs = imgs.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        features_con = model(imgs)
        f1, f2 = torch.split(features_con, [bsz, bsz], dim=0)
        features_con = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        #labels_ce = torch.cat([labels, labels], dim=0)
        
        #l_supcon = loss_supcon(features, labels)
        l = loss_supcon(features_con, labels)
        #l_ce = loss_ce(features_ce, labels_ce)
        #l = l_ce + l_con
        l.backward()
        
        optimizer.step()

        loss_avg.append(l.item())
        loss_avg_epoch.append(l.item())

        hypers['step'] += 1
        if hypers['step'] % interval == 0:
        #if hypers['step'] % 10 == 0:
            time_stamp = time.perf_counter()
            time_used += time_stamp - time_start
            
            l = sum(loss_avg)/len(loss_avg)
            pcont = "Step %d, time: %.3f, train loss: %.3f" % (hypers['step'], time_used/60, l)
            time_start = time.perf_counter()
            print(pcont)
            
            with open(log_path, 'a+') as fo:
                fo.write(pcont+"\n")
            with open(model_path, 'wb') as fo:
                torch.save(model.state_dict(), fo)
            with open(hyper_path, 'wb') as fo:
                pickle.dump(hypers, fo)
            loss_avg = []

    loss_epoch = sum(loss_avg_epoch)/len(loss_avg_epoch)
    return loss_epoch

def evaluate(model, loader, loss_supcon, loss_ce, device):
    model.eval()
    larr = []
    
    for i_batch, (imgs, labels) in enumerate(loader):
        imgs = torch.cat([imgs[0], imgs[1]], dim=0)

        imgs = imgs.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]

        with torch.no_grad():
            features_con = model(imgs)

            f1, f2 = torch.split(features_con, [bsz, bsz], dim=0)
            features_con = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            #labels_ce = torch.cat([labels, labels], dim=0)
            
            #l_supcon = loss_supcon(features, labels)
            l = loss_supcon(features_con, labels)
            #l_ce = loss_ce(features_ce, labels_ce)
            #l = l_ce + l_con
        larr.append(l)

    l = sum(larr)/len(larr)

    return l

def main():
    parser = argparse.ArgumentParser(description="Attn Encoder")
    parser.add_argument("--img", type=str, help="image dir")
    parser.add_argument("--csv", type=str, help="csv dir")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--output", type=str, help="output dir")
    parser.add_argument("--cont", action="store_true", help="continue training")
    parser.add_argument("--epoch", type=int, default=20, help="epoch")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument('--optim_step_size',nargs='+', default=[40, 20, 10], help='<Required> Set flag')
    parser.add_argument("--optim_gamma", type=float, default=0.1, help="lr decay rate")
    
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    best_path = os.path.join(args.output, "supcon_best.pth")
    latest_path = os.path.join(args.output, "supcon_latest.pth")
    latest_epoch_path = os.path.join(args.output, "supcon_latest_epoch.pth")
    log = os.path.join(args.output, "supcon_log")
    hyper_path = os.path.join(args.output, "supcon_hyper.pth")

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg, img_cfg = config['MODEL'], config['LANG'], config['IMAGE']

    char_list = 'acbedgfihkjmlonqpsrutwvyxz'
    save_interval = model_cfg.getint('interval')
    
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epoch
    optim_step_size, optim_gamma = [int(s) for s in args.optim_step_size] , args.optim_gamma

    device, cpu = torch.device('cuda'), torch.device('cpu')
    
    vocab_map, inv_vocab_map, char_list = utils.get_ctc_vocab(char_list, add_blank=False)
    print(vocab_map)

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=(224,224), scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize])
    
    dev_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    
    train_csv, dev_csv = os.path.join(args.output, 'frame-label_train.csv'), os.path.join(args.output, 'frame-label_dev.csv')
    sld_train_data = dataset.SLData_frame(args.img, train_csv, vocab_map, transform=dataset.TwoCropTransform(train_transform))
    sld_dev_data = dataset.SLData_frame(args.img, dev_csv, vocab_map, transform=dataset.TwoCropTransform(dev_transform)) # dataset.Rescale([1], [(13, 13)])

    train_loader = tud.DataLoader(sld_train_data, batch_size=batch_size, shuffle=True)
    dev_loader = tud.DataLoader(sld_dev_data, batch_size=batch_size, shuffle=True)
    print('number of batch : ', len(train_loader))

    model = SupCon(hidden=1024, num_class=len(char_list))
    model.to(device)
    
    hypers = {'step': 0, 'epoch': 0, 'best_dev_loss': 10e6, 'perm': np.random.permutation(len(sld_train_data)).tolist()}
    if args.cont:
        print("Load %s, %s" % (latest_path, hyper_path))
        model.load_state_dict(torch.load(latest_path))
        try:
            with open(hyper_path, 'rb') as fo:
                hypers = pickle.load(fo)
        except Exception as err:
            print("Error loading %s: %s" % (hyper_path, err))
            hypers = {'step': 0, 'epoch': 0, 'best_dev_acc': -1, 'perm': np.random.permutation(len(sld_train_data)).tolist()}

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=optim_step_size, gamma=optim_gamma)
    loss_supcon = SupConLoss()
    loss_ce = nn.CrossEntropyLoss(reduction = 'mean')

    print('%d training epochs' % (epochs))
    for ep in range(epochs):
        if ep < hypers['epoch']:
            continue
        for p in optimizer.param_groups:
            print('Model', p['lr'])
            
        train_loss = train_supcon(model, train_loader, hypers, optimizer, loss_supcon, loss_ce, log, latest_path, hyper_path, device, save_interval)
        dl = evaluate(model, dev_loader, loss_supcon, loss_ce, device)
        scheduler.step()
        
        #pcont = 'Epoch %d, dev loss: %.3f, dev acc (LEV): %.3f' % (ep, dl, dacc)
        pcont = 'Epoch %d, train loss: %.3f, dev loss: %.3f' % (ep, train_loss, dl)
        print(pcont)
        with open(log, 'a+') as fo:
            fo.write(pcont+"\n")
            
        # save model and hyperparameter setting
        with open(latest_epoch_path, 'wb') as fo:
            torch.save(model.state_dict(), fo)
        
        hypers['epoch'] = ep
        if hypers['best_dev_loss'] > dl:
            hypers['best_dev_loss'] = dl
            with open(best_path, 'wb') as fo:
                torch.save(model.state_dict(), fo)

        with open(hyper_path, 'wb') as fo:
            pickle.dump(hypers, fo)
    return

if __name__ == '__main__':
    main()
