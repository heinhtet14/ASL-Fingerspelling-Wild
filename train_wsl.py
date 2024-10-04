from __future__ import print_function
from __future__ import division
from lib2to3.pgen2 import grammar

import os
import sys
import argparse
import configparser
import torch
import pickle
import lev
import random
import string
import numpy as np
import torch.utils.data as tud
import torch.optim as optim
from ctc_decoder import Decoder
from lm import utils
from torch import nn
#from warpctc_pytorch import CTCLoss
from model import Model
import data as dataset
from torchvision import transforms
import time
import torch.nn.functional as F

random.seed(222)
np.random.seed(222)
torch.manual_seed(222)
torch.cuda.manual_seed(222)


def train(model, train_loader, hypers, optimizer, decoder, log_path, output, hyper_path, device, interval, vocab_map):
    model.train()
    larr, pred_arr, label_arr = [], [], []
    attn_pred_arr, attn_label_arr= [], []
    time_start = time.perf_counter()
    time_used = 0
    
    for i_batch, sample in enumerate(train_loader):
               
        optimizer.zero_grad()
        imgs, labels, prob_sizes, label_sizes = sample['image'], sample['label'], sample['prob_size'], sample['label_size']
        attn_labels = sample['attn_label']
        batch_size, target_length = list(attn_labels.size())

        attn_labels = attn_labels.to(device)
        loss, probs, attn_pred, _, _, _ = model(imgs, labels, prob_sizes, label_sizes, attn_labels, target_length)
        
        loss = torch.sum(loss)
        loss.backward()
        optimizer.step()

        larr.append(loss.item())

        # Decode
        probs = probs.cpu().data.numpy()
        for j in range(len(probs)):
            pred = decoder.greedy_decode(probs[j], digit=True)
            pred_arr.append(pred)
            label = labels[j][0:label_sizes[j]].tolist()
            label_arr.append(label)

        #---
        for j in range(len(attn_pred)):
            attn_pred_j = attn_pred[j].cpu().tolist()
            attn_pred_j = [ p for p in attn_pred_j if p != vocab_map['EOS'] and p != vocab_map['PAD']]
            attn_pred_arr.append(attn_pred_j)

        #attn_labels = attn_labels.transpose(0, 1)
        for j in range(len(attn_labels)):
            attn_label_j = attn_labels[j].cpu().tolist()
            attn_label_j = [ l for l in attn_label_j if l != vocab_map['EOS'] and l != vocab_map['PAD']]
            attn_label_arr.append(attn_label_j)
        #---   

        hypers['step'] += 1
        if hypers['step'] % interval == 0:
            
            time_stamp = time.perf_counter()
            time_used += time_stamp - time_start
            
            ctc_acc = lev.compute_acc(pred_arr, label_arr)
            attn_acc = lev.compute_acc(attn_pred_arr, attn_label_arr)

            l = sum(larr)/len(larr)
            pcont = "Step %d, time: %.3f, train loss: %.3f, ctc-acc (LEV): %.3f, attn-acc (LEV): %.3f" % (hypers['step'], time_used/60, l, ctc_acc, attn_acc)
            time_start = time.perf_counter()
            print(pcont)
            
            with open(log_path, 'a+') as fo:
                fo.write(pcont+"\n")

            with open(hyper_path, 'wb') as fo:
                pickle.dump(hypers, fo)

            larr, pred_arr, label_arr = [], [], []
            attn_pred_arr, attn_label_arr= [], []

    with open(os.path.join(output, "last-ep.pth"), 'wb') as fo:
        torch.save(model.state_dict(), fo)

    return

def evaluate(model, loader, decoder, device, vocab_map):
    model.eval()
    #hidden_size, n_layers = encoder.encoder_cell.hidden_size, encoder.encoder_cell.n_layers
    larr, pred_arr, label_arr = [], [], []
    attn_pred_arr, join_pred_arr, attn_label_arr= [], [], []
    for i_batch, sample in enumerate(loader):
        imgs, labels, prob_sizes, label_sizes = sample['image'], sample['label'], sample['prob_size'], sample['label_size']
        attn_labels = sample['attn_label']
        batch_size, target_length = list(attn_labels.size())

        attn_labels = attn_labels.to(device)
        with torch.no_grad():
            loss, probs, attn_pred, _, _, _ = model(imgs, labels, prob_sizes, label_sizes, attn_labels, target_length)
        
        larr.append(loss.item())
        
        # Decode
        probs = probs.cpu().data.numpy()
        for j in range(len(probs)):
            pred = decoder.greedy_decode(probs[j], digit=True)
            pred_arr.append(pred)
            label = labels[j][0:label_sizes[j]].tolist()
            label_arr.append(label)

        #---
        for j in range(len(attn_pred)):
            attn_pred_j = attn_pred[j].cpu().tolist()
            attn_pred_j = [ p for p in attn_pred_j if p != vocab_map['EOS'] and p != vocab_map['PAD']]
            attn_pred_arr.append(attn_pred_j)

        for j in range(len(attn_labels)):
            attn_label_j = attn_labels[j].cpu().tolist()
            attn_label_j = [ l for l in attn_label_j if l != vocab_map['EOS'] and l != vocab_map['PAD']]
            attn_label_arr.append(attn_label_j)
        #---   
            
    ctc_acc = lev.compute_acc(pred_arr, label_arr)
    attn_acc = lev.compute_acc(attn_pred_arr, attn_label_arr)

    l = sum(larr)/len(larr)
    return l, ctc_acc, attn_acc

def main():
    parser = argparse.ArgumentParser(description="Attn Encoder")
    parser.add_argument("--img", type=str, help="image dir")
    #parser.add_argument("--prior", type=str, help="prior dir")
    parser.add_argument("--csv", type=str, help="csv dir")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--output", type=str, help="output dir")
    parser.add_argument("--pretrain", type=str, default=None, help="pretrain path")
    parser.add_argument("--cont", action="store_true", help="continue training")
    parser.add_argument("--epoch", type=int, default=1, help="epoch")
    #parser.add_argument("--optim_step_size", type=int, default=30, help="lr decay step size")
    parser.add_argument('--optim_step_size',nargs='+', default=[40, 20, 10], help='<Required> Set flag')
    parser.add_argument("--optim_gamma", type=float, default=0.1, help="lr decay rate")
    parser.add_argument("--batch_size", type=int, default=40, help="batch_size")
    parser.add_argument("--lr", type=float, default=0.001, help="lr")
    parser.add_argument("--scaling", action="store_true", help="data augmentation (scaling)")
    parser.add_argument("--img_scale", type=float, default=1., nargs="+", help="image scales")
    parser.add_argument("--map_scale", type=int, default=13, nargs="+", help="map scales")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    best_path = os.path.join(args.output, "best.pth")
    latest_path = os.path.join(args.output, "latest.pth")
    log = os.path.join(args.output, "log")
    hyper_path = os.path.join(args.output, "hyper.pth")

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg, img_cfg = config['MODEL'], config['LANG'], config['IMAGE']
    hidden_size, attn_size, n_layers = model_cfg.getint('hidden_size'), model_cfg.getint('attn_size'), model_cfg.getint('n_layers')
    #prior_gamma = model_cfg.getfloat('prior_gamma')
    learning_rate = args.lr
    batch_size = args.batch_size
    char_list = lang_cfg['chars'] # " '&.@acbedgfihkjmlonqpsrutwvyxz"
    immean, imstd = [float(x) for x in config['IMAGE']['immean'].split(',')], [float(x) for x in config['IMAGE']['imstd'].split(',')] # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    upper_len = model_cfg.getint('upper_length')
    clip = model_cfg.getfloat('clip')
    save_interval = model_cfg.getint('interval')
    epochs = args.epoch
    optim_step_size, optim_gamma = [int(s) for s in args.optim_step_size] , args.optim_gamma

    train_csv, dev_csv = os.path.join(args.csv, 'train.csv'), os.path.join(args.csv, 'test.csv')

    device, cpu = torch.device('cuda'), torch.device('cpu')
    #ctc_vocab_map, ctc_inv_vocab_map, ctc_char_list = utils.get_ctc_vocab(char_list)
    _, _, attn_char_list = utils.get_vocab(char_list)
    attn_vocab_map, attn_inv_vocab_map, attn_char_list = utils.get_ctc_vocab(attn_char_list)
    ctc_vocab_map, ctc_inv_vocab_map, ctc_char_list = attn_vocab_map, attn_inv_vocab_map, attn_char_list
    
    print(ctc_inv_vocab_map)
    print(attn_vocab_map)

    if type(args.img_scale) == list and type(args.map_scale) == list:
        scale_range, hw_range = args.img_scale, [(x, x) for x in args.map_scale]
    elif type(args.img_scale) == float and type(args.map_scale) == int:
        scale_range, hw_range = [args.img_scale], [(args.map_scale, args.map_scale)]
    else:
        raise AttributeError('scale: list or float/int')

    if not args.scaling:
        tsfm_train = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])
        tsfm_test = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])
    else:

        tsfm_train = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range), dataset.Normalize(immean, imstd, device)])
        tsfm_test = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])

    sld_train_data = dataset.SLData(args.img, train_csv, ctc_vocab_map, attn_vocab_map, transform=tsfm_train, upper_len=upper_len)
    sld_dev_data = dataset.SLData(args.img, dev_csv, ctc_vocab_map, attn_vocab_map, transform=tsfm_test, upper_len=float('inf')) # dataset.Rescale([1], [(13, 13)])

    model = Model(hidden_size=hidden_size, output_size=len(ctc_char_list), vocab_map=ctc_vocab_map, attn_model='dot')
    model.to(device)


    siamese_weights = torch.load('./results/model_3/supcon_latest_epoch.pth') #load weight after finish the first round training wsl.
    for name, param in siamese_weights.items():
        name_split = name.split('.')
        if name_split[0] == 'conv' :
            model.state_dict()['encoder.{}'.format(name)].copy_(param)
            print(name, 'encoder.{}'.format(name))
    

    '''
    if torch.cuda.device_count() > 1:
        print('Using %d GPUs' % (torch.cuda.device_count()))
        model = nn.DataParallel(model)
    '''
    
    params = list(model.parameters())
    optimizer = optim.AdamW(params, lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=optim_step_size, gamma=optim_gamma)

    decoder = Decoder(ctc_char_list)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    hypers = {'step': 0, 'epoch': 0, 'best_dev_acc': -1, 'perm': np.random.permutation(len(sld_train_data)).tolist()}

    if args.cont:
        print("Load %s, %s" % (latest_path, hyper_path))
        model.load_state_dict(torch.load(best_path))
        try:
            with open(hyper_path, 'rb') as fo:
                hypers = pickle.load(fo)
        except Exception as err:
            print("Error loading %s: %s" % (hyper_path, err))
            hypers = {'step': 0, 'epoch': 0, 'best_dev_acc': -1, 'perm': np.random.permutation(len(sld_train_data)).tolist()}

    train_loader = tud.DataLoader(sld_train_data, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn_ctc)
    dev_loader = tud.DataLoader(sld_dev_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)

    print('Optimizer, decay {} after {} epochs'.format(optim_gamma, optim_step_size))
    print('%d training epochs' % (epochs))
    for ep in range(epochs):
        if ep < hypers['epoch']:
            continue
        for p in optimizer.param_groups:
            print(ep, 'Model', p['lr'])

        train(model, train_loader, hypers, optimizer, decoder, log, args.output, hyper_path, device, save_interval, ctc_vocab_map)
        dl, ctc_dacc, attn_dacc = evaluate(model, dev_loader, decoder, device, ctc_vocab_map)
        scheduler.step()
        
        pcont = 'Epoch %d, dev loss: %.3f, dev ctc-acc (LEV): %.3f, attn-acc (LEV): %.3f' % (ep, dl, ctc_dacc, attn_dacc)
        print(pcont)

        with open(log, 'a+') as fo:
            fo.write(pcont+"\n")
        # save model and hyperparameter setting
        hypers['epoch'] = ep
        if hypers['best_dev_acc'] < ctc_dacc:
            hypers['best_dev_acc'] = ctc_dacc
            
            with open(os.path.join(args.output, "best.pth"), 'wb') as fo:
                torch.save(model.state_dict(), fo)

        with open(hyper_path, 'wb') as fo:
            pickle.dump(hypers, fo)
        
    return

if __name__ == '__main__':
    main()
    