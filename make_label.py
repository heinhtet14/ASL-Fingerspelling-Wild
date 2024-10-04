from __future__ import print_function
from __future__ import division

import os
import numpy as np
import csv
import sys
import lev
import argparse
import torch
import configparser
import scipy.io as sio
import torch.utils.data as tud
import data as dataset
from lm import utils
from model import Model
from torchvision import transforms
from ctc_decoder import Decoder
from lm.lm_scorer import Scorer as Scorer

def make_label(model, loader, decoder, device, inv_vocab_map):
    
    model.eval()
    label_arr = []
    csv_list = []
    for i_batch, sample in enumerate(loader):

        imgs, labels, prob_sizes, label_sizes = sample['image'], sample['label'], sample['prob_size'], sample['label_size']
        imgs_path = sample['image_path']

        attn_labels = sample['attn_label']
        target_length, batch_size = list(attn_labels.size())
        
        attn_labels = attn_labels.to(device)

        with torch.no_grad():
            l, probs, _, _, _, _ = model(imgs, labels, prob_sizes, label_sizes, attn_labels, target_length)

        probs = probs.cpu().data.numpy()
        #attn_probs = attn_probs.transpose(0, 1).cpu().data.numpy()
        #attn_pred = attn_pred.cpu().tolist()
        for j in range(len(probs)):
            label_arr.append(labels[j][0:label_sizes[j]].tolist())
            prob = probs[j]
            indexes = np.argmax(prob, axis=1).tolist()
            greedy_pred = decoder.greedy_decode(prob, digit=True)
            #beam_pred = decoder.beam_decode(prob, beam_size=beam_size, beta=lm_beta, gamma=ins_gamma, scorer=scorer, digit=True)

            acc = lev.compute_acc([greedy_pred], [labels[j][0:label_sizes[j]].tolist()])

            if i_batch% 100 == 0 :
                print('\n{}/{}'.format(i_batch, len(loader)))
                print('label : ', labels[j][0:label_sizes[j]].tolist())
                print('greedy : ', greedy_pred)
            
            if acc == 100 :
                for idx_frame in range(len(indexes)) :
                    csv_list.append((imgs_path[j][idx_frame],inv_vocab_map[indexes[idx_frame]]))

    return csv_list

def main():
    parser = argparse.ArgumentParser(description="Attn Encoder")
    parser.add_argument("--img", type=str, help="image dir")
    parser.add_argument("--csv", type=str, help="csv dir")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--model_pth", type=str, help="encoder path")
    parser.add_argument("--output_path", type=str, help="output path")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.conf)
    model_cfg, lang_cfg, img_cfg = config['MODEL'], config['LANG'], config['IMAGE']
    hidden_size, attn_size, n_layers = model_cfg.getint('hidden_size'), model_cfg.getint('attn_size'), model_cfg.getint('n_layers')
    prior_gamma = model_cfg.getfloat('prior_gamma')
    batch_size = 1
    char_list = lang_cfg['chars']
    immean, imstd = [float(x) for x in config['IMAGE']['immean'].split(',')], [float(x) for x in config['IMAGE']['imstd'].split(',')]
    train_csv, dev_csv, test_csv = os.path.join(args.csv, 'train.csv'), os.path.join(args.csv, 'dev.csv'), os.path.join(args.csv, 'test.csv')

    device, cpu = torch.device('cuda'), torch.device('cpu')

    _, _, attn_char_list = utils.get_vocab(char_list)
    attn_vocab_map, attn_inv_vocab_map, attn_char_list = utils.get_ctc_vocab(attn_char_list)
    ctc_vocab_map, ctc_inv_vocab_map, ctc_char_list = attn_vocab_map, attn_inv_vocab_map, attn_char_list

    decoder = Decoder(ctc_char_list, blank_index=0)

    print(ctc_vocab_map)
    print(attn_vocab_map)

    model = Model(hidden_size=hidden_size, output_size=len(ctc_char_list), vocab_map=ctc_vocab_map, attn_model='dot')
    model.to(device)

    print('Load model: %s' % (args.model_pth))
    model.load_state_dict(torch.load(args.model_pth))

    scale_range = [0]
    hw_range = [(0, 0)] 
    tsfm = transforms.Compose([dataset.ToTensor(device), dataset.Rescale(scale_range, hw_range, origin_scale=True), dataset.Normalize(immean, imstd, device)])

    train_data = dataset.SLData(args.img, train_csv, ctc_vocab_map, attn_vocab_map, transform=tsfm, upper_len=float('inf'))
    dev_data = dataset.SLData(args.img, dev_csv, ctc_vocab_map, attn_vocab_map, transform=tsfm, upper_len=float('inf'))
    test_data = dataset.SLData(args.img, test_csv, ctc_vocab_map, attn_vocab_map, transform=tsfm, upper_len=float('inf'))

    train_loader = tud.DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)
    dev_loader = tud.DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)
    test_loader = tud.DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn_ctc)

    for loader_type in ['train', 'dev'] :
        
        print(loader_type)
        
        if loader_type == 'train' :
            loader = train_loader
        elif loader_type == 'dev' :
            loader = dev_loader
        elif loader_type == 'test' :
            loader = test_loader
        
        csv_list = make_label(model, loader, decoder, device, ctc_inv_vocab_map)
        
        with open(os.path.join(args.output_path, 'frame-label_{}.csv'.format(loader_type)),'w') as out:
            csv_out=csv.writer(out)
            for (path, label) in csv_list:
                csv_out.writerow((path, label))

    return


if __name__ == "__main__":
    main()
