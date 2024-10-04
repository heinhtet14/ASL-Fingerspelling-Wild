import os
import torch
import scipy.io as sio
import skimage.io as skio
import numpy as np
import random
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as tud
from torch.nn import functional as F

class SLData(tud.Dataset):
    """
    Sign Language Dataset
    """
    def __init__(self, img_dir, fcsv, ctc_vocab_map, attn_vocab_map, transform=None, upper_len=200, upper_sample=2):
        # upper_len: frame sub-sample if length larger
        self.img_dir = img_dir
        #self.prior_dir = prior_dir
        self.fcsv = fcsv
        self.ctc_vocab_map = ctc_vocab_map
        self.attn_vocab_map = attn_vocab_map
        self.transform = transform
        self.upper_len = upper_len
        self.upper_sample = upper_sample
        self._parse()

    def _parse(self):
        with open(self.fcsv, "r") as fo:
            lns = fo.readlines()
        # sub-sampling
        self.imdirs, self.labels, self.num_frames = [], [], []
        for i in range(len(lns)):
            imdir, label, nframe = lns[i].strip().split(",")
            
            if int(nframe) > self.upper_len :
                continue

            self.imdirs.append(imdir)
            self.labels.append(label)
            self.num_frames.append(int(nframe))
            
        print("%d data" % len(self.num_frames))

    def __len__(self):
        return len(self.imdirs)

    def _int2str(self, i):
        return "0"*(4-len(str(i))) + str(i)

    def __getitem__(self, idx):
        attn_label = list(map(lambda x: self.attn_vocab_map[x], self.labels[idx])) + [self.attn_vocab_map['EOS']]
        label = list(map(lambda x: self.ctc_vocab_map[x], self.labels[idx]))

        #fnames = [self._int2str(i)+".jpg" for i in range(1, self.num_frames[idx]+1)]
        fnames = [ p for p in sorted(os.listdir(os.path.join(self.img_dir, self.imdirs[idx])))\
                  if p.endswith(('.png', '.jpg'))] #filters only images
        
        imgs, priors, imgs_path = [], [], []
        for fname in fnames:
            im_fullname = os.path.join(self.img_dir, self.imdirs[idx], fname)
            #print(im_fullname)
            img = skio.imread(im_fullname).astype(np.float32)
            
            imgs_path.append(os.path.join(self.imdirs[idx], fname))
            imgs.append(img)
        imgs = np.stack(imgs)
        #prior_fname = os.path.join(self.prior_dir, self.imdirs[idx], 'prior.npy')
        #prior = np.load(prior_fname)
        if len(imgs) > self.upper_len:
            imgs = imgs[::self.upper_sample]
            imgs_path = imgs_path[::self.upper_sample]

        sample = {'image': imgs,
                  'label': label,
                  'attn_label': attn_label,
                  'prob_size': [len(imgs)],
                  'label_size': [len(label)],
                  'imdir': self.imdirs[idx],
                  'image_path': imgs_path}
        
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
class SLData_siamese(tud.Dataset):
    """
    Sign Language Dataset
    """
    def __init__(self, img_dir, fcsv, vocab_map):
        
        self.char_list = list(vocab_map.keys())
        self.img_dir = img_dir
        self.fcsv = fcsv
        self.vocab_map = vocab_map
        self._parse()

    def _parse(self):
        with open(self.fcsv, "r") as fo:
            lns = fo.readlines()
        # sub-sampling
        self.paths, self.labels, self.num_frames = [], [], []
        for i in range(len(lns)):
            path, label = lns[i].strip().split(",")
            if label in self.char_list:
                
                path = os.path.join(self.img_dir , path)

                self.paths.append(path)
                self.labels.append(label)

        self.labels_group = {}
        for i in range(0,len(self.labels)) :
            key = self.vocab_map[self.labels[i]]

            if str(key) not in list(self.labels_group.keys()) :
                self.labels_group[str(key)] = []
                self.labels_group[str(key)] += [i]
            else :
                self.labels_group[str(key)] += [i]
            
    def __len__(self):
        return len(self.paths)

    def _int2str(self, i):
        return "0"*(4-len(str(i))) + str(i)

    def __getitem__(self, idx):
        label1 = self.vocab_map[self.labels[idx]] 
        img1 = skio.imread(self.paths[idx])
        same_class = random.randint(0,1)
        if same_class:
            label = 1
            while True:
                #keep looping till the same class image is found
                idx2 = random.choice(self.labels_group[str(label1)])
                label2 = self.vocab_map[self.labels[idx2]]
                path2 = self.paths[idx2]
                if label1 == label2:
                    break
        else:
            label = -1
            while True:
                #keep looping till a different class image is found
                rand_key = random.choice(list(self.labels_group.keys()))
                idx2 = random.choice(self.labels_group[str(rand_key)])
                label2 = self.vocab_map[self.labels[idx2]]

                path2 = self.paths[idx2]
                if label1 != label2:
                    break

        img2  = skio.imread(path2)

        img1 = img1 / 255.0
        img2 = img2 / 255.0

        target = label
        img1_label = label1

        return img1, img2, target, img1_label

class SLData_frame(tud.Dataset):
    """
    Sign Language Dataset
    """
    def __init__(self, img_dir, fcsv, vocab_map, transform=None):
        
        self.char_list = list(vocab_map.keys())
        self.img_dir = img_dir
        self.fcsv = fcsv
        self.vocab_map = vocab_map
        self.transform = transform
        self._parse()

    def _parse(self):
        with open(self.fcsv, "r") as fo:
            lns = fo.readlines()
        # sub-sampling
        self.paths, self.labels, self.num_frames = [], [], []
        for i in range(len(lns)):
            path, label = lns[i].strip().split(",")
            if label in self.char_list:
                
                path = os.path.join(self.img_dir , path)

                self.paths.append(path)
                self.labels.append(label)
            
    def __len__(self):
        print('amount of data : ', len(self.paths))
        return len(self.paths)

    def _int2str(self, i):
        return "0"*(4-len(str(i))) + str(i)

    def __getitem__(self, idx):
        label = self.vocab_map[self.labels[idx]]
        img = Image.open(self.paths[idx])
    
        if self.transform is not None:
            img = self.transform(img)
            
            #img[0].save('./visualize/img_{}-1.jpg'.format(idx))
            #img[1].save('./visualize/img_{}-2.jpg'.format(idx))

        return img, label

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class Rescale(object):
    """Scale Image/Prior tensor (data augmentation)"""
    def __init__(self, scales, hws, origin_scale=False):
        self.scales = sorted(scales)
        self.hs = sorted([hw[0] for hw in hws])
        self.ws = sorted([hw[1] for hw in hws])
        self.origin = origin_scale

    def __call__(self, sample):
        if self.origin:
            sample['image'] = sample['image'].unsqueeze(dim=0)
            #sample['prior'] = sample['prior'].unsqueeze(dim=0)
            return sample

        H, W = sample['image'].size(2), sample['image'].size(3)
        Hmax, Wmax = int(H*self.scales[-1]), int(W*self.scales[-1])
        hmax, wmax = self.hs[-1], self.ws[-1]
        images, priors = [], []
        with torch.no_grad():
            for i, scaling in enumerate(self.scales):
                X = F.upsample(sample['image'], size=(int(H*scaling), int(W*scaling)), mode='bilinear')
                #M = F.upsample(sample['prior'].unsqueeze(dim=1), size=(self.hs[i], self.ws[i]), mode='bilinear')
                Xmax = X.new_zeros((len(sample['image']), 3, Hmax, Wmax))
                #Mmax = M.new_zeros((len(sample['prior']), 1, hmax, wmax))
                x0, y0, x1, y1 = max((Wmax - X.size(-1))//2, 0), max((Hmax - X.size(-2))//2, 0), min((Wmax + X.size(-1))//2, Wmax), min((Hmax + X.size(-2))//2, Hmax)
                Xmax[:, :, y0: y1, x0: x1] = X
                #x0, y0, x1, y1 = max((wmax - M.size(-1))//2, 0), max((hmax - M.size(-2))//2, 0), min((wmax + M.size(-1))//2, wmax), min((hmax + M.size(-2))//2, hmax)
                #Mmax[:, :, y0: y1, x0: x1] = M
                images.append(Xmax)
                #priors.append(Mmax)
        sample['image'] = torch.stack(images)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        image, label, prob_size, label_size = sample['image'], sample['label'], sample['prob_size'], sample['label_size']
        image_path = sample['image_path']
        attn_label = sample['attn_label']
        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C X H X W
        image, label = torch.from_numpy(image).to(self.device), torch.IntTensor(label)
        attn_label = torch.LongTensor(attn_label)
        prob_size, label_size = torch.IntTensor(prob_size), torch.IntTensor(label_size)
        image = image.transpose(2, 3).transpose(1, 2)
        sample = {'image': image, 'label': label, 'attn_label': attn_label, 'prob_size': prob_size, 'label_size': label_size, 'imdir': sample['imdir'], 'image_path' : image_path}
        return sample

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mean, std, device):
        self.mean = torch.FloatTensor(mean).to(device).view(1, 1, 3, 1, 1)
        self.std = torch.FloatTensor(std).to(device).view(1, 1, 3, 1, 1)

    def __call__(self, sample):
        image = sample['image']
        image = (image/255.0 - self.mean) / self.std
        sample['image'] = image
        return sample

class Pad(object):
    """Pad tensors"""
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, sample):
        if len(sample['image']) < self.max_len:
            n = self.max_len - len(sample['image'])
            sz = [n] + list(sample['image'].size()[1:])
            padded = sample['image'].new_zeros(*sz)
            sample['image'] = torch.cat((sample['image'], padded), dim=0)
        return sample

def collate_fn_ctc(data):
    num_scales = data[0]['image'].size(0)
    bsz, Nmax, imsz = len(data)*data[0]['image'].size(0), max([x['image'].size(1) for x in data]), list(data[0]['image'].size()[2:])
    #psz = list(data[0]['prior'].size()[2:])
    with torch.no_grad():
        frames = data[0]['image'].new_zeros(*([bsz, Nmax] + imsz))
        labels, prob_sizes, label_sizes = [], [], []
        attn_labels = []
        frames_path = []
        for i in range(len(data)):
            frames_path.append(data[i]['image_path'])
            
            frames[i*num_scales: (i+1)*num_scales, :data[i]['image'].size(1)] = data[i]['image']
            #priors[i*num_scales: (i+1)*num_scales, :data[i]['prior'].size(1)] = data[i]['prior']
            labels.extend([data[i]['label'] for _ in range(num_scales)])
            attn_labels.extend([data[i]['attn_label'] for _ in range(num_scales)])
            prob_sizes.extend([data[i]['prob_size'] for _ in range(num_scales)])
            label_sizes.extend([data[i]['label_size'] for _ in range(num_scales)])

        attn_labels = [l.view(-1) for l in attn_labels]
        attn_labels = pad_sequence(attn_labels, padding_value=1)
        labels = pad_sequence(labels, padding_value=1)
        labels = labels.transpose(0, 1)
        attn_labels = attn_labels.transpose(0, 1)
        
        prob_sizes, label_sizes = torch.cat(prob_sizes), torch.cat(label_sizes)
    imdir = [d['imdir'] for d in data for _ in range(num_scales)]
    
    sample = {'image': frames,
              'label': labels,
              'attn_label': attn_labels,
              'prob_size': prob_sizes,
              'label_size': label_sizes,
              'imdir': imdir,
              'image_path': frames_path}

    return sample

def collate_fn_siamese(data):

    imgs1 = [torch.Tensor(img1).transpose(2,1).transpose(1,0) for img1, img2, target, img1_label in data]
    imgs2 = [torch.Tensor(img2).transpose(2,1).transpose(1,0) for img1, img2, target, img1_label in data]
    targets = [target for img1, img2, target, img1_label in data]
    imgs1_label = [img1_label for img1, img2, target, img1_label in data]

    imgs1 = torch.stack(imgs1, 0).type(torch.float32)
    imgs2 = torch.stack(imgs2, 0).type(torch.float32)
    targets = torch.Tensor(targets)
    imgs1_label = torch.Tensor(imgs1_label).type(torch.long)

    return imgs1, imgs2, targets, imgs1_label
