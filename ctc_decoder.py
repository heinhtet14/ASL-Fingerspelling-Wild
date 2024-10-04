import math
import numpy as np
import torch
import torch.nn.functional as F


class Decoder(object):
    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.char_to_int = dict([(c, i) for (i, c) in enumerate(labels)])

        print(self.int_to_char)
        print(self.char_to_int)

        self.blank_index = blank_index
        self.EOS_index = self.char_to_int['EOS']
        self.PAD_index = self.char_to_int['PAD']
        self.SOS_index = self.char_to_int['SOS']

        space_index = len(labels)
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def attn_beam(self, beam_size, decoder_input, decoder_hidden, encoder_outputs, attn_decoder, max_len=40):

        init_decoder_input = decoder_input
        init_decoder_hidden = decoder_hidden

        attn_history = {}
        attn_history['SOS'] = (0, decoder_hidden)

        #attn t=0
        attn_weights = None
        decoder_output, decoder_hidden, attn_weights = attn_decoder.decoder(init_decoder_input, init_decoder_hidden, attn_weights, encoder_outputs)
        decoder_prob = F.softmax(decoder_output, dim=1)

        topk_v, topk_i = decoder_prob.topk(beam_size)
        topk_i = topk_i.squeeze().detach().cpu().tolist()
        topk_v = [ math.log(v) for v in topk_v.squeeze().detach().cpu().tolist()]

        topk_i = [[i] for i in topk_i]
        topk_h = [decoder_hidden] * len(topk_i)
        topk_a = [attn_weights] * len(topk_i)

        #topk_v = [[v] for v in topk_v]

        for t in range(1, max_len) :
            topk_list = []
            topk_score_list = []
            topk_h_list = []
            topk_a_list = []
            init_topk_i = topk_i
            init_topk_v = topk_v
            for beam, hidden, attn in zip(init_topk_i, topk_h, topk_a):
                decoder_hidden = hidden
                prev_attn_weights = attn
                
                if beam[-1] == self.EOS_index :
                    topk_list.append([])
                    topk_score_list.append([])
                    topk_h_list.append([])
                    topk_a_list.append([])
                    continue
                
                decoder_input = torch.tensor(beam[-1]).cuda().view(1, 1)
                decoder_output, decoder_hidden, attn_weights = attn_decoder.decoder(decoder_input, decoder_hidden, prev_attn_weights, encoder_outputs)
                
                decoder_prob = F.softmax(decoder_output, dim=1)
                topk_v, topk_i = decoder_prob.topk(beam_size)
                
                topk_i = topk_i.squeeze().detach().cpu().tolist()
                topk_v = topk_v.squeeze().detach().cpu().tolist()

                topk_list.append(topk_i)
                topk_score_list.append(topk_v)
                topk_h_list.append(decoder_hidden)
                topk_a_list.append(attn_weights)

            #augment
            aug_topk_i = []
            aug_topk_v = []
            aug_topk_h = []
            aug_topk_a = []
            for i in range(len(init_topk_i)) :
                if topk_list[i] == [] :
                    aug_topk_i.append(init_topk_i[i])
                    aug_topk_v.append(init_topk_v[i])
                    aug_topk_h.append(topk_h_list[i])
                    aug_topk_a.append(topk_a_list[i])
                else :
                    for j in range(len(topk_list[i])) :
                        aug_topk_i.append(init_topk_i[i] + [topk_list[i][j]])
                        aug_topk_v.append(init_topk_v[i] + math.log(topk_score_list[i][j]))
                        aug_topk_h.append(topk_h_list[i])
                        aug_topk_a.append(topk_a_list[i])

            topk_idx = np.argsort(np.array(aug_topk_v))[-beam_size:].tolist()
            topk_i = [aug_topk_i[i] for i in topk_idx]
            topk_v = [aug_topk_v[i] for i in topk_idx]
            topk_h = [aug_topk_h[i] for i in topk_idx]
            topk_a = [aug_topk_a[i] for i in topk_idx]

        #max_idx = np.argsort(np.array(topk_v))[-1:].tolist()
        #pred = [topk_i[i] for i in max_idx]
        #score = [topk_v[i] for i in max_idx]
        #clean_idx = [idx for idx, i in enumerate(topk_i) if i != self.EOS_index]
        
        topk_i = list(map(lambda x: [i for i in x if i != self.EOS_index], topk_i))
            

        return topk_i, topk_v

    def greedy_decode(self, prob, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        indexes = np.argmax(prob, axis=1)
        string = []
        prev_index = -1
        for i in range(len(indexes)):
            if indexes[i] == self.blank_index:
                prev_index = -1
                continue
            elif indexes[i] == prev_index:
                continue
            else:
                if digit is False:
                    string.append(self.int_to_char[indexes[i]])
                else:
                    string.append(indexes[i])
                prev_index = indexes[i]
        return string

    def beam_decode(self, prob, beam_size, beta=0.0, gamma=0.0, scorer=None, digit=False):
        # prob: [seq_len, num_labels+1], numpy array
        # beta: lm coef, gamma: insertion coef
        seqlen = len(prob)
        beam_idx = np.argsort(prob[0, :])[-beam_size:].tolist()
        beam_prob = list(map(lambda x: math.log(prob[0, x]), beam_idx))
        beam_idx = list(map(lambda x: [x], beam_idx))
        for t in range(1, seqlen):
            topk_idx = np.argsort(prob[t, :])[-beam_size:].tolist()
            topk_prob = list(map(lambda x: prob[t, x], topk_idx))

            aug_beam_prob, aug_beam_idx = [], []
            for b in range(beam_size*beam_size):
                aug_beam_prob.append(beam_prob[int(b/beam_size)])
                aug_beam_idx.append(list(beam_idx[int(b/beam_size)]))
            # allocate
            for b in range(beam_size*beam_size):
                i, j = b/beam_size, b % beam_size
                aug_beam_idx[b].append(topk_idx[j])
                aug_beam_prob[b] = aug_beam_prob[b]+math.log(topk_prob[j])
            # merge
            merge_beam_idx, merge_beam_prob = [], []
            for b in range(beam_size*beam_size):
                if aug_beam_idx[b][-1] == aug_beam_idx[b][-2]:
                    beam, beam_prob = aug_beam_idx[b][:-1], aug_beam_prob[b]
                elif aug_beam_idx[b][-2] == self.blank_index:
                    beam, beam_prob = aug_beam_idx[b][:-2]+[aug_beam_idx[b][-1]], aug_beam_prob[b]
                else:
                    beam, beam_prob = aug_beam_idx[b], aug_beam_prob[b]
                beam_str = list(map(lambda x: self.int_to_char[x], beam))
                if beam_str not in merge_beam_idx:
                    merge_beam_idx.append(beam_str)
                    merge_beam_prob.append(beam_prob)
                else:
                    idx = merge_beam_idx.index(beam_str)
                    merge_beam_prob[idx] = np.logaddexp(merge_beam_prob[idx], beam_prob)

            if scorer is not None:
                merge_beam_prob_lm, ins_bonus, strings = [], [], []
                for b in range(len(merge_beam_prob)):
                    if merge_beam_idx[b][-1] == self.int_to_char[self.blank_index]:
                        
                        add_string = [char for char in merge_beam_idx[b][:-1] if char != self.int_to_char[self.PAD_index] \
                                                                            and char != self.int_to_char[self.EOS_index]\
                                                                            and char != self.int_to_char[self.SOS_index]]

                        strings.append(add_string)
                        ins_bonus.append(len(add_string))
                    else:

                        add_string = [char for char in merge_beam_idx[b] if char != self.int_to_char[self.PAD_index] \
                                                                            and char != self.int_to_char[self.EOS_index]\
                                                                            and char != self.int_to_char[self.SOS_index]]

                        strings.append(add_string)
                        ins_bonus.append(len(add_string))

                lm_scores = scorer.get_score_fast(strings)
                for b in range(len(merge_beam_prob)):
                    total_score = merge_beam_prob[b]+beta*lm_scores[b]+gamma*ins_bonus[b]
                    merge_beam_prob_lm.append(total_score)

            if scorer is None:
                ntopk_idx = np.argsort(np.array(merge_beam_prob))[-beam_size:].tolist()
            else:
                ntopk_idx = np.argsort(np.array(merge_beam_prob_lm))[-beam_size:].tolist()
            beam_idx = list(map(lambda x: merge_beam_idx[x], ntopk_idx))
            for b in range(len(beam_idx)):
                beam_idx[b] = list(map(lambda x: self.char_to_int[x], beam_idx[b]))
            beam_prob = list(map(lambda x: merge_beam_prob[x], ntopk_idx))
        if self.blank_index in beam_idx[-1]:
            pred = beam_idx[-1][:-1]
        else:
            pred = beam_idx[-1]
        if digit is False:
            pred = list(map(lambda x: self.int_to_char[x], pred))
        return pred

    def multi_decoder(self, prob, beam_size, decoder_input, decoder_hidden, encoder_outputs, attn_decoder, beta=0.0, gamma=0.0, scorer=None, digit=True):

        attn_topk_i, attn_topk_v = self.attn_beam(beam_size, decoder_input, decoder_hidden, encoder_outputs, attn_decoder)

        # prob: [seq_len, num_labels+1], numpy array
        # beta: lm coef, gamma: insertion coef
        seqlen = len(prob)
        beam_idx = np.argsort(prob[0, :])[-beam_size:].tolist()
        beam_prob = list(map(lambda x: math.log(prob[0, x]), beam_idx))
        beam_idx = list(map(lambda x: [x], beam_idx))

        for t in range(1, seqlen):
            topk_idx = np.argsort(prob[t, :])[-beam_size:].tolist() #find topk prob based on beam size
            topk_prob = list(map(lambda x: prob[t, x], topk_idx))

            aug_beam_prob, aug_beam_idx = [], []
            for b in range(beam_size*beam_size):
                aug_beam_prob.append(beam_prob[int(b/beam_size)])
                aug_beam_idx.append(list(beam_idx[int(b/beam_size)]))

            # allocate
            for b in range(beam_size*beam_size):
                i, j = b/beam_size, b % beam_size
                aug_beam_idx[b].append(topk_idx[j])
                aug_beam_prob[b] = aug_beam_prob[b]+math.log(topk_prob[j])

            # merge
            merge_beam_idx, merge_beam_prob = [], []
            for b in range(beam_size*beam_size):
                if aug_beam_idx[b][-1] == aug_beam_idx[b][-2]:
                    beam, beam_prob = aug_beam_idx[b][:-1], aug_beam_prob[b] #ignore repeated character
                elif aug_beam_idx[b][-2] == self.blank_index:
                    beam, beam_prob = aug_beam_idx[b][:-2]+[aug_beam_idx[b][-1]], aug_beam_prob[b] #ignore blank label between character
                else:
                    beam, beam_prob = aug_beam_idx[b], aug_beam_prob[b]
                beam_str = list(map(lambda x: self.int_to_char[x], beam))
                if beam_str not in merge_beam_idx:
                    merge_beam_idx.append(beam_str)
                    merge_beam_prob.append(beam_prob)
                else:
                    idx = merge_beam_idx.index(beam_str)
                    merge_beam_prob[idx] = np.logaddexp(merge_beam_prob[idx], beam_prob)
            
            if scorer is not None:
                merge_beam_prob_lm, ins_bonus, strings = [], [], []
                for b in range(len(merge_beam_prob)):
                    if merge_beam_idx[b][-1] == self.int_to_char[self.blank_index]:
                        
                        #ignore blank label
                        add_string = [char for char in merge_beam_idx[b][:-1] if char != self.int_to_char[self.PAD_index] \
                                                                            and char != self.int_to_char[self.EOS_index]\
                                                                            and char != self.int_to_char[self.SOS_index]] #ignore PAD EOS SOS label

                        strings.append(add_string)
                        ins_bonus.append(len(add_string))
                    else:

                        add_string = [char for char in merge_beam_idx[b] if char != self.int_to_char[self.PAD_index] \
                                                                            and char != self.int_to_char[self.EOS_index]\
                                                                            and char != self.int_to_char[self.SOS_index]] #ignore PAD EOS SOS label

                        strings.append(add_string)
                        ins_bonus.append(len(add_string))

                lm_scores = scorer.get_score_fast(strings)
                for b in range(len(merge_beam_prob)):
                    total_score = merge_beam_prob[b]+beta*lm_scores[b]+gamma*ins_bonus[b]
                    merge_beam_prob_lm.append(total_score)
            
            #if t != seqlen-1 :
            if scorer is None:
                ntopk_idx = np.argsort(np.array(merge_beam_prob))[-beam_size:].tolist()
            else:
                ntopk_idx = np.argsort(np.array(merge_beam_prob_lm))[-beam_size:].tolist()

            '''
            else :
                if scorer is None:
                    ntopk_idx = np.argsort(np.array(merge_beam_prob)).tolist()
                else:
                    ntopk_idx = np.argsort(np.array(merge_beam_prob_lm)).tolist()
            '''

            beam_idx = list(map(lambda x: merge_beam_idx[x], ntopk_idx))
            for b in range(len(beam_idx)):
                beam_idx[b] = list(map(lambda x: self.char_to_int[x], beam_idx[b]))
            beam_prob = list(map(lambda x: merge_beam_prob[x], ntopk_idx))

        for idx in range(len(beam_idx)) :
            if self.blank_index in beam_idx[idx]:
                beam_idx[idx] = beam_idx[idx][:-1]

        attn_topk_v = np.array(attn_topk_v)
        attn_topk_v = list((attn_topk_v - np.amin(attn_topk_v))/(np.amax(attn_topk_v) - np.amin(attn_topk_v)))

        max_idx = np.argsort(np.array(attn_topk_v)).tolist()
        attn_pred = [attn_topk_i[i] for i in max_idx][-1] 
        if digit is False:
            attn_pred = list(map(lambda x: self.int_to_char[x], attn_pred))

        beam_prob = np.array(beam_prob)
        beam_prob = list((beam_prob - np.amin(beam_prob))/(np.amax(beam_prob) - np.amin(beam_prob)))
        
        '''
        for i in range(len(beam_idx)) :
            for j in range(len(attn_topk_i)) :
                if beam_idx[i] == attn_topk_i[j] :
                    attn_topk_v[j] += beam_prob[i]

        max_idx = np.argsort(np.array(attn_topk_v)).tolist()
        attn_topk_i = [attn_topk_i[i] for i in max_idx]
        pred = attn_topk_i[-1] 
        if digit is False:
            pred = list(map(lambda x: self.int_to_char[x], pred))
        '''

        for i in range(len(attn_topk_i)) :
            for j in range(len(beam_idx)) :
                if attn_topk_i[i] == beam_idx[j] :
                    beam_prob[j] += attn_topk_v[i]
                else :
                    pass
                    #beam_idx.append(attn_topk_i[i])
                    #beam_prob.append(attn_topk_v[i])

        max_idx = np.argsort(np.array(beam_prob)).tolist()
        beam_idx = [beam_idx[i] for i in max_idx]
        pred = beam_idx[-1] 
        if digit is False:
            pred = list(map(lambda x: self.int_to_char[x], pred))

        return pred, attn_pred
