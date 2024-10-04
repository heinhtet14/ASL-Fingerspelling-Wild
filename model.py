import torch
import mobilenetv3
import math
import torch.nn as nn
import random
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_map, attn_model, lamda = 0.1):
        super(Model, self).__init__()

        self.encoder = SpatialEncoder(hidden_size, output_size)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.vocab_map = vocab_map

        self.attn_decoder = AttnDecoderRNN(attn_model, hidden_size, output_size)
        self.ld = 0.1

    def forward(self, imgs, labels, prob_sizes, label_sizes, attn_labels, target_length):
        device = imgs.get_device()
        attn_labels = attn_labels.transpose(0, 1)
        
        mask = torch.where(attn_labels == self.vocab_map['PAD'], 0, 1).type(torch.bool)

        logits, probs, encoder_outputs, encoder_hidden = self.encoder(imgs)
        logits_softmax = F.log_softmax(logits, dim=-1)
        loss_ctc = self.ctc_loss(logits_softmax, labels, prob_sizes, label_sizes)

        batch_size = logits.shape[1]
        decoder_input = torch.tensor([[self.vocab_map['SOS']]]*int(batch_size), device=device)
        decoder_hidden = encoder_hidden

        attn_loss, attn_pred = self.attn_decoder(decoder_input, decoder_hidden, encoder_outputs, attn_labels, mask, target_length)

        max_label = torch.max(label_sizes).item()
        labels = labels[:, :max_label]
        loss = (1-self.ld)*loss_ctc + self.ld*attn_loss
        
        probs = probs.transpose(0, 1)
        return loss, probs, attn_pred, decoder_input, decoder_hidden, encoder_outputs

class SpatialEncoder(nn.Module):
    def __init__(self, hidden_size, output_size, pretrain=None):
        super(SpatialEncoder, self).__init__()
        self.conv = mobilenetv3.mobilenet_v3_small(pretrained = './mobilenet_v3_small.pth')
        
        self.blstm = torch.nn.LSTM(input_size=1024, hidden_size=hidden_size, num_layers=1, bidirectional=False)
        self.tagg = TAGG(chanel=1024)

        self.lt = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.contex_lstm = torch.nn.LSTM(input_size=output_size, hidden_size=512, num_layers=1, bidirectional=False)

    def forward(self, Xs, Ms=None):
        """
        Xs: (B, L, C, H, W), h0: (B, n_layer, n_hidden), Ms: (B, L, h, w)
        output: (B, L, V), (B, L, V), (B, F), (B, L, h, w)
        """
            
        xsz = list(Xs.size())
        
        Xs = Xs.view(*([-1] + xsz[2:]))
        Fs = self.conv(Xs)

        fsz = list(Fs.size())
        Fs = Fs.view(*(xsz[:2]+[fsz[1]]))
        Fs = Fs.transpose(1, 0).contiguous()#[L, B, A]

        Fs_TAGG = self.tagg(Fs)

        output, (hidden, cell) = self.blstm(Fs_TAGG)
        
        ys = output
        logits = self.lt(ys)

        probs = self.softmax(logits)
        return logits, probs, output, (hidden, cell)

################################################################################################################################

class TAGG(nn.Module):
    def __init__(self, chanel=1024):
        super(TAGG, self).__init__()

        
        self.conv1d_w3 = nn.Sequential(nn.Conv1d(1024, 384, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv1d(384, 384, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True))

        self.conv1d_w5 = nn.Sequential(nn.Conv1d(1024, 384, kernel_size=5, padding=2),
                            nn.ReLU(inplace=True),
                            nn.Conv1d(384, 384, kernel_size=5, padding=2),
                            nn.ReLU(inplace=True))

        self.conv1d_w7 = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=9, padding=4),
                            nn.ReLU(inplace=True),
                            nn.Conv1d(256, 256, kernel_size=7, padding=3),
                            nn.ReLU(inplace=True))
                
        self.gate = torch.nn.GRUCell(chanel, chanel)

    def forward(self, Fs):
        '''
        Input : [L, B, A]
        '''

        L_enc, bsz, _ = list(Fs.size())

        temp_Fs = Fs.transpose(0, 1).transpose(1, 2) # temp_Fs -> [B, A, L]

        temp_Fs_w3 = self.conv1d_w3(temp_Fs)
        temp_Fs_w5 = self.conv1d_w5(temp_Fs)
        temp_Fs_w7 = self.conv1d_w7(temp_Fs)

        temp_Fs = torch.cat((temp_Fs_w3, temp_Fs_w5, temp_Fs_w7), 1)

        temp_Fs = temp_Fs.transpose(2, 1).transpose(1, 0).contiguous() # temp_Fs -> [L, B, A]
        temp_Fs = temp_Fs.view(bsz*L_enc, -1)

        Fs = Fs.view(L_enc*bsz, -1)

        agg_Fs = self.gate(temp_Fs, Fs)
        agg_Fs = agg_Fs.view(L_enc, bsz, -1)

        return agg_Fs

################################################################################################################################
class LocationAwareAttention(nn.Module):
    """
    Applies a location-aware attention mechanism on the output features from the decoder.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.
    Args:
        hidden_dim (int): dimesion of hidden state vector
        smoothing (bool): flag indication whether to use smoothing or not.
    Inputs: query, value, last_attn, smoothing
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoder.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size * num_heads, v_len): tensor containing previous timestep`s attention (alignment)
    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoder outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    Reference:
        - **Attention-Based Models for Speech Recognition**: https://arxiv.org/abs/1506.07503
        - **ClovaCall**: https://github.com/clovaai/ClovaCall/blob/master/las.pytorch/models/attention.py
    """
    def __init__(self, hidden_dim: int, smoothing: bool = True) -> None:
        super(LocationAwareAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.smoothing = smoothing

    def forward(self, query, value, last_attn) :
        
        query = query.transpose(0, 1)
        value = value.transpose(0, 1)

        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        # Initialize previous attention (alignment) to zeros
        if last_attn is None:
            last_attn = value.new_zeros(batch_size, seq_len)

        conv_attn = torch.transpose(self.conv1d(last_attn.unsqueeze(1)), 1, 2)
        score = self.score_proj(torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + conv_attn
                + self.bias
        )).squeeze(dim=-1)

        if self.smoothing:
            score = torch.sigmoid(score)
            attn = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn = F.softmax(score, dim=-1)

        #context = torch.bmm(attn.unsqueeze(dim=1), value).squeeze(dim=1)  # Bx1xT X BxTxD => Bx1xD => BxD

        return attn
        
# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

################################################################################################################################

class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        #self.attn = Attn('dot', hidden_size)
        self.attn = LocationAwareAttention(hidden_size)

    def decoder(self, input_step, last_hidden, last_attn_weights, encoder_outputs):

        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        batch_size = last_hidden[0].shape[1]
        embedded = self.embedding(input_step).view(1, batch_size, -1)
        embedded = self.embedding_dropout(embedded)

        # Forward
        rnn_output, (hidden,cell) = self.lstm(embedded, last_hidden)

        # Calculate attention weights from the current GRU output
        #print(rnn_output.shape, encoder_outputs.shape)
        attn_weights = self.attn(rnn_output, encoder_outputs, last_attn_weights)
        attn_weights = attn_weights.view(batch_size, 1, -1)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        #output = F.softmax(output, dim=1)
        # Return output and final hidden state
       
        return output, (hidden,cell), attn_weights.view(batch_size, -1)

    def eval(self, init_decoder_input, decoder_hidden, encoder_outputs, labels, mask, target_length, max_length, vocab_map) :

        eos_id = vocab_map['EOS']
        loss = 0
        attn_weight_list = []
        outputs = []
        pred = []
        attn_weights = None
        for di in range(max_length): #40 is max length
            decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, attn_weights, encoder_outputs)
            outputs.append(decoder_output)

            decoder_output = F.softmax(decoder_output, dim=1)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            pred.append(torch.argmax(decoder_output, dim=1))
            attn_weight_list.append(attn_weights)

            if di >= target_length :
                continue
            else :
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, labels[di], mask[di])
                loss += mask_loss

        pred = torch.stack(pred, dim=0)
        attn_weights = torch.concat(attn_weight_list, dim=1)

        return loss, pred, attn_weights

    def forward(self, init_decoder_input, decoder_hidden, encoder_outputs, labels, mask, target_length):

        loss = 0
        pred = []
        outputs = []
        use_teacher_forcing = True if random.random() < 0.2 else False
        decoder_input = init_decoder_input
        attn_weights = None
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, attn_weights, encoder_outputs)
                outputs.append(decoder_output)

                decoder_output = F.softmax(decoder_output, dim=1)
                decoder_input = labels[di]  # Teacher forcing

                pred.append(torch.argmax(decoder_output, dim=1))

                if not torch.any(mask[di]) :
                    continue

                mask_loss, nTotal = self.maskNLLLoss(decoder_output, labels[di], mask[di])
                loss += mask_loss
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, attn_weights, encoder_outputs)
                outputs.append(decoder_output)

                decoder_output = F.softmax(decoder_output, dim=1)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                pred.append(torch.argmax(decoder_output, dim=1))

                if not torch.any(mask[di]) :
                    continue

                mask_loss, nTotal = self.maskNLLLoss(decoder_output, labels[di], mask[di])
                loss += mask_loss
                
        pred = torch.stack(pred, dim=0)
        pred = pred.transpose(0, 1)

        return loss, pred

    def maskNLLLoss(self, inp, target, mask):

        device = target.get_device()
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)

        return loss, nTotal.item()