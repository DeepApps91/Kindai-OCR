import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import DenseNet
from decoder import Gru_cond_layer, Gru_prob


# create gru init state
class FcLayer(nn.Module):
    def __init__(self, nin, nout):
        super(FcLayer, self).__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        out = torch.tanh(self.fc(x))
        return out


# Embedding
class My_Embedding(nn.Module):
    def __init__(self, params):
        super(My_Embedding, self).__init__()
        self.embedding = nn.Embedding(params['K'], params['m'])
        self.cuda = params['cuda']

    def forward(self, params, y):
        if y.sum() < 0.: 
            emb = torch.zeros(1, params['m'])
            if self.cuda:
                emb.cuda()
        else:
            emb = self.embedding(y)
            if len(emb.shape) == 3:  # only for training stage
                emb_shifted = torch.zeros([emb.shape[0], emb.shape[1], params['m']], dtype=torch.float32)
                if self.cuda:
                    emb_shifted.cuda()
                emb_shifted[1:] = emb[:-1]
                emb = emb_shifted
        return emb


class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.encoder = DenseNet(growthRate=params['growthRate'], reduction=params['reduction'],
                                bottleneck=params['bottleneck'], use_dropout=params['use_dropout'])
        self.init_GRU_model = FcLayer(params['D'], params['n'])
        self.emb_model = My_Embedding(params)
        self.gru_model = Gru_cond_layer(params)
        self.gru_prob_model = Gru_prob(params)
        self.cuda = params['cuda']

    def forward(self, params, x, x_mask, y, y_mask, one_step=False):
        # recover permute
        y = y.permute(1, 0)
        y_mask = y_mask.permute(1, 0)

        ctx, ctx_mask = self.encoder(x, x_mask)

        # init state
        ctx_mean = (ctx * ctx_mask[:, None, :, :]).sum(3).sum(2) / ctx_mask.sum(2).sum(1)[:, None]
        init_state = self.init_GRU_model(ctx_mean)

        # two GRU layers
        emb = self.emb_model(params, y)
        h2ts, cts, alphas, _alpha_pasts = self.gru_model(params, emb, y_mask, ctx, ctx_mask, one_step, init_state,
                                                         alpha_past=None)
        scores = self.gru_prob_model(cts, h2ts, emb, use_dropout=params['use_dropout'])

        # permute for multi-GPU training
        alphas = alphas.permute(1, 0, 2, 3)
        scores = scores.permute(1, 0, 2)
        return scores, alphas

    # decoding: encoder part
    def f_init(self, x, x_mask=None):
        if x_mask is None:
            shape = x.shape
            x_mask = torch.ones(shape).cuda()
        ctx, _ctx_mask = self.encoder(x, x_mask)
        ctx_mean = ctx.mean(dim=3).mean(dim=2)
        init_state = self.init_GRU_model(ctx_mean)
        return init_state, ctx

    # decoding: decoder part
    def f_next(self, params, y, y_mask, ctx, ctx_mask, init_state, alpha_past, one_step):
        emb_beam = self.emb_model(params, y)

        # one step of two gru layers
        next_state, cts, _alpha, next_alpha_past = self.gru_model(params, emb_beam, y_mask, ctx, ctx_mask,
                                                                  one_step, init_state, alpha_past)
        # reshape to suit GRU step code
        next_state_ = next_state.view(1, next_state.shape[0], next_state.shape[1])
        cts = cts.view(1, cts.shape[0], cts.shape[1])
        emb_beam = emb_beam.view(1, emb_beam.shape[0], emb_beam.shape[1])

        # calculate probabilities
        scores = self.gru_prob_model(cts, next_state_, emb_beam, use_dropout=params['use_dropout'])
        scores = scores.view(-1, scores.shape[2])
        next_probs = F.softmax(scores, dim=1)
        return next_probs, next_state, next_alpha_past, _alpha
