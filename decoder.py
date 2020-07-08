import torch
import torch.nn as nn


# two layers of GRU
class Gru_cond_layer(nn.Module):
    def __init__(self, params):
        super(Gru_cond_layer, self).__init__()
        # attention
        self.conv_Ua = nn.Conv2d(params['D'], params['dim_attention'], kernel_size=1)
        self.fc_Wa = nn.Linear(params['n'], params['dim_attention'], bias=False)
        self.conv_Q = nn.Conv2d(1, 512, kernel_size=11, bias=False, padding=5)
        self.fc_Uf = nn.Linear(512, params['dim_attention'])
        self.fc_va = nn.Linear(params['dim_attention'], 1)

        # the first GRU layer
        self.fc_Wyz = nn.Linear(params['m'], params['n'])
        self.fc_Wyr = nn.Linear(params['m'], params['n'])
        self.fc_Wyh = nn.Linear(params['m'], params['n'])

        self.fc_Uhz = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhr = nn.Linear(params['n'], params['n'], bias=False)
        self.fc_Uhh = nn.Linear(params['n'], params['n'], bias=False)

        # the second GRU layer
        self.fc_Wcz = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wcr = nn.Linear(params['D'], params['n'], bias=False)
        self.fc_Wch = nn.Linear(params['D'], params['n'], bias=False)

        self.fc_Uhz2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhr2 = nn.Linear(params['n'], params['n'])
        self.fc_Uhh2 = nn.Linear(params['n'], params['n'])

    def forward(self, params, embedding, mask=None, context=None, context_mask=None, one_step=False, init_state=None,
                alpha_past=None):
        n_steps = embedding.shape[0]
        n_samples = embedding.shape[1]

        Ua_ctx = self.conv_Ua(context)
        Ua_ctx = Ua_ctx.permute(2, 3, 0, 1) 
        state_below_z = self.fc_Wyz(embedding)
        state_below_r = self.fc_Wyr(embedding)
        state_below_h = self.fc_Wyh(embedding)

        if one_step:
            if mask is None:
                mask = torch.ones(embedding.shape[0]).cuda()
            h2ts, cts, alphas, alpha_pasts = self._step_slice(mask, state_below_r, state_below_z, state_below_h,
                                                              init_state, context, context_mask, alpha_past, Ua_ctx)
        else:
            alpha_past = torch.zeros(n_samples, context.shape[2], context.shape[3]).cuda()
            h2t = init_state
            h2ts = torch.zeros(n_steps, n_samples, params['n']).cuda()
            cts = torch.zeros(n_steps, n_samples, params['D']).cuda()
            alphas = (torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3])).cuda()
            alpha_pasts = torch.zeros(n_steps, n_samples, context.shape[2], context.shape[3]).cuda()
            for i in range(n_steps):
                h2t, ct, alpha, alpha_past = self._step_slice(mask[i], state_below_r[i], state_below_z[i],
                                                              state_below_h[i], h2t, context, context_mask, alpha_past,
                                                              Ua_ctx)
                h2ts[i] = h2t
                cts[i] = ct
                alphas[i] = alpha
                alpha_pasts[i] = alpha_past
        return h2ts, cts, alphas, alpha_pasts

    # one step of two GRU layers
    def _step_slice(self, mask, state_below_r, state_below_z, state_below_h, h, ctx, ctx_mask, alpha_past, Ua_ctx):
        # the first GRU layer
        z1 = torch.sigmoid(self.fc_Uhz(h) + state_below_z)
        r1 = torch.sigmoid(self.fc_Uhr(h) + state_below_r)
        h1_p = torch.tanh(self.fc_Uhh(h) * r1 + state_below_h)
        h1 = z1 * h + (1. - z1) * h1_p
        h1 = mask[:, None] * h1 + (1. - mask)[:, None] * h

        # attention
        Wa_h1 = self.fc_Wa(h1)
        alpha_past_ = alpha_past[:, None, :, :]
        cover_F = self.conv_Q(alpha_past_).permute(2, 3, 0, 1)
        cover_vector = self.fc_Uf(cover_F)
        attention_score = torch.tanh(Ua_ctx + Wa_h1[None, None, :, :] + cover_vector)
        alpha = self.fc_va(attention_score)
        alpha = alpha.view(alpha.shape[0], alpha.shape[1], alpha.shape[2])
        alpha = torch.exp(alpha)
        if (ctx_mask is not None):
            alpha = alpha * ctx_mask.permute(1, 2, 0)
        alpha = alpha / alpha.sum(1).sum(0)[None, None, :]
        alpha_past = alpha_past + alpha.permute(2, 0, 1)
        ct = (ctx * alpha.permute(2, 0, 1)[:, None, :, :]).sum(3).sum(2)

        # the second GRU layer
        z2 = torch.sigmoid(self.fc_Wcz(ct) + self.fc_Uhz2(h1))
        r2 = torch.sigmoid(self.fc_Wcr(ct) + self.fc_Uhr2(h1))
        h2_p = torch.tanh(self.fc_Wch(ct) + self.fc_Uhh2(h1) * r2)
        h2 = z2 * h1 + (1. - z2) * h2_p
        h2 = mask[:, None] * h2 + (1. - mask)[:, None] * h1
        return h2, ct, alpha.permute(2, 0, 1), alpha_past


# calculate probabilities
class Gru_prob(nn.Module):
    def __init__(self, params):
        super(Gru_prob, self).__init__()
        self.fc_Wct = nn.Linear(params['D'], params['m'])
        self.fc_Wht = nn.Linear(params['n'], params['m'])
        self.fc_Wyt = nn.Linear(params['m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_W0 = nn.Linear(int(params['m'] / 2), params['K'])

    def forward(self, cts, hts, emb, use_dropout):
        logit = self.fc_Wct(cts) + self.fc_Wht(hts) + self.fc_Wyt(emb)

        # maxout
        shape = logit.shape
        shape2 = int(shape[2] / 2)
        shape3 = 2
        logit = logit.view(shape[0], shape[1], shape2, shape3)
        logit = logit.max(3)[0]

        if use_dropout:
            logit = self.dropout(logit)

        out = self.fc_W0(logit)
        return out
