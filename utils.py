 #!/usr/bin/env python
import numpy as np
import copy
import sys
import pickle as pkl
import torch
from torch import nn

# load data
def dataIterator(feature_file, label_file, dictionary, batch_size, batch_Imagesize, maxlen, maxImagesize):
    # offline-train.pkl
    fp = open(feature_file, 'rb')
    features = pkl.load(fp, encoding='latin1')
    fp.close()

    # train_caption.txt
    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()

    targets = {}
    # map word to int with dictionary
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0]
        w_list = []
        for w in tmp[1:]:
            if dictionary.__contains__(w):
                w_list.append(dictionary[w])
            else:
                #print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                print(w + '\t' + str(len(dictionary)))
                dictionary[w] = len(dictionary)
                #sys.exit()
        targets[uid] = w_list

    imageSize = {}
    for uid, fea in features.items():
        imageSize[uid] = fea.shape[1] * fea.shape[2]
    # sorted by sentence length, return a list with each triple element
    imageSize = sorted(imageSize.items(), key=lambda d: d[1])

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = []
    biggest_image_size = 0

    i = 0
    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        lab = targets[uid]
        batch_image_size = biggest_image_size * (i + 1)
        if len(lab) > maxlen:
            print('sentence', uid, 'length bigger than', maxlen, 'ignore')
        elif size > maxImagesize:
            print(size)
            print('image', uid, 'size bigger than', maxImagesize, 'ignore')
        else:
            uidList.append(uid)
            if batch_image_size > batch_Imagesize or i == batch_size:  # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)
                i = 0
                biggest_image_size = size
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)
    print('total ', len(feature_total), 'batch data loaded')
    return list(zip(feature_total, label_total)), uidList


# load dictionary
def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon


# load mapping
def load_mapping(dictFile):
    print (dictFile)
    lexicon={}
    lexicon_r ={}
    with open(dictFile,'r') as f:
        lines = f.readlines()
        for line in lines:
            sp = line.split()
            lexicon[sp[1]]=unicode(sp[0], 'Shift_JISx0213')
            lexicon_r[unicode(sp[0], 'Shift_JISx0213')]=sp[1]
            

    print ('total words/phones',len(lexicon))
    return lexicon, lexicon_r
    
# create batch
def prepare_data(options, images_x, seqs_y, prev_x = None):
    
    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(heights_x)
    max_height_x = np.max(heights_x)
    max_width_x = np.max(widths_x)
    maxlen_y = np.max(lengths_y) + 1
    x = np.zeros((n_samples, options['input_channels'], max_height_x, max_width_x)).astype(np.float32)
    y = np.zeros((maxlen_y, n_samples)).astype(np.int64)  # <eos> must be 0 in the dict
    x_mask = np.zeros((n_samples, max_height_x, max_width_x)).astype(np.float32)
    y_mask = np.zeros((maxlen_y, n_samples)).astype(np.float32)
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx] + 1, idx] = 1.
    return x, x_mask, y, y_mask


# beam search
def gen_sample(model, x, params, gpu_flag, k=1, maxlen=30):
    sample = []
    sample_score = []
    sample_alpha = []
    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = np.zeros(live_k).astype(np.float32)
    hyp_alpha_past = [[]] * live_k

    if gpu_flag:
        next_state, ctx0 = model.module.f_init(x)
    else:
        next_state, ctx0 = model.f_init(x)
    next_w = -1 * np.ones((1,)).astype(np.int64)
    next_w = torch.from_numpy(next_w)
    next_alpha_past = torch.zeros(1, ctx0.shape[2], ctx0.shape[3])
    ctx0 = ctx0.cpu().numpy()

    if gpu_flag:
        next_w.cuda()
        next_alpha_past.cuda()

    for ii in range(maxlen):
        ctx = np.tile(ctx0, [live_k, 1, 1, 1])
        ctx = torch.from_numpy(ctx)
        if gpu_flag:
            ctx.cuda()
            next_w.cuda()
            next_state.cuda()
            next_alpha_past.cuda()

            next_p, next_state, next_alpha_past, alpha = model.module.f_next(params, next_w, None, ctx, None, next_state,
                                                                      next_alpha_past, True)
        else:
            next_p, next_state, next_alpha_past, alpha = model.f_next(params, next_w, None, ctx, None, next_state,
                                                               next_alpha_past, True)
        next_p = next_p.cpu().numpy()
        next_state = next_state.cpu().numpy()
        next_alpha_past = next_alpha_past.cpu().numpy()

        cand_scores = hyp_scores[:, None] - np.log(next_p)
        cand_flat = cand_scores.flatten()

        ranks_flat = cand_flat.argsort()[:(k - dead_k)]
        voc_size = next_p.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = np.zeros(k - dead_k).astype(np.float32)
        new_hyp_states = []
        new_hyp_alpha_past = []
        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))
            new_hyp_alpha_past.append(hyp_alpha_past[ti] + [copy.copy(next_alpha_past[ti])])
        #print (new_hyp_alpha_past)
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        hyp_alpha_past = []
        for idx in range(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == 0:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                sample_alpha.append(new_hyp_alpha_past[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
                hyp_alpha_past.append(new_hyp_alpha_past[idx])
        #print (hyp_alpha_past)
        hyp_scores = np.array(hyp_scores)
        live_k = new_live_k

        # whether finish beam search
        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = np.array([w[-1] for w in hyp_samples])
        next_state = np.array(hyp_states)
        #next_alpha_past = np.array(hyp_alpha_past)
        next_alpha_past = np.array([w[-1] for w in hyp_alpha_past])
        #print (np.shape(next_alpha_past))
        next_w = torch.from_numpy(next_w)
        next_state = torch.from_numpy(next_state)
        next_alpha_past = torch.from_numpy(next_alpha_past)
    return sample, sample_score, sample_alpha






# init model params
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        try:
            nn.init.constant_(m.bias.data, 0.)
        except:
            pass
