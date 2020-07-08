import argparse
import numpy as np
import os
import re
import torch
import time
from utils import dataIterator, load_dict, gen_sample
from encoder_decoder import Encoder_Decoder
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
from scipy.misc import imread, imresize, imsave

def main(model_path, dictionary_target, fea, latex, saveto, output, beam_k=5):
    # model architecture
    params = {}
    params['n'] = 256
    params['m'] = 256
    params['dim_attention'] = 512
    params['D'] = 684
    params['K'] = 5748
    params['growthRate'] = 24
    params['reduction'] = 0.5
    params['bottleneck'] = True
    params['use_dropout'] = True
    params['input_channels'] = 3

    # load model
    model = Encoder_Decoder(params)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.cuda()

    # load dictionary
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

    start_time = time.time()
    channels = 1
    folder = './kokumin/'
    out = './kokuminOut/'
    index = 0

    # testing
    model.eval()
    with torch.no_grad():
        for img_file in os.listdir(folder):
            if  '.jpg' in  img_file:
                label_file = folder + 'res_' + img_file.replace('jpg', 'txt')
                if os.path.isfile(label_file) == False: continue
                out_file = out + img_file
                out_txtfile = out + img_file.replace('jpg', 'txt')
                img_file = folder + img_file
                #print img_file, label_file
                im = imread(img_file)
                arr = Image.fromarray(im).convert('RGB')
                draw = ImageDraw.Draw(arr)
   
                #print im.shape
                with open(label_file) as f:
                    BBs = f.readlines()
                BBs = [x.strip().split(',') for x in BBs]
                f = open(out_txtfile, 'w')
                for BB in BBs:
                    x1 = min(int(BB[0]), int(BB[2]), int(BB[4]), int(BB[6]))
                    y1 = min(int(BB[1]), int(BB[3]), int(BB[5]), int(BB[7]))
                    x2 = max(int(BB[0]), int(BB[2]), int(BB[4]), int(BB[6]))
                    y2 = max(int(BB[1]), int(BB[3]), int(BB[5]), int(BB[7]))
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0

                    draw.rectangle((x1, y1, x2, y2), fill=None, outline=(255, 0 , 0))

                    f.write(str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',')
                    input_img = im[y1:y2, x1:x2]
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    #print x1, y1, x2, y2
                    #print w, h
                    if w < h:
                        rate = 20.0/w
                        w = int(round(w*rate))
                        h = int(round(h* rate / 20.0) * 20)
                    else:
                        rate = 20.0/h
                        w = int(round(w*rate / 20.0) * 20)
                        h = int(round(h* rate))
                    #print w, h
                    input_img = imresize(input_img, (h,w))

                    mat = np.zeros([channels, h, w], dtype='uint8')  
                    mat[0,:,:] = input_img
                    #mat[0,:,:] =  0.299* input_img[:, :, 0] + 0.587 * input_img[:, :, 1] + 0.114 * input_img[:, :, 2]

                    xx_pad = mat.astype(np.float32) / 255.
                    xx_pad = torch.from_numpy(xx_pad[None, :, :, :]).cuda()  # (1,1,H,W)
                    sample, score, alpha_past_list = gen_sample(model, xx_pad, params, False, k=beam_k, maxlen=600)
                    score = score / np.array([len(s) for s in sample])
                    ss = sample[score.argmin()]
                    result = ''
                    for vv in ss:
                        if vv == 0: # <eol>
                            break
                        result += worddicts_r[vv] + ' '
                    print ('resutl:',  index, result)
                    f.write(result + '\n')
                f.close()
                arr.save(out_file,"JPEG")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('model_path', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('fea', type=str)
    parser.add_argument('latex', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    main(args.model_path, args.dictionary_target, args.fea, args.latex, args.saveto, args.output, beam_k=args.k)
