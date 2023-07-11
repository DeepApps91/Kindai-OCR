"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: cp932 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image, ImageDraw, ImageFont
from utils import dataIterator, load_dict, gen_sample, load_mapping
from encoder_decoder import Encoder_Decoder

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import codecs
from craft import CRAFT
from transformer.lit_bttr import LitBTTR
from torchvision.transforms import transforms
    
from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def pil2cv(imgPIL):
    imgCV_RGB = np.array(imgPIL, dtype = np.uint8)
    imgCV_BGR = np.array(imgPIL)[:, :, ::-1]
    return imgCV_BGR

def cv2pil(imgCV):
    imgCV_RGB = imgCV[:, :, ::-1]
    imgPIL = Image.fromarray(imgCV_RGB)
    return imgPIL
def cv2_putChar(draw, char, x, y, fontPIL, colorRGB):
    draw.text(xy = (x,y), text = char, fill = colorRGB, font = fontPIL)

def cv2_putText_1(img, text, org, fontFace, fontScale, color):
    min_x, max_x, min_y, max_y = org
    
    imgPIL = cv2pil(img)
    draw = ImageDraw.Draw(imgPIL)
    fontPIL = ImageFont.truetype(font = fontFace, size = fontScale)
    if max_x - min_x >= max_y- min_y:
        #horizontal line
        y =  max_y
        x = min_x
        for char in text:
             cv2_putChar(draw, char, x, y, fontPIL, color )
             w, h = draw.textsize(char, font = fontPIL)
             x += w + 10
    else:
        #vertical line
        y = min_y
        x = max_x - 10
        for char in text:
             cv2_putChar(draw, char, x, y, fontPIL, color )
             w, h = draw.textsize(char, font = fontPIL)
             y += h + 10
    imgCV = pil2cv(imgPIL)
    return imgCV


    

parser = argparse.ArgumentParser(description='Kindai document Recognition')
#params for text detection
parser.add_argument('--trained_model', default='./pretrain/synweights_4600.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--canvas_size', default=1000, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')

#params for text recognition
parser.add_argument('--model_path', default='./pretrain/transformer.ckpt', type=str)
parser.add_argument('--dictionary_target', default='./pretrain/kindai_voc.txt', type=str)



args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files('./data/test')

result_folder = './data/result2/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()
    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text



def test(text_detection_modelpara, ocr_modelpara, dictionary_target):
    # load net
    net = CRAFT()     # initialize

    print('Loading text detection model from checkpoint {}'.format(text_detection_modelpara))
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(text_detection_modelpara)))
    else:
        net.load_state_dict(copyStateDict(torch.load(text_detection_modelpara, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False



    ckp_path = "./pretrain/transformer.ckpt"
    OCR = LitBTTR.load_from_checkpoint(ocr_modelpara)


    OCR.eval()
    net.eval()

    # load dictionary
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.items():
        worddicts_r[vv] = kk
    t = time.time()

    fontPIL = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf' # japanese font
    size = 40
    colorBGR = (0,0,255) 

    
    paper = ET.Element('paper') 
    paper.set('xmlns', "http://codh.rois.ac.jp/modern-magazine/")
    # load data
    for k, image_path in enumerate(image_list[:]):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        res_img_file = result_folder + "res_" + os.path.basename(image_path)

        #print (res_img_file, os.path.basename(image_path), os.path.exists(res_img_file)) 
        #if os.path.exists(res_img_file): continue
        #image = imgproc.loadImage(image_path)
        '''image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret2,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        height = image.shape[0]
        width = image.shape[1]
        scale = 1000.0/height
        H = int(image.shape[0] * scale)
        W = int(image.shape[1] * scale)
        image = cv2.resize(image , (W, H))
        print(image.shape, image_path)
        cv2.imwrite(image_path, image) 
        continue'''
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        print(image_path)
        page = ET.SubElement(paper, "page") 
        page.set('file', os.path.basename(image_path).replace('.jpg', ''))
        page.set('height', str(h))
        page.set('width', str(w))
        page.set('dpi', str(100))
        page.set('number', str(1))

        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly)
        text = []
        localtions = []
        for i, box in enumerate(bboxes):
            poly = np.array(box).astype(np.int32)
            min_x = np.min(poly[:,0])
            max_x = np.max(poly[:,0])
            min_y = np.min(poly[:,1])
            max_y = np.max(poly[:,1])
            if min_x < 0: 
                min_x = 0
            if min_y < 0:
                min_y = 0

            #image = cv2.rectangle(image,(min_x,min_y),(max_x,max_y),(0,255,0),3)
            input_img = image[min_y:max_y, min_x:max_x]

            w = max_x - min_x + 1
            h = max_y - min_y + 1
            line = ET.SubElement(page, "line") 
            line.set("x", str(min_x))
            line.set("y", str(min_y))
            line.set("height", str(h))
            line.set("width", str(w))
            if w < h:
                rate = 20.0/w
                w = int(round(w*rate))
                h = int(round(h* rate / 20.0) * 20)
            else:
                rate = 20.0/h
                w = int(round(w*rate / 20.0) * 20)
                h = int(round(h* rate))
            #print (w, h, rate)
            
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_img = Image.fromarray(input_img)
	    
            input_img = input_img.resize((w, h), Image.LANCZOS)
            input_img = input_img.convert("L")
            input_img = transforms.ToTensor()(input_img)
            
            xx_pad = torch.zeros(1, h, w)
            xx_pad[:, :, :] = input_img # (1,H,W)
            if args.cuda:
                xx_pad.cuda()
            
            unicode_result = OCR.beam_search(xx_pad, beam_size = 3, max_len = 50)
            result = ""
            for character in unicode_result.split():
            	if character == '<eos>': continue
            	result += chr(int(character, 16))
            
            line.text = result
            text.append(result)
            image = cv2_putText_1(img = image, text = result, org = (min_x, max_x, min_y, max_y), fontFace = fontPIL, fontScale = size, color = colorBGR)


        print('save image')    
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        #cv2.imwrite(mask_file, score_text)
        file_utils.saveResult(image_path, image, polys, dirname=result_folder)

    xml_string = ET.tostring(paper, 'Shift_JIS')
       
    fout = codecs.open('./data/result.xml', 'w', 'shift_jis')
    fout.write(xml_string.decode('shift_jis'))
    fout.close()


    print("elapsed time : {}s".format(time.time() - t))

if __name__ == "__main__":
    test(args.trained_model, args.model_path, args.dictionary_target)

