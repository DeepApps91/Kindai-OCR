# Kindai-OCR
OCR system for recognizing modern Japanese magazines

## About

This repo contains an OCR sytem for converting modern Japanese images to text.
This is a result of [N2I project](http://codh.rois.ac.jp/collaboration/#n2i) for digitization of modern Japanese documents.

The system has 2 main modules: text line extraction and text line recognition. The overall architechture is shown in the below figures.
![alt text](https://github.com/ducanh841988/Kindai-OCR/blob/master/images/TextlineExtraction.jpg "text line extraction")

For text line extraction, we retrain the CRAFT (Character Region Awareness for Text Detection) on 1000 annotated images provided by Center for Research and Development of Higher Education, The University of Tokyo.
![alt text](https://github.com/ducanh841988/Kindai-OCR/blob/master/images/TextlineRecognition.jpg "text line recognition")

For text line recognition, we employ the attention-based encoder-decoder on our previous publication. We train the text line recognition on 1000 annotated images and 1600 unannotated images provided by Center for Research and Development of Higher Education and National Institute for Japanese Language and Linguistics, respectively.






## Installing Kindai OCR



## Running Kindai OCR
- You should first download the pre_trained models and put them into ./pretrain/ folder. 
[VGG model](https://drive.google.com/file/d/1_A1dEFKxyiz4Eu1HOCDbjt1OPoEh90qr/view?usp=sharing), [CRAFT model](https://drive.google.com/file/d/1-9xt_jjs4btMrz5wzrU1-kyp2c6etFab/view?usp=sharing), [OCR model](https://drive.google.com/file/d/1mibg7D2D5rvPhhenLeXNilSLMBloiexl/view?usp=sharing) 
- Copy your images into ./data/test/ folder   
- run the following script to recognize images:   
`python test.py`   
- The recognized text transcription is in ./data/result.xml and the result images are in ./data/result/   
- If you may have to check the path to Japanese font in test.py for correct visualization results.   
    `fontPIL = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf' # japanese font`   

