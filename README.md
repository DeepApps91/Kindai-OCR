# Kindai-OCR
OCR system for recognizing modern Japanese magazines

## About

This repo contains an OCR sytem for converting modern Japanese images to text.
This is a result of [N2I project](http://codh.rois.ac.jp/collaboration/#n2i) for digitization of modern Japanese documents.

The system has 2 main modules: text line extraction and text line recognition. The overall architechture is shown in the below figures.
![alt text](https://github.com/ducanh841988/Kindai-OCR/blob/master/images/TextlineExtraction.jpg "text line extraction")

For text line extraction, we retrain the CRAFT (Character Region Awareness for Text Detection) on 1000 annotated images provided by Center for Research and Development of Higher Education, The University of Tokyo.
![alt text](https://github.com/ducanh841988/Kindai-OCR/blob/master/images/TextlineRecognition.jpg "text line recognition")

For text line recognition, we employ the attention-based encoder-decoder on our previous publication. We train the text line recognition on 1000 annotated images and 1600 unannotated images provided by Center for Research and Development of Higher Education, The University of Tokyo and National Institute for Japanese Language and Linguistics, respectively.






## Installing Kindai OCR
python==3.7.4   
torch==1.4.0   
torchvision==0.2.1   
opencv-python==3.4.2.17   
scikit-image==0.14.2   
scipy==1.1.0   
Polygon3   
pillow==4.3.0   


## Running Kindai OCR
- You should first download the pre_trained models and put them into ./pretrain/ folder. 
[VGG model](https://drive.google.com/file/d/1_A1dEFKxyiz4Eu1HOCDbjt1OPoEh90qr/view?usp=sharing), [CRAFT model](https://drive.google.com/file/d/1-9xt_jjs4btMrz5wzrU1-kyp2c6etFab/view?usp=sharing), [OCR model](https://drive.google.com/file/d/1mibg7D2D5rvPhhenLeXNilSLMBloiexl/view?usp=sharing) 
- Copy your images into ./data/test/ folder   
- run the following script to recognize images:   
`python test.py`   
- The recognized text transcription is in ./data/result.xml and the result images are in ./data/result/   
- If you may have to check the path to Japanese font in test.py for correct visualization results.   
    `fontPIL = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf' # japanese font`   
- using --cuda = True for GPU device and Fasle for CPU device    
- using --canvas_size ot set image size for text line detection   
 - An example result from our OCR system
 <img src="https://github.com/ducanh841988/Kindai-OCR/blob/master/data/result/res_k188701_021_39.jpg" width="700">
 
 ## Running Kindai OCR
 If you find Kindai OCR useful in your research, please consider citing:   
 Anh Duc Le, Daichi Mochihashi, Katsuya Masuda, Hideki Mima, and Nam Tuan Ly. 2019. Recognition of Japanese historical text lines by an attention-based encoder-decoder and text line generation. In Proceedings of the 5th International Workshop on Historical Document Imaging and Processing (HIP ’19). Association for Computing Machinery, New York, NY, USA, 37–41. DOI:https://doi.org/10.1145/3352631.3352641   
 
 
 ## Acknowledgment

We thank The Center for Research and Development of Higher Education, The University of Tokyo, and National Institute for Japanese Language and Linguistics for providing the kindai datasets.     

## Contact
Dr. Anh Duc Le, email: leducanh841988@gmail.com or anh@ism.ac.jp    


