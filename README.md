# Kindai-OCR
OCR system for recognizing modern Japanese magazines

## About

This repo contains an OCR sytem for converting modern Japanese images to text.
This is a result of [N2I project](http://codh.rois.ac.jp/collaboration/#n2i) for digitization of modern Japanese documents.

The system has 2 main modules: text line extraction and text line recognition. The overall architechture is shown in the below figure.

For text line extraction, we retrain the CRAFT (Character Region Awareness for Text Detection) on 1000 annotated images provided by Center for Research and Development of Higher Education, The University of Tokyo.
For text line recognition, we employ the attention-based encoder-decoder on our previous publication. We train the text line recognition on 1000 annotated images and 1600 unannotated images provided by Center for Research and Development of Higher Education and National Institute for Japanese Language and Linguistics, respectively.






## Installing Kindai OCR



## Running Kindai OCR


