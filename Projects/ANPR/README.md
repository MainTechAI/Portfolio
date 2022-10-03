# Automatic number-plate recognition (ANPR)

The project was made in 2018. 


* The first stage of the system is the number-plate detection. You can watch a short video here [YouTube](https://www.youtube.com/watch?v=Y9FtcxOLk1M). This step is done by YOLO.
* The second stage is the text detection. This step is done by [AdvancedEAST](https://github.com/huoyijie/AdvancedEAST).
* The third stage is the сharacter recognition. This step is done by [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

___

## Training

#### 1. YOLO

The training/testing data were collected from the internet, some were manually uploaded, 
and some were automatically parsed. In total there are 11100 images of different sizes.

These images were manually labeled using [OpenLabeling](https://github.com/Cartucho/OpenLabeling/) tool.
It was chosen for labeling because it supports Pascal VOC annotation format. 
Then, YOLO was trained using these images.

#### 2. AdvancedEAST
...

#### 3. Tesseract OCR
...