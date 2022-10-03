# Automatic number-plate recognition (ANPR)

The project was made in 2018. 


* The first stage of the system is the number-plate detection. You can watch a short video here [YouTube](https://www.youtube.com/watch?v=Y9FtcxOLk1M). This step is done by YOLO.
* The second stage is the text detection. This step is done by [AdvancedEAST](https://github.com/huoyijie/AdvancedEAST).
* The third stage is the сharacter recognition. This step is done by [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).

___

### Training



The training/testing data were collected from the internet, some were manually uploaded, 
and some were automatically parsed from search engine results. In total there are 
11100 images of different sizes.

These images were manually labeled using [OpenLabeling](https://github.com/Cartucho/OpenLabeling/) tool.
It was chosen for labeling because it supports the required annotation formats (PASCAL VOC, YOLO darknet).