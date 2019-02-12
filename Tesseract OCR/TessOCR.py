
# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import pandas as pd
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe' 

#image location to be OCR
img = 'img to OCR'
# load the example image and convert it to grayscale
image = cv2.imread(img)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#add preprocessing technique
preprocessing = "thresh" 
# check to see if we should apply thresholding to preprocess the
# image
if preprocessing == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
# make a check to see if median blurring should be done to remove
# noise
elif preprocessing == "blur":
	gray = cv2.medianBlur(gray, 3)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
data = pytesseract.image_to_data(Image.open(filename))
pdf = pytesseract.image_to_pdf_or_hocr(Image.open(filename), extension='pdf')
os.remove(filename)
file = open('textfile.txt','w') 
file.write(str(text.encode("utf-8"))) 
file.close()

file = open('datafile.txt','w') 
file.write(data) 
file.close()

file = open('pdffile.pdf','w') 
file.write(str(pdf)) 
file.close()
 
# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
