import cv2
import pytesseract
import numpy as np

from tesserocr import PyTessBaseAPI
from PIL import Image

import easyocr


def main():
    
    image_name = 'test_5'
    
    img = cv2.imread(f"{image_name}.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # get only black text
    # img_gray = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY )[1]
    
    cv2.imwrite(f"{image_name}_gray.png", img_gray)
    
    # cv2.imshow("img", img)
    # cv2.imshow("img_gray", img_gray)
    # cv2.waitKey(0)
    
    reader = easyocr.Reader(['th'], )
    result = reader.readtext(f"{image_name}_gray.png", output_format="dict", min_size=50)
    
    print(result)


if __name__ == "__main__":
    main()
