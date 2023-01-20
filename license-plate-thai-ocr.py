import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn
import pytesseract

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms

from PIL import Image


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = input_dir

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print('Performing OCR...')

		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])
   
			img = Image.open(img_path)
			img = img.convert("L")
			lp_str = pytesseract.image_to_string(img, lang="tha", )
			print(bname, lp_str)
   
			with open('%s/%s_str.txt' % (output_dir, bname),'w') as f:
				f.write(lp_str + '\n')

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
