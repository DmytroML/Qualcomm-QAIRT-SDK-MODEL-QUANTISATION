from ultralytics.utils.downloads import download
from ultralytics.data.augment import LetterBox
import os
import cv2
import numpy as np

datasets = 'datasets'

class get_calibration_data():
    def __init__(self,imgsz):
        # This automatically downloads and extracts COCO128 into a '../datasets' folder
        self.url = 'https://ultralytics.com/assets/coco128.zip'
        self.imgsz = imgsz
        # This automatically downloads and extracts COCO128 into a '../datasets' folder
        download(self.url, dir=f'./{datasets}/input', unzip=True)
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, scale_fill=False, scaleup=True, stride=32,)
        self.datasets=datasets




# Apply to your image array
#img = np.array(Image.open("your_image.jpg").convert("RGB"))
#processed_img = letterbox(image=img)


    def process(self):

        #python preprocess.py <INPUT PATH> <OUTPUT PATH> 1 0
        if not os.path.exists(f'./{self.datasets}/input'):
            os.makedirs(f'./{self.datasets}/input')
        if not os.path.exists(f'./{self.datasets}/output'):
            os.makedirs(f'./{self.datasets}/output')

        # os.walk returns root path, subfolders, and filenames for every level
        for dirpath, dirnames, filenames in os.walk(f'./{self.datasets}/input'):
            for filename in filenames:
                # Create the full path to the file inside the subfolder
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    input_path = os.path.join(dirpath, filename)


                    output_path = os.path.join(f'./{self.datasets}', 'output', os.path.splitext(filename)[0] + '.raw')
                    output_path_img = os.path.join(f'./{self.datasets}', 'output', os.path.splitext(filename)[0] + '.jpg')

                    img = self.letterbox(image=cv2.imread(input_path))

                    cv2.imwrite(output_path_img, img)
                    # BGR -> RGB
                    img = img[:, :, ::-1]
                    img = np.transpose(img, (2, 0, 1))

                    img = img / 255
                    numpy_img = np.asarray(img).astype(np.float32)

                    #img = img / 255
                    numpy_img = np.asarray(img).astype(np.float32)
                    print (numpy_img.shape)
                    numpy_img.tofile(output_path)


get_calibration_data = get_calibration_data(imgsz=640)
get_calibration_data.process()

'''
def create_file_list(input_dir, output_filename, ext_pattern, print_out=False, rel_path=False):
    input_dir = os.path.abspath(input_dir)
    output_filename = os.path.abspath(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.isdir(input_dir):
        raise RuntimeError('input_dir %s is not a directory' % input_dir)

    if not os.path.isdir(output_dir):
        raise RuntimeError('output_filename %s directory does not exist' % output_dir)

    glob_path = os.path.join(input_dir, ext_pattern)
    file_list = glob.glob(glob_path)

    if rel_path:
        file_list = [os.path.relpath(file_path, output_dir) for file_path in file_list]

    if len(file_list) <= 0:
        if print_out: print('No results with %s' % glob_path)
    else:
        with open(output_filename, 'w') as f:
            f.write('\n'.join(file_list))
            if print_out: print('%s created listing %d files.' % (output_filename, len(file_list)))


'''