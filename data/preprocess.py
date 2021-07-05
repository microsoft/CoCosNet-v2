import os
import skimage.util as util
from skimage import io
from skimage.transform import resize


with open('train.txt', 'r') as fd:
    image_files = fd.readlines()

total = len(image_files)
cnt = 0

# path/to/deepfashion directory
root = '/path/to/deepfashion'
# path/to/save directory
save_root = 'path/to/save'

for image_file in image_files:
    image_file = os.path.join(root, image_file).strip()
    image = io.imread(image_file)
    pad_width_1 = (1101-750) // 2
    pad_width_2 = (1101-750) // 2 + 1
    image_pad = util.pad(image, ((0,0),(pad_width_1, pad_width_2),(0,0)), constant_values=232)
    image_resize = resize(image_pad, (1024, 1024))
    image_resize = (image_resize * 255).astype('uint8')
    dst_file = os.path.dirname(image_file).replace(root, save_root)
    os.makedirs(dst_file, exist_ok=True)
    dst_file = os.path.join(dst_file, os.path.basename(image_file))
    # dst_file = dst_file.replace('.jpg', '.png')
    io.imsave(dst_file, image_resize)
    cnt += 1
    if cnt % 20 == 0:
        print('Processing: %d / %d' % (cnt, total))
