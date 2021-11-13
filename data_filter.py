import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from tqdm import tqdm

def count_pixels_in_image(image, color):
    return np.sum(np.all(image == color, axis=2))

depth10k_segs = './depth10k/segs'
depth10k_imgs = './depth10k/imgs'

label_map = np.array([[32, 64, 32], [0, 0, 255], [96, 128, 128], [102, 255, 0], [255, 0, 204]])
distributions = []
max_props = []
high_prop_imgs = []
high_prop_thresh = 0.8

imgs_and_segs = list(zip(os.listdir(depth10k_imgs), os.listdir(depth10k_segs)))
for img, seg in tqdm(imgs_and_segs):
    segment = cv2.imread(os.path.join(depth10k_segs, seg))
    half_segment = segment[segment.shape[0]//2:, :, :]

    num_pixels = half_segment.shape[0] * half_segment.shape[1]

    road_amnt = count_pixels_in_image(half_segment, label_map[0]) / num_pixels
    lane_amnt = count_pixels_in_image(half_segment, label_map[1]) / num_pixels
    undrivable_amnt = count_pixels_in_image(half_segment, label_map[2]) / num_pixels
    movable_amnt = count_pixels_in_image(half_segment, label_map[3]) / num_pixels
    car_amnt = count_pixels_in_image(half_segment, label_map[4]) / num_pixels
    #print(road_amnt, lane_amnt, undrivable_amnt, movable_amnt, car_amnt)

    distributions.append([road_amnt, lane_amnt, undrivable_amnt, movable_amnt, car_amnt])
    max_prop = max(road_amnt, lane_amnt, undrivable_amnt, movable_amnt, car_amnt)
    max_props.append(max_prop)
    if max_prop > high_prop_thresh:
        image = cv2.imread(os.path.join(depth10k_imgs, img))
        high_prop_imgs.append((image, img, segment, seg))

print(f'Bad Images: {len(high_prop_imgs)}/{len(imgs_and_segs)}')

distributions = np.array(distributions)[:1000]

fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].set_title('Proportions of SegNet Outputs')
axs[0].plot(distributions[:,0])
axs[0].plot(distributions[:,1])
axs[0].plot(distributions[:,2])
axs[0].plot(distributions[:,3])
axs[0].plot(distributions[:,4])
axs[0].legend(['Road', 'Lane', 'Undrivable', 'Movable', 'Car'])

max_props = np.array(max_props)[:1000]
axs[1].set_title('Maximum Proportions')
axs[1].bar([i for i in range(len(max_props))], max_props)

fig.savefig('distributions.png')

bad_images = open("bad_images.txt", "w")
os.makedirs('./depth10k/bad_imgs')
os.makedirs('./depth10k/bad_segs')
for img, img_name, seg, seg_name in high_prop_imgs:
    bad_images.write(img_name + '\n')
    shutil.move(os.path.join(depth10k_imgs, img_name), os.path.join('./depth10k/bad_imgs', img_name))
    shutil.move(os.path.join(depth10k_segs, seg_name), os.path.join('./depth10k/bad_segs', seg_name))

    '''
    concat = cv2.vconcat([img, seg])
    cv2.imshow('bad', concat)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    '''
    
bad_images.close()
