import os 
import yaml

import cv2
import random

import numpy as np


def n_frames(p):
	cap = cv2.VideoCapture(p)
	n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	cap.release()
	return int(n)
	

def time_data(npy_path):
	data = np.load(npy_path)
	return data

def main():
	base_path = os.getcwd() + '/dataset';
	
	dir_list = os.listdir(base_path)
	
	# print(dir_list)
		
	#assert len(dir_list) == 2 and dir_list[0] == 'touch' and dir_list[1] == 'vision', 'Not the correct base path.'


	#vision_path = base_path + '/vision'
	#touch_path = base_path + '/touch'
	
	#vision_list = os.listdir(vision_path)
	#touch_list = os.listdir(touch_path)
	
	#assert len(vision_list) == len(touch_list), 'vision and touch dirs do not match'
	
	samples = [
		(
		 motion,
		 n_frames(os.path.join(base_path, motion, 'gelsight.mp4')),
   		 n_frames(os.path.join(base_path, motion, 'video.mp4')),
   		 #time_data(os.path.join(base_path, motion, 'time1.npy')),
   		 #time_data(os.path.join(base_path, motion, 'time2.npy')),
		)
	#	(v, rec, len(os.listdir(os.path.join(vision_path, v, rec))))  \
		for motion in dir_list  
	#	for motion in os.listdir(os.path.join(base_path, motion))
	]
	
	n = len(samples)
	
	train_idx = random.sample(range(0, n), int(0.7 * n))
	val_idx = [i for i in range(0, n) if i not in train_idx ]
	
	print(train_idx)
	print(val_idx)
	
	train_samples = [samples[x] for x in train_idx]
	val_samples = [samples[x] for x in val_idx]


	with open('train.yaml', 'w') as ot:
		yaml.dump(train_samples, ot)
	    
	with open('val.yaml', 'w') as ov:
		yaml.dump(val_samples, ov)
	    
	    
	#print('Done. ' + str(len(data)));	



if __name__ == '__main__':
	main()
	

		
