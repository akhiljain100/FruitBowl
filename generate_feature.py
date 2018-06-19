
import numpy as np
import tensorflow as tf
from fruits_feature import category_index
import itertools
def load_image_into_numpy_array(image):
  	(im_width, im_height) = image.size
  	return np.array(image.getdata()).reshape(
			(im_height, im_width, 3)).astype(np.uint8)

class fruit_feature(object):
	def __init__(self,model_path):
		self.model_path = model_path
		# Size, in inches, of the output images.
		detection_graph = tf.Graph()
		with detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(model_path, 'rb') as fid:
				serialized_graph = fid.read()

				od_graph_def.ParseFromString(serialized_graph)

				tf.import_graph_def(od_graph_def, name='')
		with detection_graph.as_default():
			
			with tf.Session(graph=detection_graph) as self.sess:
				# Definite input and output Tensors for detection_graph
				self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
				self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				# Each score represent how level of confidence for each of the objects.
				# Score is shown on the result image, together with the class label.
				self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
				self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
				self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	def remove_overlap(self,box):

		to_remove = []
		for x,y in itertools.combinations(box,2):
				if (x == y).all():
					continue
				elif ((abs(x[0] - y[0]) + abs(x[1] - y[1])) < 0.03 ):
					if(x[4]>y[4]):
						to_remove.append(y)
					else:
						to_remove.append(x)
				else:
					continue

		return to_remove

	def detect_image(self,image):
		image_np = np.asarray(image)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		(boxes, scores, classes, num) = self.sess.run(
				[self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
				feed_dict={self.image_tensor: image_np_expanded})
		#print('original',boxes,scores)

		box = []
		for i in range(len(scores[0])):
			if(scores[0][i]> 0.5):
				box.append(np.append(np.append(boxes[0][i],scores[0][i]),i))
		to_remove = self.remove_overlap(box)
		boxes = list(boxes[0])
		scores = list(scores[0])
		classes = list(classes[0])

		for j in to_remove:
			boxes.pop(int(j[5]))
			scores.pop(int(j[5]))
			classes.pop(int(j[5]))
		print(len(boxes))
		#for i in range(iter):
		#	for j in to_remove:
		#		c = list(boxes[0][i])
		#		d = list(j[0:4])
		#		print(c ,d)
		#		if(c == d):
		#			print('check',i)
					#print('here',i,boxes[0][i])
		#			np.delete(boxes, i,0)
		#			np.delete(scores[0], i)
		#			np.delete(classes[0],i)
		#print(len(boxes[0]))
		#for i in range(len(boxes)):
		#	for j in to_remove:
		#		print(j[0:4],boxes[0][i])
		#		if((j[0:4] == boxes[0][i])):
		#			print('check')
		#			boxes.remove(boxes[0][i])
		#			scores.remove(scores[0][i])
		#print(boxes,scores)
		features = []
		center_coord = [0.5,0.5]
		sigma = 0.5
		#print(classes,boxes,scores)
		weight_fruits = []
		for i in range(len(scores)):
			if(scores[i]> 0.5):
				print('Detected Fruit : ',category_index[classes[i]]['name'])
				fruit_coord = [boxes[i][0], boxes[i][1]]
				#print(fruit_coord)
				w=np.exp(-(np.linalg.norm(np.array(center_coord) - np.array(fruit_coord))) / sigma)
				print('weight given to it',w)
				weight_fruits.append(w)
				result = map(lambda x: x * w, category_index[classes[i]]['feature'])
				features.append(list(result))
		if(len(features) == 0):
			return None
		return ((np.sum(features, axis=0) / len(features))/np.sum(weight_fruits))

