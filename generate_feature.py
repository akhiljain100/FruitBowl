
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from fruits_feature import category_index

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




	def detect_image(self,image_path):
		image = Image.open(image_path)
		im_width, im_height = image.size
		# the array based representation of the image will be used later in order to prepare the
		# result image with boxes and labels on it.
		image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		(boxes, scores, classes, num) = self.sess.run(
				[self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
				feed_dict={self.image_tensor: image_np_expanded})


		features = []
		for i in range(len(scores[0])):
			if(scores[0][i]> 0.5):
				print('Detected Fruit : ',category_index[classes[0][i]]['name'])
				features.append(category_index[classes[0][i]]['feature'])
		return np.sum(features, axis=0)/len(features)








