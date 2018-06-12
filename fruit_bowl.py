
import numpy as np
from train_som import SOM
from song_player import Song
from generate_feature import fruit_feature
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2
import os

parser = argparse.ArgumentParser(description="Fruit Bowl plays music")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-songs","--songs_path", "-s",  help="Provide path to the folder of all the songs",default ='Songs_element_nature/' )
parser.add_argument("-features","--features_path","-f",  help="path to csv file having features of all the songs",default = 'emotional_list_songs_new.csv')
parser.add_argument("-p","--play","-p",  help="play a song or a playlist 'play_list'",default = 'play_song')
args = parser.parse_args()

class FruitBowl(object):
    """Initialize the model and train SOM."""

    # Singleton framework for the class, only one player should be initialized
    _shared_state = {}
    def __new__(cls, *args, **kwargs):
        obj = super(FruitBowl, cls).__new__(cls, *args, **kwargs)

        obj.__dict__ = cls._shared_state
        return obj

    def __init__(self, model_path):

        #Initializing the model
        self.model_path = model_path
        self.model = fruit_feature(self.model_path)

        # Objects subject to selection
        self.song = Song(args.features_path)
        self.songs_feature = self.song.get_song_feature()
        self.video_capture = None

        # Train a 5x3 SOM with 400 iterations
        #self.som = SOM(20, 30, 5, 400)
        #self.som.train(self.songs_feature)

        # Get output grid after training, weights of each neuron and their location
        #self.image_grid = np.asarray(self.som.get_centroids())

    def start_camera(self):
        self.capturing = True
        if self.video_capture is None:
            print('lets check')
            self.video_capture = cv2.VideoCapture(1)

        rubbish, self.original_frame = self.video_capture.read()
        self.original_frame = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        plt.imshow(self.original_frame)
        plt.show()
        self.original_detected_feature = fruit_bowl.model.detect_image(self.original_frame)

if __name__ == "__main__":

        fruit_bowl = FruitBowl('Model_Inference/frozen_inference_graph.pb')
        fruit_bowl.start_camera()

        while (fruit_bowl.capturing and fruit_bowl.video_capture.isOpened()):

            ret, frame = fruit_bowl.video_capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame)
            plt.show()
            try:
                diff = cv2.absdiff(frame, fruit_bowl.original_frame)
                mean_diff = float(np.mean(diff))

                #detected_feature= fruit_bowl.model.detect_image(frame)

                print("mean diff", mean_diff)
                if mean_diff < 120:

                    detected_feature= fruit_bowl.model.detect_image(frame)
                    
                    if(np.linalg.norm(fruit_bowl.original_detected_feature-detected_feature)<0.2):
                        print('no change in the environment')

                    else:
                        if (args.play == 'play_song'):
                            #Find index of the song which is closest to the detected feature
                            min_index = min([j for j in range(len(fruit_bowl.songs_feature))],
                                    key=lambda x: np.linalg.norm(fruit_bowl.songs_feature[x]-detected_feature))
                            fruit_bowl.song.play_song(min_index, args.songs_path)

                        else:
                            # Alternatievly for creating the play list, sort the features
                            sorted_songs_list = sorted([j for j in range(len(fruit_bowl.songs_feature))],
                                                       key=lambda x: np.linalg.norm(
                                                           fruit_bowl.songs_feature[x] - detected_feature))
                            fruit_bowl.song.play_list(sorted_songs_list,args.songs_path)

                        if args.verbose:
                            for i in range(len(fruit_bowl.songs_feature)):
                                print('song feature : ',fruit_bowl.songs_feature[i])
                                print('Distance : ',np.linalg.norm(fruit_bowl.songs_feature[i]-detected_feature))
                                print(fruit_bowl.song.get_song_name(min_index if min_index is not None else sorted_songs_list))

            except IOError:
                print('An error occured trying to read the file.')

            except ValueError:
                print('Non-numeric data found in the file.')

            except ImportError:
                print("NO module found")
        fruit_bowl.video_capture.release()