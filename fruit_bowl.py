
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
    def __init__(self, model_path):

        #Initializing the model
        self.model_path = model_path
        self.model = fruit_feature(self.model_path)

        # Objects subject to selection
        self.song = Song(args.features_path)
        self.songs_feature = self.song.get_song_feature()


        # Train a 5x3 SOM with 400 iterations
        #self.som = SOM(20, 30, 5, 400)
        #self.som.train(self.songs_feature)

        # Get output grid after training, weights of each neuron and their location
        #self.image_grid = np.asarray(self.som.get_centroids())


if __name__ == "__main__":
        fruit_bowl = FruitBowl('Model_Inference/frozen_inference_graph.pb')
        capturing = True
        c = cv2.VideoCapture(1)
        rubbish, original_frame = c.read()
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

        print(c.isOpened())# ret is useless
        previous_frame = np.array([])
        just_snapped = False
        snapshot_flag = False

        while (capturing and c.isOpened()):
            ret, frame = c.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                diff = cv2.absdiff(frame, original_frame)
                mean_diff = float(np.mean(diff))
                detected_feature= fruit_bowl.model.detect_image(frame)
                print("mean diff", mean_diff)
                if mean_diff < 40:
                    new_detected_feature= fruit_bowl.model.detect_image(frame)


                    #Find index of the song which is closest to the detected feature
                    min_index = min([j for j in range(len(fruit_bowl.songs_feature))],
                            key=lambda x: np.linalg.norm(fruit_bowl.songs_feature[x]-detected_feature))
                    # Alternatievly for creating the play list, sort the features
                    sorted_songs_list = sorted([j for j in range(len(fruit_bowl.songs_feature))],key=lambda x: np.linalg.norm(fruit_bowl.songs_feature[x]-detected_feature))

                    if args.verbose:
                        for i in range(len(fruit_bowl.songs_feature)):
                            #print('Mapped neuron : ',fruit_bowl.image_grid[mapped[0], mapped[1],:])
                            print('song feature : ',fruit_bowl.songs_feature[i])
                            print('Distance : ',np.linalg.norm(fruit_bowl.songs_feature[i]-detected_feature))
                            print(fruit_bowl.song.get_song_name(min_index))
                    if (args.play == 'play_song'):
                        fruit_bowl.song.play_song(min_index, args.songs_path)
                    else:
                        fruit_bowl.song.play_list(sorted_songs_list,args.songs_path)
            except IOError:
                print('An error occured trying to read the file.')

            except ValueError:
                print('Non-numeric data found in the file.')

            except ImportError:
                print("NO module found")
