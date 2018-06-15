
import numpy as np
import cv2
from song_player import Song
import argparse
import webcam_video_stream as ws
import time
from generate_feature import fruit_feature

parser = argparse.ArgumentParser(description="Fruit Bowl plays music")

group = parser.add_mutually_exclusive_group()

group.add_argument("-v", "--verbose", action="store_true")

parser.add_argument("-songs","--songs_path", "-s",\
                    help="Provide path to the folder of all the songs",\
                    default ='Songs_element_nature/' )
parser.add_argument("-features","--features_path","-f", \
                    help="path to csv file having features of all the songs",\
                    default = 'emotional_list_songs_new.csv')
parser.add_argument("-p","--play","-p", \
                    help="play a song or a playlist 'play_list'",\
                    default = 'play_song')

args = parser.parse_args()

class FruitBowl(object):
    """Initialize the model, Singleton framework for the class, only one player should be initialized"""
    _shared_state = {}
    def __new__(cls, *args, **kwargs):
        obj = super(FruitBowl, cls).__new__(cls, *args, **kwargs)

        obj.__dict__ = cls._shared_state
        return obj

    def __init__(self, model_path,frame):

        #Initializing the model
        self.model_path = model_path
        self.model = fruit_feature(self.model_path)
        self.original_frame = frame
        # Objects subject to selection
        self.features_path = args.features_path
        self.song = Song(args.features_path)
        self.songs_feature = self.song.get_song_feature()

    def cal_initial_feature(self):
        self.original_detected_feature = self.model.detect_image(self.original_frame)

if __name__ == "__main__":
    detection = True
    min_index = None
    sorted_songs_list = None
    last_min_index = None
    read_image = None
    get_feature=None

    try:

        read_image = ws.WebcamVideoStream().start()
        frame = read_image.read()
        fruit_bowl = FruitBowl('Model_Inference/frozen_inference_graph.pb',frame)
        fruit_bowl.cal_initial_feature()
        get_feature = ws.detect_fruit(fruit_bowl.model_path,read_image,fruit_bowl.features_path).start()

        while (detection == True):


                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #    print('pressed q')
                        time.sleep(2)
                        frame = read_image.read()
                        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                        detected_feature = get_feature.read_feature()
                        #detected_feature = None
                        if(detected_feature is None):
                            print('Initial environtment should be changed for playing the song')


                        elif(np.linalg.norm(fruit_bowl.original_detected_feature-detected_feature)<0.2):
                            print('no change in the environment')

                        else:
                            if (args.play == 'play_song'):
                                # Find index of the song which is closest to the detected feature
                                min_index = min([j for j in range(len(fruit_bowl.songs_feature))],
                                                key=lambda x: np.linalg.norm(
                                                    fruit_bowl.songs_feature[x] - detected_feature))
                                if last_min_index is None:
                                    last_min_index = min_index
                                    fruit_bowl.song.play_song(min_index, args.songs_path)
                                elif last_min_index == min_index:
                                    print('no change in the environment')
                                else:
                                    last_min_index = min_index
                                    fruit_bowl.song.play_song(min_index, args.songs_path)
                                time.sleep(5)

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
        print("No module found")
    except KeyboardInterrupt:
        print('Keyboard interruption')
        detection =  False
        if read_image is not None:
            read_image.stop()
        if get_feature is not None:
            get_feature.stop()
