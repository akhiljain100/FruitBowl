
import numpy as np
from train_som import SOM
from song_player import Song
from generate_feature import fruit_feature
import argparse

parser = argparse.ArgumentParser(description="Fruit Bowl plays music")
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-songs","--songs_path", "-s",  help="Provide path to the folder of all the songs",default ='Songs_element_nature/' )
parser.add_argument("-features","--features_path","-f",  help="path to csv file having features of all the songs",default = 'emotional_list_songs_new.csv')
args = parser.parse_args()

class FruitBowl(object):
    """Initialize the model and train SOM."""
    def __init__(self, model_path):

        #Initializing the model
        self.model_path = model_path
        self.model = fruit_feature(self.model_path)

        #Training SOM according to the song features
        self.song = Song(args.features_path)
        self.songs_feature = self.song.get_song_feature()


        # Train a 5x3 SOM with 400 iterations
        self.som = SOM(20, 30, 5, 400)
        self.som.train(self.songs_feature)

        # Get output grid after training, weights of each neuron and their location
        self.image_grid = np.asarray(self.som.get_centroids())


if __name__ == "__main__":
    fruit_bowl = FruitBowl('Model_Inference/frozen_inference_graph.pb')
    for i in range(10):
        image_path = 'Images/image'+str(i)+'.jpg'
        detected_feature= fruit_bowl.model.detect_image(image_path)

        #Map fruits to their closest neurons
        mapped = fruit_bowl.som.map_vects(detected_feature)

        #Find index of the song which is closest to the detected neuron
        min_index = min([j for j in range(len(fruit_bowl.songs_feature))],
                            key=lambda x: np.linalg.norm(fruit_bowl.songs_feature[x]-
                                                         fruit_bowl.image_grid[mapped[0], mapped[1],:]))
        if args.verbose:
            for i in range(len(fruit_bowl.songs_feature)):
                print('Mapped neuron : ',fruit_bowl.image_grid[mapped[0], mapped[1],:])
                print('song feature : ',fruit_bowl.songs_feature[i])
                print('Distance : ',np.linalg.norm(fruit_bowl.songs_feature[i]-
                                                             fruit_bowl.image_grid[mapped[0], mapped[1],:]))
                print(fruit_bowl.song.get_song_name(min_index))
        fruit_bowl.song.play_song(min_index,args.songs_path)