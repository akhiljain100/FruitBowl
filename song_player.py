import csv
from pygame import mixer # Load the required library


class Song(object):
    """docstring for ."""
    def __init__(self, filename):
        mixer.init()
        self._filename = filename

        self._songs = []
        with open(self._filename) as f:
          reader = csv.DictReader(f, delimiter=',')
          header = reader.fieldnames

          #a_line_after_header = next(reader)
          # iterate over remaining lines
          for row in reader:

              self._songs.append([float(row['Earth']),float(row['Air']),float(row['Water']),float(row['Fire']),float(row['Metal'])])

    def get_song_feature(self):
        return self._songs
    def get_song_name(self,index):
        with open(self._filename) as f:
          reader = csv.DictReader(f, delimiter=',')
          header = reader.fieldnames

          #a_line_after_header = next(reader)
          # iterate over remaining lines
          for row in reader:
              if(self._songs[index] == [float(row['Earth']),float(row['Air']),float(row['Water']),float(row['Fire']),float(row['Metal'])]):
                  return row['Name of song']

    def play_song(self,index,songs_path):

        self._song_name = self.get_song_name(index)

        mixer.music.load(songs_path+self._song_name)
        mixer.music.play()
        print('playing',self._song_name)


    def stop_song(self):
        mixer.music.stop()
