import csv
import pygame
from pygame import mixer # Load the required library


class Song(object):
    """docstring for ."""
    def __init__(self, filename):
        mixer.init()
        pygame.display.init()
        self._filename = filename

        self._songs = []
        with open(self._filename) as f:
          reader = csv.DictReader(f, delimiter=',')
          header = reader.fieldnames
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
    def play_list(self,sort_index,songs_path):
        screen = pygame.display.set_mode ( ( 420 , 240 ) )
        mixer.music.load ( songs_path+self.get_song_name(sort_index.pop(0)) )  # Get the first track from the playlist

        for index in sort_index:
            mixer.music.queue ( songs_path+self.get_song_name(sort_index.pop(0)) ) # Queue the 2nd song
        #mixer.music.set_endevent ( pygame.NEXTSONG )    # Setup the end track event
        mixer.music.play()           # Play the music

        #running = True
        #while running:
        #    for event in pygame.event.get():
        #        if event.type == pygame.NEXTSONG:    # A track has ended
        #            if len ( sort_index ) > 0:       # If there are more tracks in the queue...
        #                mixer.music.queue ( songs_path+self.get_song_name(sort_index.pop(0)) ) # Queue the next on
    def play_song(self,index,songs_path):

        self._song_name = self.get_song_name(index)

        mixer.music.load(songs_path+self._song_name)
        mixer.music.play()
        print('playing',self._song_name)


    def stop_song(self):
        mixer.music.stop()
