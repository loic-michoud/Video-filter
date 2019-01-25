''' pg_playmp3f.py
play MP3 music files using Python module pygame
pygame is free from: http://www.pygame.org
(does not create a GUI frame in this case)
'''

import pygame as pg
from threading import Thread

class Music(Thread):

    def __init__(self,volume,music):
        Thread.__init__(self)
        self.volume=volume
        self.music=music

    def run(self):
        '''
        stream music with mixer.music module in a blocking manner
        this will stream the sound from disk while playing
        '''
        # set up the mixer
        self.freq = 44100     # audio CD quality
        self.bitsize = -16    # unsigned 16 bit
        self.channels = 2     # 1 is mono, 2 is stereo
        self.buffer = 2048    # number of samples (experiment to get best sound)
        pg.mixer.init(self.freq, self.bitsize, self.channels, self.buffer)
        # volume value 0.0 to 1.0
        pg.mixer.music.set_volume(self.volume)
        clock = pg.time.Clock()
        try:
            pg.mixer.music.load(self.music)
            print("Music file {} loaded!".format(self.music))
        except pg.error:
            print("File {} not found! ({})".format(self.music, pg.get_error()))
            return
        pg.mixer.music.play()

        # while pg.mixer.music.get_busy():
        #     check if playback has finished
        #     clock.tick(30)



    def stop(self):
        pg.mixer.music.set_volume(0)

    def play_music(self):

        pg.mixer.music.rewind()
        pg.mixer.music.set_volume(self.volume)



