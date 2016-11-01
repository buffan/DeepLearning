# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:03:33 2016

@author: Andrew
"""

import os
from sys import platform
    
    
def remove_spaces(music_dir):
    for part_dir in os.listdir(music_dir):
        if part_dir != 'Mixed':
            for song in os.listdir(os.path.join(music_dir, part_dir)):
                song_path = os.path.join(os.path.join(music_dir, part_dir), song)
                os.rename(song_path, song_path.replace(' ', ''))
                
                
if platform == 'linux' or platform == 'linux2':
    music_dir = r'./Music'
    remove_spaces(music_dir)
elif platform == 'win32':
    music_dir = r'.\Music'
    remove_spaces(music_dir)
else:
    print('Error: invalid OS')