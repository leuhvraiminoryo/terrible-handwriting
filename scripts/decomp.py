# this file's purpose is to correctly decompose a lowercase-full-alphabet long pic into its individual letters
# using pygame.image cuz I know how to use it for such a goal, eh

import pygame

low_letters = [chr(ord('a')+i) for i in range(26)]

def decomp(image_path, series_id, char_size_x=20, char_size_y=20, bg_color=(0,0,0)):
    print(low_letters)
    full_image = pygame.image.load(image_path).convert()
    surf = pygame.Surface((full_image.get_width(),full_image.get_height())).convert()
    surf.fill(bg_color)
    full_image.set_colorkey((0,0,0))
    surf.blit(full_image,(0,0))
    full_image = surf.copy()
    full_image.set_colorkey((255,255,255))
    num = 0
    for char in FontOrder:
        full_image.set_clip(pygame.Rect(((TileSize+1)*num),0,char_size_x,char_size_y))
        letter_image = full_image.subsurface(full_image.get_clip())
        # todo : save letter_image with correct "char" + id number name, and add corresponding line to csv
        num += 1

