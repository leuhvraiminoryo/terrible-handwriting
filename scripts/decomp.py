# this file's purpose is to correctly decompose a lowercase-full-alphabet long pic into its individual letters
# using pygame.image cuz I know how to use it for such a goal, eh

import pygame, os

pygame.display.init()
pygame.display.set_mode()

low_letters = [chr(ord('a')+i) for i in range(26)]

def decomp(image_path, lastpart, char_size_x=20, char_size_y=20, bg_color=(0,0,0)):
    full_image = pygame.image.load(image_path).convert()
    surf = pygame.Surface((full_image.get_width(),full_image.get_height())).convert()
    surf.fill(bg_color)
    full_image.set_colorkey((0,0,0))
    surf.blit(full_image,(0,0))
    full_image = surf.copy()
    full_image.set_colorkey((255,255,255))
    num = 0
    file = open("data/shite.csv", "a")
    for char in low_letters:
        full_image.set_clip(pygame.Rect(((char_size_x+1)*num),0,char_size_x,char_size_y))
        letter_image = full_image.subsurface(full_image.get_clip())
        name = char + lastpart
        path_name = "data/train_pics/"+name
        pygame.image.save(letter_image, path_name)
        file.write(name+","+str(ord(char)-ord("a"))+"\n")
        num += 1

def load_all_strips(dir_path):

    #TODO : create the directories if they do not exist

    open("data/shite.csv","w").close() # clearing the csv file

    for strip in os.listdir(dir_path):
        decomp(os.path.join(dir_path, strip), strip.split("_")[-1])
