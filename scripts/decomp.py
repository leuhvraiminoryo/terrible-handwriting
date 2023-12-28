# this file's purpose is to correctly decompose a lowercase-full-alphabet long pic into its individual letters
# using pygame.image cuz I know how to use it for such a goal, eh

import os

from PIL import Image

low_letters = [chr(ord('a')+i) for i in range(26)]

def decomp(image_path, lastpart, char_size_x=20, char_size_y=20):
    full_image = Image.open(image_path)

    num = 0
    file = open("data/shite.csv", "a")
    for char in low_letters:
        letter_image = full_image.crop((((char_size_x+1)*num),0,((char_size_x+1)*(num+1)-1),char_size_y))
        name = char + lastpart
        path_name = "data/train_pics/"+name
        letter_image.save(path_name)
        file.write(name+","+str(ord(char)-ord("a"))+"\n")
        num += 1

def load_all_strips(dir_path):

    #TODO : create the directories if they do not exist

    open("data/shite.csv","w").close() # clearing the csv file

    for strip in os.listdir(dir_path):
        decomp(os.path.join(dir_path, strip), strip.split("_")[-1])
