# this file's purpose is to correctly decompose a lowercase-full-alphabet long pic into its individual letters
# using pygame.image cuz I know how to use it for such a goal, eh

import os, shutil

from PIL import Image

low_letters = [chr(ord('a')+i) for i in range(26)]

def decomp(image_path, lastpart, train=True, char_size_x=20, char_size_y=20):
    full_image = Image.open(image_path)

    num = 0
    file = open("data/"+("train" if train else "test")+".csv", "a")
    for char in low_letters:
        letter_image = full_image.crop((((char_size_x+1)*num),0,((char_size_x+1)*(num+1)-1),char_size_y))
        name = char + lastpart
        path_name = "data/" + ("train_pics/" if train else "test_pics/") + name
        letter_image.save(path_name)
        file.write(name+","+str(ord(char)-ord("a"))+"\n")
        num += 1

def load_all_strips(dir_path):
    for strip in os.listdir(dir_path+"/training"):
        decomp(os.path.join(dir_path, "training", strip), strip.split("_")[-1])
    
    for strip in os.listdir(dir_path+"/testing"):
        decomp(os.path.join(dir_path, "testing", strip), strip.split("_")[-1], False)

def create_data_filesystem():
    try:
        shutil.rmtree("data")
    except Exception as e:
        print(f"could not delete 'data' dir : {e}")

    os.mkdir("data")
    os.mkdir("data/train_pics")
    os.mkdir("data/test_pics")

def treat_data(dir_path):
    create_data_filesystem()

    load_all_strips(dir_path)
