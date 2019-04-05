from PIL import Image
import os

def crop_pic_and_save(path, pic_name):
    img = Image.open(path)
    size = img.size
    start_xpos = size[0] * (6 / 16)
    end_xpos = size[0] * (11 / 16)
    start_ypos = size[1] * (3 / 8)
    end_ypos = size[1] * (16 / 16)
    img2 = img.crop((start_xpos, start_ypos, end_xpos, end_ypos))
    img2.save('F:/FacialData/cropped/' + pic_name)

if __name__ == '__main__':
    home_path = 'F:/FacialData/'
    pics = os.listdir(home_path)

    for pic in pics:
        pic_path = home_path + pic
        if os.path.isfile(pic_path) and pic.split('.')[1] == 'jpg' and int(pic.split('-')[0]) in [18, 19]:
            crop_pic_and_save(pic_path, pic)