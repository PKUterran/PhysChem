import os
import PIL.Image as img

BOX = (180, 40, 460, 450)


def crop_png(source: str, target: str):
    im = img.open(source)
    im = im.crop(BOX)
    im.save(target)


if __name__ == '__main__':
    if not os.path.exists('crop_png'):
        os.mkdir('crop_png')
    for file in os.listdir('png'):
        if file.endswith('.png'):
            crop_png(f'png/{file}', f'crop_png/{file}')
