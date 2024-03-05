from PIL import ImageGrab
from cnocr import CnOcr


if __name__ == '__main__':
    # 获取第二个屏幕的坐标
    screen = ImageGrab.grab()
    screens = screen.size
    screen_width = screens[0]
    screen_height = screens[1]
    bbox = (int(screen_width/2), 0, screen_width, int(screen_height/2)-10)

    # 截取第二个屏幕的图像
    image = ImageGrab.grab(bbox=bbox)
    #

    # 将图像转换为灰度图像
    image = image.convert('L')
    image.show()
    # 创建 OCR 对象
    ocr = CnOcr()
    text = ocr.ocr(image)

    # 打印OCR结果
    print(text)