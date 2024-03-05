import pytesseract
from PIL import ImageGrab

# pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


if __name__ == '__main__':
    # 获取第二个屏幕的坐标
    screen = ImageGrab.grab()
    screens = screen.size
    screen_width = screens[0]
    screen_height = screens[1]
    bbox = (int(screen_width / 2), 0, screen_width, int(screen_height / 2))

    # 截取第二个屏幕的图像
    image = ImageGrab.grab(bbox=bbox)

    # 将图像转换为灰度图像
    image = image.convert('L')
    image.show()
    # 使用Tesseract OCR引擎进行OCR
    text = pytesseract.image_to_string(image, lang='chi_sim+eng')

    # 打印OCR结果
    print(text)
