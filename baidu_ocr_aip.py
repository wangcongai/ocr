from aip import AipOcr
from PIL import ImageGrab

# Replace with your own API key and secret
APP_ID = '48308493'
API_KEY = 'QG6S289oPgAjccEktnKTNmqY'
SECRET_KEY = 'F7pL2Vz2T14cvLd2IO5MGrzd2oPhHGR2'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


def baidu_ocr_text(img_p_n):
    # 百度文本识别AipOcr
    image = open(img_p_n, 'rb').read()
    # 识别模式，有好几种，下面有介绍
    # msg = client.basicGeneral(image)
    msg = client.basicAccurate(image)
    # msg = client.webImage(image)
    text = 'result:\n'
    for i in msg.get('words_result'):
        text += (i.get('words') + '\n')
    print(type(text))
    text = text.replace('\u04B0', '').replace('\uFFE5', '').replace('\u00A5', '')
    print(text)


def main():
    # 获取第二个屏幕的坐标
    screen = ImageGrab.grab()
    screens = screen.size
    screen_width = screens[0]
    screen_height = screens[1]
    # bbox = (int(screen_width / 2), 0, screen_width, int(screen_height / 2))
    bbox = (0, 0, screen_width, screen_height)
    # bbox = (0, int(screen_height / 2)+50, screen_width, int(screen_height))

    # 截取第二个屏幕的图像
    image = ImageGrab.grab(bbox=bbox)

    # 将图像转换为灰度图像
    image = image.convert('L')
    image.show()
    img_p_n = './img/test.jpg'
    image.save(img_p_n)
    baidu_ocr_text(img_p_n)


if __name__ == '__main__':
    main()