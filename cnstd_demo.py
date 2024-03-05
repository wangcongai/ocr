from cnstd import CnStd
from PIL import ImageGrab


if __name__ == '__main__':
    # 获取第二个屏幕的坐标
    screen = ImageGrab.grab()
    screens = screen.size
    screen_width = screens[0]
    screen_height = screens[1]
    bbox = (int(screen_width / 2), 0, screen_width, int(screen_height / 2))

    # 截取第二个屏幕的图像
    image = ImageGrab.grab(bbox=bbox)
    #

    # 将图像转换为灰度图像
    image = image.convert('L')
    image.show()

    # 创建文本检测对象
    std = CnStd()

    # 进行文本检测
    result = std.detect(image)

    # 打印检测结果
    print(result)
