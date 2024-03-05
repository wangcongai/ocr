# 文字识别OCR

## 原理

光学字符识别，全称**Optical Character Recognition（OCR）**，技术原理主要基于图像处理和机器学习算法。首先，通过光学方式将纸质文档上的印刷体字符转换成黑白点阵图像文件。然后，利用图像处理技术，如二值化、去噪、分割等，将图像中的字符分离出来。接下来，OCR系统通过特征提取和匹配，对分离出的字符进行识别。OCR技术的实现，总体上可以分为五步：**预处理图片、切割字符、识别字符、恢复版面、后处理文字**。详细介绍可以参考：[How Does Optical Character Recognition Work](https://www.baeldung.com/cs/ocr)

![image-20240131155301352](https://piclist-1321200338.cos.ap-nanjing.myqcloud.com/image-20240131155301352.png)传统的识别字符模型有hidden Markov models ([HMMs](https://www.baeldung.com/cs/hidden-markov-model))和Support Vector Machines ([SVMs](https://www.baeldung.com/cs/naive-bayes-vs-svm)) ，现在则主要使用深度学习模型。





## 本地实现方案

常用的OCR识别模块可以选择：Transym, Tesseract, ABBYY, Prime, Azure 。下面介绍本地安装[Pytesseract](https://pypi.org/project/pytesseract/)，识别中英文图片。

Pytesseract是一个Python的OCR工具，底层使用的是Google的Tesseract-OCR引擎，支持识别图片中的文字，支持jpeg, png, gif, bmp, tiff等图片格式。它可以识别和“读取”图像中嵌入的文本。

从4.0版本开始，Tesseract引入了基于长短期记忆（LSTM）的深度学习模型来进行字符识别。LSTM是一种特殊的递归神经网络（RNN），它可以学习长期依赖性，这对于字符识别任务来说非常有用。这种方法使得Tesseract在处理连续的文本，尤其是手写文本时，具有更高的准确率。

**安装命令**：

```shell
# 先在macOS系统里安装tesseract
brew install tesseract
# 安装中文语言包
brew install tesseract-lang

# python interpreter里安装
pip install pytesseract
pip install pillow
```

**OCR截图脚本**：

```python
import pytesseract
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

    # 将图像转换为灰度图像
    image = image.convert('L')
    image.show()
    # 使用Tesseract OCR引擎进行OCR
    text = pytesseract.image_to_string(image, lang='chi_sim+eng')

    # 打印OCR结果
    print(text)
```



## 百度API实现方案

**baidu-aip**是百度AI开放平台的Python SDK，提供了一系列的人工智能技术，如语音识别、图像识别、自然语言处理等。

登录[百度智能云平台OCR技术](https://console.bce.baidu.com/ai/#/ai/ocr/overview/index)，领取免费的通用文字识别标准版和高精度版。一个月可以免费调用1000次，并发度最大为2。

调用脚本里，用于鉴权的3个变量为：`APP_ID`、`API_KEY`和`SECRET_KEY`。它们是在百度AI开放平台创建应用后，系统分配给用户的，用于标识用户，进行签名验证

![image-20240201164249287](https://piclist-1321200338.cos.ap-nanjing.myqcloud.com/image-20240201164249287-20240201164339108.png)



**安装百度接口**

```shell
pip install baidu-aip
# chardet用于检测字符串的编码格式
pip install chardet
```

**调用脚本**

```python
from aip import AipOcr
from PIL import ImageGrab

# Replace with your own API key and secret
APP_ID = 'your app id'
API_KEY = 'your api key'
SECRET_KEY = 'your secret key'

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
    bbox = (0, int(screen_height / 2)+50, screen_width, int(screen_height))

    # 截取第二个屏幕的图像
    image = ImageGrab.grab(bbox=bbox)

    # 将图像转换为灰度图像
    image = image.convert('L')
    # image.show()
    img_p_n = './img/test.jpg'
    image.save(img_p_n)
    baidu_ocr_text(img_p_n)


if __name__ == '__main__':
    main()
```