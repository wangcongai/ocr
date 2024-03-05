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





# 语音识别STT

## 原理

**Speech to Text**，也被称为语音识别，是一种将人类的语音转化为文本的技术。这个过程通常涉及以下几个步骤：

1. **声音信号的采集和预处理**：首先，需要通过麦克风等设备捕获声音信号，并进行噪声消除、静音检测等预处理步骤，以提高语音识别的准确性。
2. **特征提取**：然后，从预处理后的声音信号中提取有用的特征，如梅尔频率倒谱系数（MFCC），这些特征能够有效地表示语音的内容。
3. **声学模型**：接下来，使用声学模型将提取的特征映射到音素或者词的概率分布。这个模型通常是一个深度神经网络，如长短期记忆网络（LSTM）。
4. **解码**：最后，使用解码算法，如维特比算法，从音素或词的概率分布中找出最可能的序列，即识别结果

目前，主流的语音识别模型包括：

- **CTC（Connectionist Temporal Classification）**：CTC是一种用于序列学习任务的损失函数，它可以让循环神经网络（RNN）直接对序列数据进行学习，而无需事先标注好训练数据中输入序列和输出序列的映射关系。
- **Attention-based模型**：这种模型使用一种称为“注意力”的技术来对输入进行加权汇聚。在每个时间步骤上，该模型会根据当前状态和所有输入计算出一个分布式权重向量，并将其应用于所有输入以产生一个加权平均值作为输出。
- **RNN-Transducer**：这个算法结合了编码器-解码器框架和自回归建模思想，在生成目标序列时同时考虑源语言句子和已生成部分目标语言句子之间的交互作用。

详细参考：[目前效果最好、应用较广且比较成熟的语音识别模型是什么？]([目前效果最好、应用较广且比较成熟的语音识别模型是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/349970899))



## 实现方案

登录[百度智能云平台语音技术](https://console.bce.baidu.com/ai/#/ai/speech/overview/index)，领取免费的短语音识别和实时语音服务。短语音免费赠送**15w**次调用。最大**并发度限制为5**。

![image-20240201183703034](https://piclist-1321200338.cos.ap-nanjing.myqcloud.com/image-20240201183703034.png)

需要预先安装`PyAudio`  Python库，它提供了对音频输入和输出的支持。使用`PyAudio`你可以在Python程序中播放和录制音频

核心调用代码：

```python
# 创建AipSpeech对象
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
# 调用百度语音识别API, 语音模型id为1537
result = client.asr(speech, 'wav', RATE, {'dev_pid': 1537})
```



## 双线程实时语音SST

参考：[python百度语音实时识别成文字](https://blog.csdn.net/zcs2632008/article/details/123334807)，但是该脚本是单线程，在录制麦克风讲话的时候，无法进行语音识别。我将代码逻辑进行了改造，采用双线程。一个线程用来录制麦克风讲话，一个线程用来对语音文件进行语音识别。这样可以保证不遗漏麦克风的讲话内容：

```python
import threading
import pyaudio
import os
import time
import wave
from aip import AipSpeech

# 百度AI平台提供的凭证信息
APP_ID = 'your app id'
API_KEY = 'your api key'
SECRET_KEY = 'your secret key'

# 创建AipSpeech对象
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 15


# 语音识别函数
def recognize_speech():
    while True:
        # 等待音频文件生成
        while not os.path.exists('output.wav'):
            time.sleep(0.1)

        # 读取音频数据
        with open('output.wav', 'rb') as f:
            speech = f.read()

        time.sleep(10)
        # 调用百度语音识别API
        result = client.asr(speech, 'wav', RATE, {'dev_pid': 1537})

        # text = ""
        try:
            text = result['result'][0]
            print(text)
        except Exception as e:
            # print(e)
            continue


# 录音函数
def record_audio():
    while True:
        # 创建PyAudio对象
        p = pyaudio.PyAudio()

        # 打开音频流
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        # 录音缓存
        frames = []

        # 录音
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存音频数据
        wf = wave.open('output.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        # 等待一段时间
        time.sleep(0.1)


if __name__ == '__main__':
    # 创建两个线程
    speech_thread = threading.Thread(target=recognize_speech)
    audio_thread = threading.Thread(target=record_audio)

    # 启动线程
    speech_thread.start()
    audio_thread.start()

    # 等待线程结束
    speech_thread.join()
    audio_thread.join()
```



# 后台OCR和STT效果展示

使用Tmux同时展示两个shell窗口。上窗口为实时语音识别；下窗口每次截屏后做OCR

![](https://piclist-1321200338.cos.ap-nanjing.myqcloud.com/gif-snapshot.gif)



# 参考资料

[How Does Optical Character Recognition Work](https://www.baeldung.com/cs/ocr)

[python百度语音实时识别成文字](https://blog.csdn.net/zcs2632008/article/details/123334807)

[目前效果最好、应用较广且比较成熟的语音识别模型是什么？]([目前效果最好、应用较广且比较成熟的语音识别模型是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/349970899))