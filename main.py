import jieba
from wordcloud import wordcloud
import matplotlib.pyplot as plt  # 在任何绘图之前，我们需要一个figure对象，可以理解成我们需要一张画板才能开始绘图
import jieba  # jieba库是中文分词的第三方库（中文文本需要通过分词获得单个的词语）
from wordcloud import \
    WordCloud  # 导入wordcloud库text = open(r'test.txt', "r").read()  # 读入txt文本数据，在字符串前面加上字符r或R之后表示原始字符串，字符串中的任意字符都不再进行转义，后一个r表示“只读”

cut_text = jieba.cut('word.txt')  # 结巴中文分词，生成字符串，默认精确模式，如果不通过分词，无法直接生成正确的中文词云
result = " ".join(cut_text)  #必须给个符号分隔开分词结果来形成字符串,否则不能绘制词云
# join函数的用法：'sep'.join(seq)参数说明：sep：分隔符。可以为空；seq：要连接的元素序列、字符串、元组、字典；即：以sep作为分隔符，将seq所有的元素合并成一个新的字符串# 生成词云图，这里需要注意的是WordCloud默认不支持中文，所以这里需已下载好的中文字库
# 无自定义背景图：需要指定生成词云图的像素大小，默认背景颜色为黑色,统一文字颜色：mode='RGBA'和colormap='pink'
wc = WordCloud(
    # 设置字体，不指定就会出现乱码background_color='white',  # 设置背景色，默认为黑色width=500,  # 设置背景宽height=350,  # 设置背景高max_font_size=50,  # 最大字体min_font_size=10,  # 最小字体mode='RGBA'  # 当参数为“RGBA”并且background_color不为空时，背景为透明
)
wc.generate(result)  # 根据分词后的文本产生词云
wc.to_file(r"wordcloud.png")  # 保存绘制好的词云图
plt.imshow(wc)  # 以图片的形式显示词云
plt.axis("off")  # 关闭图像坐标系，即不显示坐标系
plt.show()
