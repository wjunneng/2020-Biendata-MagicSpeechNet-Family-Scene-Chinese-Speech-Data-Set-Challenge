# 2020-Biendata-MagicSpeechNet-Family-Scene-Chinese-Speech-Data-Set-Challenge
2020 Biendata MagicSpeechNet 家庭场景中文语音数据集挑战赛

![image](/image/rule_0.png)

![image](/image/rule_1.png)


## 一、解压数据集

>> ls -ltrh
>
>> zip -s 0 magicdata_speech.zip --out single.zip
>
>> unzip single.zip 


    baseline:
    
    1、数据扩增(SpecAugment)
    2、增加参数量扩大模型维度
    3、增加模型迭代次数
    4、结合语言模型进行解码
    5、参数平均
    
    参考：
    https://github.com/ZhengkunTian/OpenTransformer
    https://github.com/espnet/espnet
    


## 分析
    
    1. 训练集只有一种录音设备.
    2. 验证集每段对话有4个通道的同步录音,包括3个远讲通道和1个近讲通道.远讲通道包括由 安卓/iOS 平台,录音笔录制的文件. 
        
    
    
    
    
    
### trick ###
---

### Done ：

| trick | + score |
| :----:| :----: |
| Label Smoothing (无法收敛) | -0.03 | 
| Time mask | - |
| Frequency mask | - |
| Time wrap | - |

### TODO ：

- 一. 噪声增强：就是对一段干净环境下录制的、没有背景噪声的语音,人为混入一定的背景噪声,使训练数据能够覆盖更多种类的噪声；

    
    1. 噪声增强 [信噪比]
    当加入的噪声过强时,识别准确率会下降,而随着信噪比的增加,正确率逐渐上升.当信噪比较为合适时.正确率可以超过Baseline.
    这证明通过噪声增强,模型性能确实可以得到一定的提升.
    
- 二. 音量增强：就是改变一段语音的音量,使训练数据能够覆盖不同的音量；

    
    2. 音量增强 [音量的变动范围, 也就是音量scale值的上下限]
    音量增强也可以带来一定的性能提升.

- 三. 语速增强：就是改变一段语音的语速,使训练数据能够覆盖不同的语速.


    3. 语速增强 [语速变动的范围]
    由于语速的变化会导致较为明显语音改变, 因此我们选取语速变动范围较小, 只选取了0.9倍速/1.1倍速/1.0倍速（原速）
    将原始训练数据被分为三部分,其中一份保持原速,一份变为0.9倍速,一份变为1.1倍速,然后再把三者合并起来作为训练集.
    
-
    
    同时使用多种的增强方式效果可能下降, 潜在原因:
    a. 一种可能是,是由于在真实测试集中,同时改变语音多种属性的情况并不常见,这样做反而使得训练集与测试集不匹配；
    b. 另一种可能是,因为数据增强是利用音频处理手段对数据进行改造,并非真实情景下录音所得,
    因此有可能在处理过程中引入一些畸变,如果同时进行多种数据增强操作,可能使这种畸变叠加.


    A. 注意: 
        加入原始数据是有帮助的.即便原始数据是干净的录音室数据,把原始数据与做完数据增强之后的数据放在一起进行训练,
    也可以提高总体的准确率.
    
    B. 另外: 
        不应该对一份数据进行多种数据增强操作, 而应该把数据分为几部分, 分别做不同种类的数据增强操作.
    
    C. 理想的处理方式:
        保留一份原始数据,然后额外增加一份数据增强的数据；把这份数据增强的数据分成几个部分,每个部分做一种数据增强操作.
    这样一来,总的数据会变成两份.这样既保证了数据的全面性,同时又没有让数据量倍增,在性能和训练开销上达到了比较好的平衡.

- 四. 多通道特征融合

