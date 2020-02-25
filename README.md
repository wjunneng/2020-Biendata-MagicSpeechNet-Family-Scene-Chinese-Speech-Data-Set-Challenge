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