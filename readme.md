# 验证码深度学习
10w训练，1k验证数据，100次迭代，6个小时，最后准确率大概60%，网络结构有待优化。
## 环境
pytorch+cuda+python(3.7+)

## 生成验证码
generate_captcha.py
captcha_num=100000  #100000改为自己想要的数量
captcha_dir,captcha_file=get_captcha_dir(captcha_type[0]) #0表示类型，有'train','valid','test','predict'
会在项目的同级目录下生成captcha_data目录（为什么不是在项目里面，是因为生成文件大多，ide每次打开加载文件很耗时）

## 训练
train.py

## 测试
test.py

## 预测
predict.py

## 配置文件
在util/common.py里面