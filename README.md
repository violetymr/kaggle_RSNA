# RSNA Intracranial Hemorrhage Detection

- 最初选题在颅内出血多分类问题和3D点云徘徊，最终选择了简单的图片多分类问题
- 问题Description
    - input:图片zip & csv
    - output:6种分类(如图五种+any)
        - Id - An image Id. 
        - Label - ID 对应的 probability of each type

![images](https://github.com/violetymr/kaggle_RSNA/blob/master/image/classes.png)

- 数据集
    - provided by the RSNA with members of the American Society of Neuroradiology and MD.ai.
    
- 了解CNN架构以及发展历史,参数、feature map等维度计算
    - 发展史： https://blog.csdn.net/u012679707/article/details/80870625
    - pytorch optimizer： https://blog.csdn.net/gdymind/article/details/82708920
    - apex https://blog.csdn.net/c9Yv2cf9I06K2A9E/article/details/100135729
    - 参考60mins pytorch入门 ，lihongyi2019backpropaganda,CNN相关tutorial，相关blog实现模型
    
# baseline

   - 数据预处理：
        - 首先处理csv文件构造train.csv test.csv
        - 165G->14G／3.5G  window resize dataloader构造dataset. ----albumentations库
  
   - model构建：efficientnet pretrain-model 
        - forward y_pred -> loss
        - loss.backward -> update weight(optimizier.step())
        - grad update zero
       
   - train/test 
        epoch | px | loss | result
        |-------|:---:|-----------|-------:|
        7 | 256 | 0.08 | 0.119    有些过拟合
        5 | 256 | 0.09 | 0.118
        2 | 128 | 0.10 | 0.124
        
   - gpu环境运行 
        - cuda:0 device
        - apex加速，实际安装后报gcc链接错误，只能改代码取消apex






