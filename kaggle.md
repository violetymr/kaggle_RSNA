# RSNA Intracranial Hemorrhage Detection
- 最初选题在颅内出血多分类问题和3D点云徘徊，最终选择了简单的图片多分类问题
- 问题Description
    - input:图片zip & csv
    - output:6种分类(如图五种+any)
        - Id - An image Id. Each Id corresponds to a unique image, and will contain an underscore .
        - Label - The probability of whether that sub-type of hemorrhage (or any hemorrhage in the case of any) exists in the indicated image.

![images](https://github.com/violetymr/kaggle_RSNA/blob/master/inbox_603584_56162e47358efd77010336a373beb0d2_subtypes-of-hemorrhage.png)
- 数据集
    - provided by the Radiological Society of North America (RSNA®) in collaboration with members of the American Society of Neuroradiology and MD.ai.
- 了解CNN架构以及发展历史,参数、feature map等维度计算
- baseline
    - 数据预处理：165G->14G window resize
    - pytorch实现流程：处理csv,dataset/dataloader,model/data->device，efficientnet&apex加速
    -  gpu环境运行 gcc问题






