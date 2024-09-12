## 简介
这是SRC的pytorch版本

论文名称：Set-to-Sequence Ranking-Based Concept-Aware Learning Path Recommendation

论文地址：https://ojs.aaai.org/index.php/AAAI/article/view/25630

原仓库地址：https://gitee.com/mindspore/models/tree/master/research/recommend/SRC

## 依赖包版本

- pytorch: 2.2.2
- numpy: 1.24.3
- tqdm:  4.66.2
- scikit-learn: 1.4.2

## 硬件环境
- GPU: 4060 laptop
- memory: 16G

# 注意：
1. 在转换版本时，只进行了基本的替换，模型只有SRC，知识追踪模型只有DKT,就当前的默认环境就可以运行，如果修改情自行查看代码。
2. ~~DKT正常训练，acc和auc指标在0.73左右。SRC训练遇到问题，基本上不收敛，奖励最高只能到-0.32.不清楚是转换过程出现问题，还是代码本身的问题.~~
3. DKT正常训练，SRC正常训练，KT辅助模块还未添加,增加了绘图，训练结束后直接运行dataShow.py查看。（24/9/12）
3. 本代码仅供参考，为后来者提供一条快速验证的捷径，减少版本修改的时间浪费。