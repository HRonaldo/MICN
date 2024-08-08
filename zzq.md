# 我的实验计划和日志

## 输入端的修改计划

### Patch

1. 跑一下PatchTST
2. debug看一下维度信息
3. 迁移到MICN框架下面

### FFT

1. 跑一下Pathformer或者其他用到的模型
2. debug
3. 学一下傅里叶变换
4. 迁移

## 实验日志

### PatchTST的流程

（分解）--- 平稳化 --- 划分patch --- 投影 --- 加入位置编码（x+pos(x)） --- 多层注意力层 --- Flatten --- 逆平稳化     
通道独立性：[bs x nvars x patch_num x d_model] --- [bs * nvars x patch_num x d_model]     
将m个变量的n个batch变成了，m*n个batch的单变量（即共享权重），先合并再进行pos编码    

分解的思想也有，看到patchtst中，还支持先分解，然后season和trend分别用transformer进行预测    

### micn进行修改的主要步骤  

主要在于编码部分   
原来的处理是，m个通道的序列，映射到d_model维度上，这样其实就达到了CD的目的，可以说是相当简单（）   

所以，可以进行的尝试是，先进行patch分解，然后再进行编码，这样就是单变量序列映射到512维  
