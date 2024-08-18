受到cv中FFC（傅里叶卷积）的启发（https://papers.nips.cc/paper_files/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf）  
改写得到了1d情况下的FFC，核心思想是融合时域和频域的信息

ffc.py 是包含了原来的ffc代码（下成ffc_2d）和我改写的（ffc_1d）
ffc_1d.py 只有1d情况下的ffc代码
ffc_try.py 可以输入一个向量检验一下ffc.py中的ffc_2d和ffc_1d
