## 代码结构说明  
### feature(同名文件夹自己建立)  
处理后特征用于训练  
### result(同名文件夹自己建立)  
用于提交的结果文件  
### src  
代码目录  
Main.py 算法主脚本  

DataAnalysis.py 各个特征处理文件（调用 xx_feature.py）  

click_feature.py  点击特征  
loan_feature.py 贷款特征  
order_feature.py 订单特征  

order_loan_feature.py 订单_贷款交叉特征  
user_loan_feature.py 用户_贷款交叉特征    

util.py 工具脚本  

Stack.py stacking模型融合（这次比赛没用到）