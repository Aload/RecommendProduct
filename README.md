# RecommendProduct
基于ALS的简易推荐系统

### 基本数据请自行在 https://grouplens.org/datasets/movielens/下载

###相关阅读资料请参考 Hadoop+Spark巨量数据分析 

基于spark的mlib库进行构建，基于textFile换算出来的RDD[Rating]进行随机权重分组，然后ALS train训练。
参数训练 我们选取三组基本值来进行分组训练 得到最佳的rank interaction lambda参数值，
然后根据全量训练得到最佳的参数模型。最佳模型止于根据RMSE(均平方根)进行对比得到。RMSE越小说明误差越小，即预测值与真实的值更加契合，
当然也要避免overFitting(过度训练)[训练评估阶段的RMSE很低，测试阶段很高说明过度训练，反之]
###关于spark mlib笔者也是处于学习阶段 如有误 欢迎指正。