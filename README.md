# Aliyun Security Maliware Detection

https://tianchi.aliyun.com/competition/entrance/231694/introduction

## Approach

### scheme1



## Todo

* [ ] Report
* [ ] PPT
* [ ] Plot (data, trend, target)
  * 词云
  * 箱型图
  * 柱状图
  * 散点图
* [ ] 特征重要性(卡方特征、LGB)
* [ ] 天花板
  * [ ] 算法以及数学模型
  * [ ] 领域、行业深入

## Feature

* Statistic Feature
* Model Feature

* v1 (only one feature by TF-IDF)
  * api sorted by tid and index grouped by file_id
* v2
  * tid_count
  * tid_distinct_count
  * api_distinct_count
  * tid_api_count_max
  * tid_api_count_min
  * tid_api_count_mean
  * tid_api_distinct_count_max
  * tid_api_distinct_count_min
  * tid_api_distinct_count_mean
  * 
* v3
  * v1 + v2

## Model

* N-Gram
* TF-IDF
* XGBoost
* NB-LR
* 卡方校验

## Packages

* Numpy
* Pandas
* Scikit-learn
* SciPy

## References

* [阿里云安全恶意程序检测，线上成绩0.443705，受邀分享比赛思路](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.75c867b5H5JMb8&postId=56989)
* [gitHappyboy/ML](https://github.com/gitHappyboy/ML)
* [API based sequence and statistical features in a combined malware detection architecture](http://jst.tsinghuajournals.com/CN/rhhtml/20180510.htm?WebShieldDRSessionVerify=yCHA5I2sqLqmh2INS4AO#)
* [Google Machine Learning Crash Courses](https://developers.google.com/machine-learning/crash-course/ml-intro)
* [XGBoost multiclass_classification demo](https://github.com/dmlc/xgboost/blob/master/demo/multiclass_classification/train.py)
* [TF-IDF与余弦相似性的应用（一）：自动提取关键词](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html)
* [深度学习基础 (九)--Softmax (多分类与评估指标)](https://testerhome.com/topics/11262)
* [使用Random Forest(随机森林)进行多分类和模型调优](http://blog.sciencenet.cn/blog-259145-785585.html)
* [用机器学习进行恶意软件检测——以阿里云恶意软件检测比赛为例](https://xz.aliyun.com/t/3704)
* [python – LightGBM的多类分类](https://codeday.me/bug/20190410/934719.html)
* [LR文本分类](https://www.jianshu.com/p/4f865aeaba44)
* [特征选择方法：卡方检验和信息增益](https://blog.csdn.net/lk7688535/article/details/51322423)
* [第三届阿里云安全算法挑战赛](https://tianchi.aliyun.com/course/video?spm=5176.12586971.1001.139.4ece67b57DXKOo&liveId=24268)
* [one-vs-rest与one-vs-one以及sklearn的实现](https://www.jianshu.com/p/9332fcfbd197)
* [模型融合方法概述](https://tianchi.aliyun.com/course/courseConsole?spm=5176.12282070.0..6d7c2042yXLaMB&courseId=284)
