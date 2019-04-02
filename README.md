>本项目在Point-generatory论文基础上修改,原始功能为生成文本的summary,修改后用来提取大篇幅评论中关于某主题词的评论,训练数据使用汽车领域,比如输入“外观大气，质量稳定，保值率高,乘坐空间：总体比较宽敞"和"外观",输出:"外观大气，质量稳定".

# dataProcessBefore
先运行data_handler.py,在运行make_datafiles.py,在finish下生成训练数据,将训练数据复制到TypicalOpinions目录下。

# TypicalOpinions
运行run_summarization.py

#测试结果截图
<img src="https://github.com/lingyixia/TypicalOpinions/blob/master/result.jpg" alt="demo2" />
