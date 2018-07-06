实验过程中思路的记录

1.系数的设置

original_loss :  the original loss type, for example, the cross entropy for the classification task.

var_loss : the volatility of the original loss

超参数alpha

loss = (1-alpha) * var_loss + alpha * original_loss

而关于超参数alpha
- 可以设置为固定的值
- 可以随着训练结果发生改变，比如开始时alpha比较大，然后随着batch的增加而逐渐变小，（对应着var_loss的影响逐渐变大）
- 另外一个comment是：之前做过这件事情，从整体的角度来看，收益性的目标和风险性的目标无法同时被优化，但是具体某一段时间内，同时优化目标在一定程度上是可行的。
- 也即对不同时间段采用不同的模型来进行训练,而如何衡量不同的时间段，这是一个新的问题
    - 使用attention机制进行训练，
    - 使用动态multi-task的方法进行动态调整权重系数

2.使用课程学习的概念

一个很有趣的概念，动态对模型进行调整。

先优化简单的训练目标，之后再优化困难的目标

比如设置一个epoch_threshold
- 再它之前，优化original_loss
- 之后则优化困难的目标，优化original_loss+var_loss，或者直接优化var_loss