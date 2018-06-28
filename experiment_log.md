实验过程中思路的记录

1.系数的设置

original_loss :  the original loss type, for example, the cross entropy for the classification task.

var_loss : the volatility of the original loss

超参数alpha

loss = (1-alpha) * var_loss + alpha * original_loss

而关于超参数alpha
- 可以设置为固定的值
- 可以随着训练结果发生改变，比如开始时alpha比较大，然后随着batch的增加而逐渐变小，（对应着var_loss的影响逐渐变大）


2.使用课程学习的概念

先优化简单的训练目标，之后再优化困难的目标

比如设置一个epoch_threshold
- 再它之前，优化original_loss
- 之后则优化困难的目标，优化original_loss+var_loss，或者直接优化var_loss