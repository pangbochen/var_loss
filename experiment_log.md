实验过程中思路的记录

Question:

1.系数的设置

original_loss :  the original loss type, for example, the cross entropy for the classification task.

var_loss : the volatility of the original loss

超参数alpha

loss = (1-alpha) * var_loss + alpha * original_loss

而关于超参数alpha
- 可以设置为固定的值
- 可以随着训练结果发生改变，比如开始时alpha比较大，然后随着batch的增加而逐渐变小，（对应着var_loss的影响逐渐变大）


2.使用课程学习的概念

感谢冠军 童鞋提出的这个有效的思路。

先优化简单的训练目标，之后再优化困难的目标

比如设置一个epoch_threshold
- 再它之前，优化original_loss
- 之后则优化困难的目标，优化original_loss+var_loss，或者直接优化var_loss

关于困难阶段的学习：
- 一种是使用original_loss + var_loss的方法
- 还有就是使用一个逐渐变大的系数alpha来控制模型进化的方向。一种简单的思路是：original_loss+ alpha * var_loss
alpha设计为一个随着时间缓慢增长的系数，从0.01逐渐增加到0.2等等

关于上面“时间”这一个概念的讨论：
- 用训练的epoch来度量这一点
- 用时间端来度量，即不同的时间段使用不同的模型

3.使用其它的multi-task的训练方法

模型也可以看作是多任务学习的一个实例，模型使用多个优化目标来进行。

## Var loss 角度的后记

从股票交易的角度

使用同样的方法，收益，回撤，成本，三者之间是互相矛盾

收益和风险本身也是互相独立的

从这个项目的角度来看：

- original loss是收益性的衡量
- var loss是风险性的衡量

二者本身是相互矛盾的