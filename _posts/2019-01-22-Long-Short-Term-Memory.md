---
layout: post
comments: true
title:  "Long Short Term Memory"
excerpt: "-"
date:   2019-01-22 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
## 长短期记忆

长短期记忆模型是一个**门控RNN**（`Gated RNN`）模型，用于解决RNN中的长期依赖问题。

---

### 短/长期依赖问题

先介绍一下长期依赖问题。

RNN存在的一个问题是，人们希望RNN能够在当前任务中使用先前学习到的经验，例如在理解当前视频帧的时候，充分考虑到前面若干视频帧的内容。如果RNN能够做到这一点，那么RNN就能满足人们的需求。理论上RNN是这样被设计的，那么实际上又如何呢？这要视情况而定了。

#### 短期依赖问题

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RNNShortDependency.png)

有时候人们在解决当前问题的时候只需要考虑近期的经验，例如语言模型，单词的预测

`The clouds are in the _____.`

此时我们不需要长期的经验就可以完成预测，即

`The clouds are in the sky.`

这种情况下一般的充分训练的RNN可以完美地完成任务。

#### 长期依赖问题

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RNNLongDependency.png)

但是如果在下面的问题中

`I grew up in France, ..., I speak fluent ______.`

中间的语境可以充分长，甚至可能出现`English`相关词汇，虽然实际结果应是`French`。这时RNN还能解决问题吗？

考虑到前面说到的RNN，一般在处理问题时所取的窗大小$$m$$都是比较小的，这种前后文问题应当是难以解决的。我们也许可以预测这个空应该是`a kind of language`，但是却不能预测究竟是一个什么语言。实际应用中这样的问题很多，因此vanilla RNN在解决这种问题的时候就显得比较吃力了。

LSTM的模型正是用来解决这个问题的。

---

这里有一张用烂了的图，暂且继续使用一下。如果将RNN的模型绘制如下，

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/SimpleRNN.png)

那么LSTM的模型如下

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTM.png)



![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTMNotation.png)

LSTM的关键是里面新加入的一个参量，$$C_t$$。这个是`cell state`，代表每一个LSTM cell的状态。这个状态突出一点描述起来是这样的

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTMC.png)

看起来有些像传送带，里面只有一些微小的线性相互作用，信息很容易在传送线上流动而不发生较大的改变。

LSTM能够将信息从`cell state`中移除或添加到其中，这个过程是通过门控的，该门控结构如图

<img src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTMGate.png" alt="" data-disqus-identifier="/posts/2015-08-Understanding-LSTMs/disqussion-58">



门是一种选择性通过信息的方法，它们通过一个`Sigmoid`层和一个逐点乘法组成。我们知道`Sigmoid`函数的输出结果是$$(0,1)$$之内的，因此这里被赋予通过信息量的意义，门控信号越小，则允许通过信息越少，反之越大。每个LSTM单元有三个这样的门结构，分别负责不同的功能。以下是LSTM每一个cell的实际工作过程

> 备注一句，目前常用的RNN模型中，很多人都将以下过程写在一起，本文也不例外。这样的操作可以减少参数的使用，从而加速优化。

### 第一步 - 遗忘门

LSTM首先决定要从`cell state`遗忘什么信息，这个是通过遗忘门实现的，遗忘门位于输入的位置，如图

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTMForget.png)

通过图示可以看出，这是由$$h_{t-1}$$和$$x_t$$决定的。在语言模型中，可以列举前面的例子，即根据前面输入的结果去预测下一个单词输出。`cell state`中可能包含当前对象的性别等于输出有关的信息，例如代词的选择。当我们的输入中遇到了新的对象时，我们就有理由忘记旧对象的信息。

### 第二步 - 输入门

此时的LSTM要决定`cell state`中需要保存什么新信息，这个结构是由两部分组成的。第一个结构是**输入门**，是一个`Sigmoid`层；第二个是`tanh`层，产生一个$$\hat C_t$$候选。

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTMInput.png)

可以说，在上述语言模型的例子中，新对象的信息就是通过这个门进行输入的。同样的，这个门需要的输入信息也是$$h_{t-1}$$和$$x_t$$。

### 第三步 - 更新cell状态

此步骤根据以上信息将$$C_{t-1}$$更新为$$C_t$$。

目前我们得到的信息是

$$f_t=Sigmoid\left(W_f\cdot [h_{t-1},x_t]+b_f\right)\tag{1}​$$

$$i_t=Sigmoid\left( W_f\cdot [h_{t-1},x_t]+b_f \right)\tag{2}$$

$$\hat C_t=tanh\left( W_C\cdot [h_{t-1},x_t]+b_C \right)\tag{3}$$

根据上述的三个结果进行参数更新。

对于旧状态保留比例为$$f_t$$，对于新状态的接纳比例是$$i_t$$，则新的状态为

$$C_t=f_t\cdot C_{t-1}+i_t \cdot \hat C_t \tag{4}​$$

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LSTMUpdateC.png)

值得注意的是，由于我们的参数实际都是矩阵（或者说向量），所以可以被忘记的信息很多，被输入的新信息也不仅一条。

### 第四步 - 输出

接下来这一层与vanilla RNN相似，但是会使用前面的计算出的$$C_t​$$进行一定程度的筛选。通过激活函数计算的输出为

$$o_t=Sigmoid\left( W_o[h_{t-1},x_t]+b_o \right) \tag{5}$$

当然这并不是输出，实际输出结果要经过筛选，即

$$h^t = o_t\cdot tanh(C_t)\tag{6}​$$

整个过程中的激活函数，其作用大多是对范围进行规整。

对筛选结果举例，例如对于一个对象，我们下一个输出词可能是一个动词。这时候就需要LSTM决定输出动词的形式，例如单复数。通过筛选有利于我们选择正确的结果。

---

以上实际上是一个较为正常的LSTM，就像前面说的RNN一样，`without any fancy stuff`。实际见于论文与应用的LSTM结构都经历了微小的变构。

### 变构的LSTM



---

`2019-1-22 15:34:41`

看不懂，还是看不懂，啥都看不懂，他们好厉害啊怎么啥都会。

## Reference

[1] [colah.github.io](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)中的博文