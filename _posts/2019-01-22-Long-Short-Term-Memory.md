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
---

## Long Short Term Memory

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



---

`2019-1-22 15:34:41`

看不懂，还是看不懂，啥都看不懂，他们好厉害啊怎么啥都会。

## Reference

