---
layout: post
comments: true
title:  "Vanilla RNN"
excerpt: "-"
date:   2019-01-13 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


---

Vanilla原义是香草，此处含义是`without fancy stuff`，即最原初的版本。

本来想参考GitHub上的某个项目写，结果这个项目的主人本身也不懂什么是vanilla RNN，甚至于Gradient Decent的时候求导计算都是错的。下面还是通过简单计算推导来说明一下Vanilla RNN。

首先是RNN的最基础模型。RNN是Recurrent Neural Network，国内说法很多，有的地方翻译为循环神经网络，有的地方翻译为递归神经网络。这两个翻译个人认为均不准确。RNN是一种结构稍显特殊的网络，相比传统的深度网络和卷积网络这样的结构，RNN的结构并不是简单的层状结构，而是一种类似循环的结构，其基础结构如下

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RNN1.png)

左侧是折叠后的网络结构，右侧是展开后的结构。这个结构是目前来讲最为基础的一种。如图，下方的$$x^t$$表示的是该时刻的输入，这个输入往往是一个向量，因此写成小写的向量形式；中间始终向右传递的$$h^t$$是`hidden state`，隐藏状态，这个值在不同的`RNN Cell`中不断传递并更新。每一个Cell的计算结果经过一个**激活函数**后将得到一个输出，这个输出与我们期望的输出进行比较，用`Loss function`来描述两者的**误差**。本文中可能将误差描述为损失或错误(**E**rror)。

图中的几个加粗大写字母是矩阵，但是使用这些矩阵难免造成混淆，因此此处对矩阵进行重命名。输入后的矩阵命名为$$W_{hx}$$，状态间转化矩阵命名为$$W_{hh}$$，输出矩阵命名为$$W_{hy}$$。

因为没有抽到阿比所以所以不是很想学习

他妈的心情好了再写