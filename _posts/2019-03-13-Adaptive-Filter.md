---
layout: post
comments: true
title:  "自适应滤波器"
excerpt: "-"
date:   2019-03-13 12:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---

## 自适应滤波器基础

自适应滤波的基本原理是，设计一个参数可调的滤波器（例如FIR横向滤波器），然后设计自适应算法，根据目的信号和输出信号之间的差值的某种函数关系，最小化这个目标函数以达到误差最小化，调整可调系数。

因为在环境中可以自行调整参数的变化，因此我们说这时一种自适应算法、自适应滤波器。

首先可实现的数字滤波器我们一般认为有两类，即`IIR`滤波器和`FIR`数字滤波器

两类滤波器得名都是因为其结构特点的，由于`IIR`存在较为复杂的反馈结构，使其能够产生无限长的响应，而同理，由于`FIR`滤波器并没有反馈结构，因此并不能产生无限长的响应。这两类滤波器的结构都是固定的，其中的参数也是经过设计得到的

`IIR Filter`
$$
\begin{equation}
\begin{split}
H(z)=\frac{\sum\limits_{k=0}^Mb_kz^{-k}}{1-\sum\limits_{k=1}^Ma_kz^{-k}}
\end{split}
\tag{1}
\end{equation}
$$





## 维纳滤波器

### 线性组合器

### FIR横向滤波器

