---
layout: post
date: 2019-6-27 14:20:06
title: "Compressed Sensing"
author: Shicong Liu
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

很久没有打理过博客了，主要经历都放在了[bitlecture](bitlecture.github.io)项目中了，今天开始重新来写写文章。

[TOC]

## 背景

信号处理中，我们往往是将信号通过`ADC`后做处理。这个过程需要`ADC`的高速采样，因为目前已有较多的信号频段逐渐靠近`6GHz`，需要更高的采样速率，这将增大`ADC`的功耗，也会带来处理与存储上的困难。

但是实际上我们处理的许多通信信号都是稀疏的，或者说在某个域上是稀疏的（这也将导致奈奎斯特采样速率下得到的信号存在大量的冗余），我们在恢复信号的时候就可以不必受限于奈奎斯特采样速率，从而在更少的采样点数中恢复信号。摆脱这一限制的方法就是压缩感知方法，可以通过远小于奈奎斯特采样率的速率通过非线性重建算法较完美地重建原有信号。

一个典型的压缩感知问题如下，假设我们的信号$$x$$通过观测矩阵$$h$$后得到接收结果$$y$$，其表示为矩阵如下
$$
\begin{equation}
\begin{split}
\left[\begin{matrix}y_1\\y_2\\...\\...\\y_M\end{matrix}\right]=\left[\begin{matrix}\\ \\ h_1\:h_2\:...\:h_N\\ \\ \\\end{matrix}\right]\left[\begin{matrix}x_1\\x_2\\...\\...\\x_N\end{matrix}\right]
\end{split}
\tag{1}
\end{equation}
$$

其中我将观测矩阵写成了列向量的形式，直观些的写法还可以画成图像如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/cs1.png" style="display: inline-block;" width="200"/>
</div>


