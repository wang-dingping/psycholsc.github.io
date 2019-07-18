---
layout: post
date: 2019-6-27 14:20:06
title: "Compressed Sensing"
author: Shicong Liu
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

很久没有打理过博客了，主要精力都放在了[bitlecture](bitlecture.github.io)项目中了，今天开始重新来写写文章。

[TOC]



<div style="text-align:center" title="fig.1"><img alt="fig.1" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/cstitle.png" style="display: inline-block;" width="600" />
<p>Fig.0 CS Abstract </p>
</div>

*总体来说我个人还是感觉这不是什么复杂技术的啦*

## Notation

### 厄米特转置

一个矩阵$$A$$，我有时会直接写作$$\boldsymbol A$$，其转置为$$\boldsymbol A^T$$，但是对于复数矩阵往往需要厄米特转置，即转置的同时进行共轭处理$$\boldsymbol A^H$$。

### argmax

取最大值时变量在某范围内的对应值。相比$$\max$$，这个函数求得是自变量的取值。

### p范数

这个问题说过很多遍了，对于`p`范数而言

## 背景&概述

直到`2004`年压缩感知技术的正式提出之前，奈奎斯特采样定理始终统治着信号处理。即使有的信号没有明显的带宽，其采样也要遵循着时空间分辨率的限制，这实际上也是隐性的应用了那奎斯特采样定理，因为往往需要低通滤波器进行抗混叠。因此在使用`ADC`时，典型的场景就是利用高于奈奎斯特采样频率的速率进行均匀采样。

这个过程需要`ADC`的高速采样，但因为目前已有较多的信号频段逐渐靠近`6GHz`，因此需要更高的采样速率，这将增大`ADC`的功耗，也会带来处理与存储上的困难。

实际上我们处理的许多通信信号都是稀疏的，或者说在某个域上是稀疏的（这也将导致奈奎斯特采样速率下得到的信号存在大量的冗余），我们在恢复信号的时候就可以不必受限于奈奎斯特采样速率，从而在更少的采样点数中恢复信号。摆脱这一限制的方法就是压缩感知方法，可以通过远小于奈奎斯特采样率的速率通过非线性重建算法较完美地重建原有信号。

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

<div style="text-align:center" title="fig.1"><img alt="fig.1" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/cs1.png" style="display: inline-block;" width="250" />
<p>Fig.1 压缩感知信号模型 </p>
</div>


如果$$x$$是一个通常意义上的信号，直观上不存在稀疏性，则在线性代数的知识中我们知道，通过$$y$$和$$D$$我们不能重建出信号$$x$$，因为对于$$M>N$$个未知参量，我们只做了$$N$$次线性采样，一定无法计算出正确的$$x$$。如果此时我们的方程式过定的，即$$M>N$$，这样我们其实实现了过采样，通过最小二乘方法是可以获得近似解的。注意如果是$$M=N$$的饱和采样，我们通过直接求逆的方法可以直接求得完美解，欠定的情况虽然也能获取一个最小二乘表达式，但是仍然不能获取正确的解（毕竟欠采样信息量丢的一塌糊涂），由于自由度没有被完全约束，实际上可以有无穷多组解都满足条件。

那么问题来了，如果我采用欠采样，该如何重建原信号呢？

如果信号是稀疏的，那么在上述欠定采样中，我们实际上还是保留了恢复$$x$$的绝大部分信息量的。压缩感知的目的实际是在欠定采样的条件下计算原方程的稀疏解，在稀疏性尽可能大的条件下寻找最有可能的解。因此原问题等效为一个最优化问题
$$
\begin{equation}
\begin{split}
s^*=argmin\left\| s \right\|_0\\
s.t.\:y=Hs
\end{split}
\tag{2}
\end{equation}
$$
但是对`L0`范数的优化问题一般是`NP`问题，求解过程往往需要穷举法计算，因此在一定条件下将这种优化问题（在某些条件下）等效为`L1`范数的最优化问题
$$
\begin{equation}
\begin{split}
s^*=argmin\left\| s \right\|_1\\
s.t.\:y=Hs
\end{split}
\tag{3}
\end{equation}
$$
实际计算中，一般采用匹配追踪等算法求得次优解。

---

压缩感知相关技术已存在很久，算法在`2004`年被几位数学家再次发展，从而有了更多应用。在此之前首先介绍一些其他的相关内容

## 稀疏性定义

压缩感知进行信号重建的一个前提就是，该信号本身（或者在某个变换域内）是稀疏的。例如复杂的正弦叠加在傅里叶变换域中是稀疏的，自然图像信号在离散余弦变换域中是稀疏的。**稀疏度**这个性质的定义一般是
$$
\begin{equation}
\begin{split}
\left\| s \right\|_0=\{i:s_i\neq 0\}
\end{split}
\tag{4}
\end{equation}
$$
的个数，即信号$$x$$中非零项的个数。这个数字越小，我们认为这个信号就越**稀疏**（在变换域中同理）。

稀疏信号所包含的信息量比较集中，也比较少，因此在压缩中有一定的优势。也正是因此可以降低采样速率恢复信号。

有时我们说$$\theta $$是`K`稀疏的，即$$\theta$$中包含`K`个非零项，其他项均为`0`。不过实际中也往往并非如此，实际应用中稀疏往往是其他项远小于`K`个明显的非零项（即近似为零但并不为零），呈现簇状分布。

### 信号的稀疏表示

信号的稀疏表示说，在给定的超完备字典中用尽可能少的原子来表示信号，可以获得信号更为简洁的表示方式，从而方便信号的编码和压缩等。信号的稀疏表示与压缩感知中的一般模型相似，例如假设我们用$$D$$矩阵表示字典矩阵，$$\theta$$表示信号的稀疏分解（稀疏表示），原信号为$$X$$，信号模型就可以描述为
$$
\begin{equation}
\begin{split}
X=D\theta
\end{split}
\tag{5}
\end{equation}
$$
这个表达就相当于将信号$$X$$表示为一个字典下的稀疏形式。如果这个字典矩阵是一个方阵，我们说这是一个**完备字典**，例如傅里叶变换阵；如果字典阵的形式类似于`Fig.1`，即列数多于行数，我们就称之为**过完备字典**。在**字典学习**问题中，我们优化$$\theta$$的选择，使得$$\alpha$$尽量稀疏的条件下，能够让$$X$$与$$D\theta$$匹配。这也是一个优化问题，即
$$
\begin{equation}
\begin{split}
argmin \sum\limits_{i=1}^m \left\| x_i-D\theta_i \right\|_2^2+\lambda\sum\limits_{i=1}^m\left\| \theta_i \right\|_1
\end{split}
\tag{6}
\end{equation}
$$
寻找尽可能少的最重要系数来表示原始信号的技术被称为**稀疏编码**或信号的**稀疏分解**（本质上是将原始信号分解为尽量稀疏系数的线性组合形式），从信号和字典矩阵中寻找稀疏表示的方法就是稀疏重建算法，其中包括贪婪算法、松弛算法、穷举法等。

> 对完备字典的任意线性独立补全都可视作过完备字典

## 字典学习

寻找合适字典的过程被称为是字典学习。我们假设一个字典对于指定的信号具有稀疏表示，因此选择字典的原则是能够稀疏地表达信号。

在自然图像处理中，我们常常会使用傅里叶变换基或离散余弦变换基进行处理，尤其是离散余弦变换基可以将大多数自然图像变换到一个相对稀疏的域。尽管如此，直接选择一个已有的变换基，虽然可能有利于数学分析和实现，但是却不能适用于所有情况。因此会设计许多字典学习方法，设计自适应的字典，以适应各种不同的信号。

> 具体方法日后补全

### 字典相关性

 我们假设$$X=D\theta$$中字典为
$$
\begin{equation}
\begin{split}
D=\left[\begin{matrix}\\ \\ d_1\:d_2\:...\:d_N\\ \\ \\\end{matrix}\right]
\end{split}
\tag{7}
\end{equation}
$$
则稀疏$$\theta$$将选出字典中的$$K$$列进行线性组合，表达信号$$X$$。我们说，字典的每一列$$d_k$$是字典的一个基。从这个角度来看，当这一组基恰好是一组正交基的时候，例如`DFT`变换基，字典就可以说是完备的了，但是这样的字典表达能力可能并不完善，因此人们会采用过完备字典，例如将矩阵横向间隔减小，从而使矩阵变得更长，因此增加了基的数量，提高模型的表现力。

字典的相关性是一个重要的问题。我们假设字典是高度相关的，则每一个线性组合之间就存在高度冗余，这将导致稀疏解不唯一或难以求解等问题。从另一个角度看，相关性越小可以证明信息量越大，则字典矩阵包含的信息量越大，也可以表达更多的信号。

相关性被定义为
$$
\begin{equation}
\begin{split}
\mu=\max\limits_{l\neq m}\mid \varPhi_l^T \varPhi_m \mid
\end{split}
\tag{8}
\end{equation}
$$
字典中的各个基之间的相关性越低，可以认为字典表达能力越强。

> 除此之外有人评价$$\varPhi$$和$$\varPsi$$之间元素的相关性，这种方法其实是另一个角度的解读。$$\varPhi$$是对一个信号的观测，而$$\varPsi$$则是一个信号的基，即原信号采用什么形式被表达。这个基函数矩阵往往是傅里叶基、小波基等，对应的会有时域冲激或`Noiselet`作为$$\varPhi$$观测矩阵，但是这仅仅是为了其相关性存在上界而保证正交性、非相关性，从而更容易对一个信号进行表达和重建，一般情况下，我们直接采用随机矩阵，例如白噪声随机矩阵作为观测矩阵$$\varPhi$$，就可以达到我们的要求。
>
> 这个相关性主要在压缩感知中提到，由于重建过程中存在一个取值范围，即
> $$
> \begin{equation}
> \begin{split}
> m\geq C\mu^2(\varPhi,\varPsi)S\log n
> \end{split}
> \tag{9}
> \end{equation}
> $$
> 此时重建失败就是一个小概率事件。为了保证重建成功率，我们确实需要更小的相关性、更小的稀疏度等关键因素。根据文献[2]的指示，经验上讲，以稀疏度的4倍进行采样就是一个足够的样本，即对于一个稀疏数据点进行四次非相干采样即可达成效果。

## 压缩感知

所以到底什么才是压缩感知？再看了一些文献之后，这个问题对于我来说更加的模糊。从数学角度来看，压缩感知是一种通过欠定方程组求解稀疏解的过程，考虑一个实际场景较容易理解，即`Fig.1`中的场景。但是实际上信号并非是完全稀疏的，我们只能在一些指定的变换域中得到一个信号的稀疏表示，其中才用的稀疏基函数可能是傅里叶、小波等，也可以是随机矩阵，自适应字典，但是通过一个字典表示后的稀疏信号，就可以通过一些算法重建。我个人认为这就是简单的压缩感知技术，更加深入的理解，将随着我的进一步学习更新。

压缩感知有时也被称为是压缩采样算法，因为确实可以通过这种算法降低传输的数据量等，有压缩数据的效果，也可以降低采样的负担。压缩还意味着更高的传输速率，这也是引发研究兴趣的一个原因。

不过压缩感知的问题并非到此为止了，因为实际应用中，压缩感知需要面对噪声和准稀疏（`nearly sparse`）的情况。一来，实际信号并非是完全稀疏的，而是我们所谓的准稀疏，这里就需要考虑，准稀疏信号是否可以通过指定算法重建；另外由于测量精度等问题，我们得到的结果总是有噪声的（测量噪声等）。**压缩感知算法应当拥有解决细微扰动的足够鲁棒性。**

原来的问题模型将修正为
$$
\begin{equation}
\begin{split}
\left[\begin{matrix}y_1\\y_2\\...\\...\\y_M\end{matrix}\right]=\left[\begin{matrix}\\ \\ h_1\:h_2\:...\:h_N\\ \\ \\\end{matrix}\right]\left[\begin{matrix}x_1\\x_2\\...\\...\\x_N\end{matrix}\right]+\left[\begin{matrix}n_1\\n_2\\...\\...\\n_M\end{matrix}\right]
\end{split}
\tag{10}
\end{equation}
$$
简化为
$$
\begin{equation}
\begin{split}
y=Hx+n
\end{split}
\tag{11}
\end{equation}
$$
$$H$$矩阵是感知矩阵，是用于采样的矩阵，或者称为观测矩阵，后面增加了噪声项。需要注意，本文中的$$x$$是稀疏信号，是在$$\varPsi$$基下的一个稀疏信号。因此我们说$$f=\varPsi x$$，即原信号在指定的$$\varPsi$$基之下可以分解/表示为稀疏的$$x$$。也是由于符号的问题，这个表达式可以写成很多形式，但是本质上是对稀疏的信号进行重建，从而进一步恢复出原信号。

> 这里先引入**有限等距原理(RIP)**这一概念。对于`S`稀疏的$$x$$，我们都有
> $$
> \begin{equation}
> \begin{split}
> (1-\delta_S)\left\| x \right\|_{l_2}^2\leq\left\|Hx\right\|_{l_2}^2\leq(1+\delta_S)\left\| x \right\|_{l_2}^2
> \end{split}
> \tag{12}
> \end{equation}
> $$
> 此时我们说矩阵$$H$$服从`S`阶的`RIP`准则。这个准则的意义是，通过$$H$$的映射使得这个`S`稀疏信号的欧几里得长度不发生明显的改变（不会将两个不同的稀疏信号映射到同一个结果），这对于信号的恢复极为重要，否则不满足单射条件，就无法达成信号重建的目的。
>
> 一种等效的表达是说，$$H$$中的`S`列是相互正交的。

所以重建的过程就等效为了一个优化问题，最小化$$\hat x$$的$$l_1$$范数。

理论上恢复结果是，如果我们的信号是严格的`S`稀疏，则重建将是完美的，否则理论上我们可以获取$$S$$个最大的位置的完美结果，前提是这些位置在一定程度上是稀疏的。



## 重建算法

拿到了稀疏信号按照字典的线性组合后，我们通过观测矩阵对原信号进行重建。*说原信号是稀疏信号，往往是在某一个特定的表示方式下的。*

### 贪婪

### 松弛

### 穷举



## Reference

[1] D. L. Donoho, "Compressed sensing," in *IEEE Transactions on Information Theory*, vol. 52, no. 4, pp. 1289-1306, April 2006.

[2] E. J. Candes and M. B. Wakin, "An Introduction To Compressive Sampling," in *IEEE Signal Processing Magazine*, vol. 25, no. 2, pp. 21-30, March 2008.

 