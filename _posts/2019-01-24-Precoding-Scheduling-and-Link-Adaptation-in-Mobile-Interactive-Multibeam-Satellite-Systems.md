---
layout: post
comments: true
title:  "Precoding Scheduling and Link Adaptation in Mobile Interactive Multibeam Satellite Systems"
excerpt: "-"
date:   2019-01-22 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
---

看标题也差不多明白，这篇论文像是一篇基础论文，讲解移动交互多波束卫星系统的预编码、调度与链路自适应问题。这个文章是强化学习解决自适应问题的一个基础吧。正好我也不是很懂这一方面，来看一看好了。

`2019-2-1 10:50:24`

真的感谢通信老师给我上80

## 摘要   -   Abstract

本文讨论了下一代移动交互式**多波束卫星系统**的预编码(`precoding`)、调度(`scheduling`)和链路自适应(`link adaptation`)问题。

相比固定卫星服务，当用户终端在覆盖范围内进行移动的时，额外的挑战会出现。由于时变信道(`time varying channel`)的原因，网关只能访问到信道状态信息(`Channel State Information, CSI`)的延迟版，而这最终会限制整个系统的性能表现。

然而，相比通常的多用户多输入多输出的地面系统，多波束移动应用中的CSI退化对典型衰落信道和系统假设的影响非常有限。在实际条件下，数值计算的结果表明，与保守的频率复用分配相比，预编码可以在系统的吞吐量(`throughput`)上提供一个非常好的(`attractive`)的增益。

关键词索引：预编码`Precoding`, 多波束卫星系统`multibeam satellite systems`, 自适应编码与调制`adaptive coding and modulation`, 不完全的信道状态信息`imperfect channel state information`

## I - 引入   -   Introduction

移动交互卫星(`Mobile interactive satellite`)通过低用户成本设备提供了全球语音和数据传输服务。由于用户终端(`User terminal, UT`)的要求，它们（移动交互卫星）的硬件与固定服务的一般的高增益天线不同，*因为人们更需要非定向的低剖面天线(`low profile antenna`)。通信服务与数据传输服务往往被分配在**L**波段，因为`L`波段相比`Ka-Ku`波段卫星传输，即使可用带宽很小，但是路径损失(`pathloss`)要小一些。

随着L波段卫星容量(`capacity`)的增加，波束间的高频复用成为一个非常有趣的选择(`interesting alternative`)。事实上，当它？发生在地面系统中时，增加给定频段的频谱效率可以通过更积极的频率复用和干扰缓解技术来实现。

减少多用户干扰既可以在发射机上通过预编码完成，也可以在接收机上通过多用户检测技术完成。虽然后一种方法增加了用户终端(**UT**)的复杂性，但是预编码技术要求使用额外的资源来反馈信道状态信息(**CSI**)，网关(**GW**)设计也需要额外的复杂度。这些从操作者的角度来看，对整个系统的影响是有限的。考虑到预编码在固定卫星服务中的高增益，以及多用户检测技术在移动系统中的有限增益，并考虑到移动系统中追求的低成本用户终端，本文的主要重点放在了预编码方法上。

事实上，由于预编码技术很大程度上依赖于发射机对CSI了解的准确程度，用户终端的移动性通常会降低固定状态下的可实现率。另一方面，L波段移动系统的一个优势是，可以使用单个GW来馈送通过卫星的总流量。这增加了预编码的增益。

本文研究了有延迟的CSI信息对 L波段预编码移动多波束卫星系统 的影响。从作者的了解来看，这应是首次在移动环境下研究卫星通信系统的预编码问题。本文设计了一个全新框架，将移动性纳入到预编码多波束卫星系统中。考虑到用户终端和网关之间存在延迟，并认为速率分配是完美的(`GW对调制和编码进行最优分配`)，我们评估了一种低复杂度的闭式线性预编码技术。由于多波束卫星信道的特殊衰落特性，只有接收到的信号功率受到影响。此外，我们发现给定任意预编码设计  和  理想的调制以及编码方案选择 (`modulation and coding scheme (MCS) selection`)，无论是完美的CSI还是延迟的CSI都会得到相同的各态历经的和速率(`ergodic sum-rate`).

当我们执行一个实际的MCS分配时，理想CSI和延迟CSI的差异出现了。这种性能表现的损失主要是由于 增加固定冗余(`ﬁxed margins`)而减少的中断(`outage`) 引起的。即使是在完美CSI甚至延迟CSI上会出现性能的下降，但预编码在平均吞吐量方面也会有显著提高。具体而言，在比较具有相同传输功率和目标中断(`outage`)  的系统时，预编码能够在单播(`unicast`)传输情况下提供较大的吞吐量增益。

尽管延迟的CSI会影响调度过程，但我们通过使用所建议的调度算法和固定的余量，观察到所考虑的场景具有相当高的增益。

后面的安排主要是

- 第**二**部分描述移动预编码卫星系统的信号与系统模型
- 第**三、四、五**部分描述了指定场景下推荐的预编码、调度与链路自适应算法
- 第**六**部分评估了一个接近现实场景的设想
- 第**七**部分主要是总结结论

---

**符号说明**

- 黑色大写字母表示**矩阵**，黑色小写字母表示**列向量**
- `Hermitian transpose:`$$ (\cdot)^H$$
- `transpose:`$$(\cdot)^T$$
- `conjugate:`$$(\cdot)^*$$
- `diagonal(with positive diagonal elements):`$$(\cdot)^+$$
- $$\left[ X \right]_{i,j}$$表示$$X$$矩阵的$$i$$行$$j$$列元素，$$\left[ x \right]_{i}$$表示$$X$$向量的$$i$$行元素
- $$\mid\mid \cdot \mid\mid$$表示弗罗贝尼乌斯范数运算（或希尔伯特-施密特范数），通俗地讲就是**2-范数**；$$\mid\cdot\mid$$就是绝对值运算符。
- $$\circ$$表示`Hadamard Matrix Product`

## II - 系统与信道模型   -   System Model and Channel Model

### A - 系统模型

假设有一个多波束卫星系统，其中卫星配备有阵列馈电反射天线，总馈源数为$$N$$。这些馈送信号被组合起来形成一个$$K$$波束辐射模式，其中这个$$K$$是一个固定值。在每一帧中，我们假定来自同一波束$$N_u$$用户的多路信号是多路复用(`multiplexed`)的。这是卫星标准中的一种方式，以便通过统计复用实现高帧效率。这其实就是说，每一个时间帧中，$$KN_u$$个用户被提供服务。

考虑到所有波束都在同一频段内辐射，因此可以将任意时刻$$t$$接收的信号建模为

$$y(t)^{[i]}=H(t)^{[i]}x(t)+n(t)^{[i]};i=1,2,...,N_u;\tag{1}​$$

其中$$y(t)^{[i]}$$是一个复向量，该向量包含了第$$i$$个用户终端的接收信号。

- $$\left[y(t)^{[i]}\right]_k$$代表第$$k$$波束的第$$i$$个用户终端的接收信号









---

`2019-1-22 15:34:41`

看不懂，还是看不懂，啥都看不懂，他们好厉害啊怎么啥都会。

## Reference

[1] [Precoding Scheduling and Link Adaptation in Mobile Interactive Multibeam Satellite Systems](https://ieeexplore.ieee.org/document/8353925)

