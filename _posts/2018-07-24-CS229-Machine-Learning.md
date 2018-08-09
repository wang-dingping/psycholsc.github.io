---
layout: post
title:  "CS229 I"
date:   2018-07-24 06:35:54 +0000
categories: Notes
comments: true
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

*那一天，他终于想起来自己还有坑要填*

[CS229](http://cs229.stanford.edu/) 是斯坦福大学 **Machine Learning**的公开课

由于官网视频质量一般，此处部分参考了[Coursera](https://www.coursera.org/)上的新版视频。

# CS229

## Introduction

在Stanford网站上有完整的讲义和演讲稿，可以作为参考。

本次lecture的topic是

> The Motivation & Applications of Machine Learning, The Logistics of the Class, The Definition of Machine Learning, The Overview of Supervised Learning, The Overview of Learning Theory, The Overview of Unsupervised Learning, The Overview of Reinforcement Learning 

用中文说就是

> 机器学习的动机（出现原因）与应用，定义，监督学习的概述，学习理论的概述，无监督学习的概述，强化学习概述，（课堂后勤）。

某种意义上说是一节导论课，机器学习导论。Andrew Ng，或者说吴恩达，是这个领域较为出名的人物，已经在机器学习领域进行了大约15年的研究。班级助教也都是一些相关领域的研究生，例如神经科学、计算机视觉等，可以看出Stanford的教育资源是十分顶级的。参与这个课堂的学生也来自不同系，希望使用Machine Learning解决各自领域的问题。



首先关于Machine Learning的起源，大约是早期的人工智能领域的相关工作。而在近15-20年里，ML被看作是电脑的一种正在发展的新能力。因为越来越多的实践表明，许多应用程序仅仅依靠手动编程并不能完成所有任务。举例说明的话，比如希望用电脑来读取识别手写字符数字，这个实际上很难（在后面这里会有一个基于mnist数据集的handwriting recognition，简述最简单模式识别的基本原理）。如果我想使用传统算法进行计算，那么就需要我去做特征提取与匹配（这些内容在“数字图像处理”相关课程中会有详细说明），这个过程其实相当麻烦，尤其是针对字符这种类型的数字图像；再比如我要写一个飞行器的主控，一个复杂飞行器的主控，想要使其还能适应复杂天气条件，那么软件在编写上就需要很大的工程量和实践数据。相比之下想要完成这些任务，使用一个学习算法会十分的有效。实际上目前为止，识别使用的唯一有效方法就是让计算机去学习。



而现在，学习算法也已经应用到了医学上，例如随着计算机的发展，医院会记录每一位病人的病例，并且依据巨大的数据库，将学习算法应用到医疗数据上，使得我们可能利用统计方法学习到医学的相关知识或规律。在过去的15-20年里，美国已经在建立电子的医疗数据库了。

在我们的生活中实际上我们每个人每天都有可能接触十多次学习算法相关的产品或技术，但我们并不知晓。就像我们发送邮件的时候，邮件的自动分类；通过邮局使用邮政系统的时候（美国邮政），机器自动识别邮政编码；又或者是支票上手写数字的识别，这些都是拜学习算法所赐。甚至是使用信用卡的时候，都有专门的学习算法设计来判断你是否被盗刷。

再比如当你使用一些购物或视频网站的时候，你的浏览记录也会被用来作为学习素材来供学习算法使用，网页也会依据这些信息向用户推荐合适的产品或视频。

除此之外，学习算法还应用于人类对基因组信息的挖掘等多个方向。



前面铺垫了这么多，就是希望所有人都能对机器学习产生兴趣、学会在各种问题上应用学习算法，并对相关研究产生兴趣（当然随便看看也是可以的）。



接下来就是正式内容了，首先我们假定所有人都对计算机相关的基础知识有一定了解，并有一定的编程能力，例如知道什么是复杂度，什么是数据结构。课程不会对编程有过多的要求，但是我们会编一些简单的程序。另外这个课程会涉及一部分的概率与统计的知识，学校开设的本科生概率论与数理统计已经是足够的了。另外还需要了解线性代数的相关知识，本科的线性代数课程已经绰绰有余，知道矩阵与向量和他们的基本运算即可。



那么究竟什么是机器学习呢，这个问题要追溯到1959年，机器学习被Arthur Samuel精确定义，这也是机器学习的第一个定义，大致为让机器在不被明确编程的条件下赋予计算机一定的学习的能力。他编写了一个简单的西式跳棋游戏，并让游戏AI在对抗自己的时候进行学习，久而久之这个程序甚至比他本人玩的还好。这一点甚至可以证明计算机可以完成不被明确编程的任务，即通过学习去做。

更近一些的机器学习的定义为，“a computer program is set to learn from an experience E with respect to some task T and some performance measure P if its performance on T as measured by P improves with experience E”，一个程序被认为能从经验 E 中学习,解决任务 T,达到性能度量值 P,当且仅当,有了经验 E 后,经过 P 评判,程序在处理 T 时的性能有所提升。 在上述跳棋中，E是跳棋的经验，而设置对人类获胜的游戏得分为P，通过这个定义我们就可以认为这个机器学会了跳棋。



首先我们开始整个课程的第一部分，**监督学习（Supervised Learning）**的相关内容。

首先举一个监督学习的例子，假如你收集到了一个数据集，用来表示附近房屋的价格，注意这个价格是出现在一个确定地理区域的。我们以房屋的面积为x轴，房屋的价格为y轴，这时可以绘制一个散点图，图像如下。

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/HousingPrice.png)

假如我也想在这个区域卖房，那么我们是否可以根据我的房屋面积来确定它的期望价格呢？实际上有许多方式，这里只说其中的一种。最简单的方式是使用一个直线对数据进行拟合，我们称其为线性回归，但是如果我希望使用其他函数对数据进行拟合，结果可能会更加精确。这种学习去预测房屋价格的问题被称为监督学习问题，因为我们需要给算法提供一个面积的数据集合一个房屋价格的参考值。

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/HousingPriceLinear.png)

这是线性拟合可能的结果

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/HousingPriceSquare.png)

假如使用二次函数进行拟合，就会得到这样的结果。

至于如何选择拟合函数，这在后面会讲到，另外值得一提的是，使用何种函数进行拟合，这个函数往往被称为是一个超参数（实际上是指函数的幂）。监督学习中的超参数不能通过学习来获取。

监督学习要求我们给出具体的数据集，在本问题中，就需要给出房屋价格与面积的对应数据集，通过对这个数据集的建模拟合，才能得到我们想要的结果。

这个例子是一个**回归(Regression)**问题，而回归分析是指企图去预测一个变量是**连续值**的输出值。

还有一类监督学习问题叫做**分类(Classification)**问题，这一类问题的变量往往是离散的，例如，你收集了一份乳腺肿瘤的数据集，我们需要一个算法去判断一个肿瘤是否为恶性肿瘤，因此我们收集了关于这个乳腺肿瘤问题的诸多信息，假设我们只通过大小来判断，那么我们的问题就是，假设输入参数为肿瘤大小，输出结果为是否恶性。这个结果只能是0或1的，这时的问题就是分类问题。这时Y轴的坐标就是0或1，即表示肿瘤是良性还是恶性。这还是一个典型的二分类问题。

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BreastCancer.png)

假设收集到的数据如上，我们对这个问题进行建模的时候就会发现这是一个分类问题。

当然分类问题的类别往往不只是一类。分类问题的类别还可以是很多类别，这一些在感知机等地方应该已经学到过了，其中感知机是线性分类器，而利用机器学习的分类器就未必是线性分类器了。这个内容有机会会单独说一下（或者补在下面）。

对于二维的分类问题，往往会将数据画成这样

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BreastCancerII.png)

对于更多维度的分类问题，会有不同的表示方式。

当我们遇到一位用粉色标出的病人时，我们就可以利用我们的模型确定，这个肿瘤是良性的（但是实际结果还是要依据医生的判断。正规的判断流程必须要有经验丰富且接受过专业培训的医师进行参与，本视频系列的作者、讲师吴恩达前些时间带领的团队在医学影像诊断肿瘤的实验中，并没有成功建立一个识别率可以替代专业医师的模型）。

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/BreastCancerIII.png)

当然上面的分类器也只是一个例子，因为实际上对于医学影像进行识别的过程往往需要更多的参数来辅助分类器进行诊断，例如块厚度，肿瘤细胞的大小形状的一致性等等。特征量往往会很多，这对于传统的线性分类器来说也会降低其精准度。不过这里要先从支持向量机说起，这个简单的算法可以让计算机理论上处理无穷多的特征。

以上是**监督学习**的相关概述，以下是**无监督学习（Unsupervised Learning）**的相关内容。

当我们的样本数据集没有被标识哪些输入应当对应哪些输出，即没有标签的数据，例如

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/UnsupervisedI.png)

我们的问题就成为了，“假如我给定了一个数据集，你是否能够应用学习算法发现数据集中的某些结构或规律吗”。

对于上述图像的数据集，我们可以得到的信息也有不少，例如我们知道这个数据集可以组成两组**聚类（cluster）**。实际上聚类算法就是典型的无监督学习方法。聚类算法的应用很广，例如新闻的分类推荐，医学上不同基因组个体的分类，计算机集群管理、社交网络分析、天文数据分析、市场分配管理等。在算法执行前，我们并没有告诉程序要如何进行聚类，所有的聚类都是算法在无监督条件下完成的。

我们在电子领域常见的**鸡尾酒会（Cocktail Party）算法**，即把两个相互叠加的音频相互分离的算法，实际上也是一种无监督学习算法。如果我们只关注实现，而不关心算法要如何去实现的时候，实际上这是一个较为简单的工作，只需要一行代码就能解决

[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

这实际上是一条Octave语句，在简单的考虑实现效果的条件下可以对Octave，MATLAB或Python等语言进行了解。Octave与MATLAB语句较为相似，易于移植。对于学习过程，吴恩达推荐了Octave。实际上目前条件下使用Python也是较为简单的。当我们用这类语言确定可以实现后，我们就可以将其移植到C++等较快的编译平台上去执行。

## Linear Regression

线性回归是什么？这都不知道的话还是回去学数学比较好。初中和高中数学都学过使用最小二乘法解决线性回归的问题，这里不再赘述。

线性回归是很多初学者的第一课。吴恩达老师仍然使用了房价数据集，此处可能做调整。用一个曾经使用的例子来看，假设散点是这样的



那么我们似乎可以认为，这些散点可以用一条直线来拟合，即线性回归。我们将拟合直线绘制到平面上，效果如图。



本程序使用的线性回归的方法被称为“正规方程法”，需要学习对矩阵的求导后才能熟练掌握该算法。

我们假设整个数据集拥有的样本总数为$$m$$个，输入特征量为$$x$$，输出变量为$$y$$，那么可以使用$$(x,y)$$这种形式来表示一组训练样本。对于整个数据集中的第$$i$$个数据，此处将用$$(x^{(i)},y^{(i)})$$来表示。

这时我们手里会有一个训练集，包括了$$x$$和对应的标签$$y$$。这时将训练集输入给我们的学习算法，学习算法将输出一个函数$$y=h(x)$$，描述了$$y$$与$$x$$的对应关系。这里用$$h$$是代表了$$hypothesis$$，实际我个人习惯还是写成$$f$$。对这个输出函数输入一个$$x$$，这个函数将给出一个拟合的$$y$$。

![HousingPrice](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/LinearRegressionI.png)

对于线性回归的关系，我们将表示$$h$$为

$$h_{\theta}(x)=\theta_0 +\theta_1x$$

当然这个结果可以不是线性函数，但此处只对线性回归进行说明。





{% if page.comments %}
<div id="container"></div>
<link rel="stylesheet" href="https://imsun.github.io/gitment/style/default.css">
<script src="https://imsun.github.io/gitment/dist/gitment.browser.js"></script>
<script>
var gitment = new Gitment({
  id: '21', // 可选
  owner: 'psycholsc',
  repo: 'temp',
  oauth: {
    client_id: '9183e7259ea6d850a7df',
    client_secret: 'd0a82473ca685629b50ded0553f402b6ba2b2dee',
  },
})
gitment.render('container')
</script>
{% endif %}