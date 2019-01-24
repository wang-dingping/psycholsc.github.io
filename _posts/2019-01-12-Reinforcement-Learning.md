---
layout: post
comments: true
title:  "Reinforcement Learning"
excerpt: "-"
date:   2019-01-12 14:42:24 +0000
categories: Notes
---

<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


---

很久没更过了。2018年下半年过得很狗屎。十分难受。

# Reinforcement Learning

`Reinforcement Learning`的正统翻译目前是**强化学习**，是所谓的机器学习的一个分支领域，强调如何基于环境而行动，以获取最大化的预期利益。其灵感来自于心理学中的行为主义理论，即有机体如何在环境基于的奖励或惩罚的刺激下，逐步形成对刺激的预期，产生能获得最大利益的习惯性行为。这个方法具有普适性，因此在其他许多领域都有研究，例如博弈论、控制论、运筹学、信息论、仿真优化等。在一些研究环境下，RL还被称为近似动态规划(`approximate dynamic programming, ADP`)。在最优控制理论中也有研究，主要研究方向为最优解的存在与特性，而非此学科的学习、近似等。该理论还被解释为在有限理性的条件下如何平衡。

在机器学习问题上，环境通常被规范为Markov决策过程，所以许多强化学习算法在这种条件下使用动态规划技巧。强化学习不需要对Markov决策过程的知识。强化学习和标准的监督式学习的区别在于不需要出现正确的输入输出对，也不需要精确校正次优化的行为。强化学习更加专注于在线规划，需要在**探索**和**遵从**之间找到平衡。

>Reinforcement is about taking suitable action to maximize reward in a particular situation.

由于不像传统强化学习一样是通过正确输入输出来进行训练的，而是通过一个`agent`决策在指定任务下应怎么做。在没有数据基础的前提下，强化学习是通过自己的经验去学习的。

强化学习是一个很大的坑，从头写写看，看看能坚持多久吧。

## 第一章 Introduction

### 引入

当我们说起学习的本质的时候，我们总会认为第一个想到，我们是通过与环境的互动学习的。我们出生的时候，并没有老师教我们如何进行活动，但是我们自己确实与环境有直接的感觉运动联系。人类的一生中，许多学习都是这样的，就像今天`2019-1-23`，面对两把外观相同的钥匙，经过一段时间的观察，我就发现哪一把是刚配的，哪一把是旧的。

在我们的一生中，通过与环境的互动来进行学习的例子很多，我们大多数时间都是通过环境的反应进行学习的，开车，交谈，一切都是这样，我们敏锐地意识到我们的环境是如何对我们所做的做出反应的，我们寻求通过我们的行为来影响所发生的事情。从互动中学习是几乎所有学习和智力理论的基础。

现在我们尝试找出一种计算方法，让我们从互动中进行学习。后面的探索基本上也都是基于理想的学习情境，站在研究者的角度评价不同学习策略的有效性。

强化学习是学习如何决策的，如何将情境映射到决策，从而最大化某种数字化的奖励信号（被称为回报，`reward`）。学习者并没有被告知应该如何决策，而是通过尝试来发现哪一种决策可以获得最大的收益。在复杂而有挑战性的情境中，行动可能不仅会立即影响回报收益，还有可能影响下一种情况，并通过这种情况进而影响后续的回报。

书中利用动力学系统理论的思想，特别是作为不完全已知马尔可夫决策过程的最优控制，将强化学习问题形式化，基本想法是简单地捕获`agent`随时间与环境交互以实现目标所面临的实际问题的最重要方面。

`agent`必须能够在某种程度上感知环境的状态，并且必须能够采取影响状态的行为。

`agent`还必须有一个或多个与环境状态相关的目标。

`Markov`决策过程就是以包含这三个层次为目标的，即**感知**、**决策**和**目标**。仅仅对于它们可能的最简单的形式，但是绝对不会将任何一个看的不重要。任何一个非常适合用来解决这些问题的方法都可以被认为是强化学习方法。

**监督学习**是通过`training set`和`labeled data`进行学习的，通常是分类器或回归问题；典型的**非监督学习**是发现`unlabeled data`中的隐藏的一些信息结构。这两个分类看似对机器学习的范式进行了具体的分类，但是并没有。强化学习并不属于典型的非监督学习，但是他并不是用于发现隐藏于信息中的内容的，而是希望让某种`reward`最大化，仅仅发现信息结构并不能完全达到这个目的，虽然是有益的。因此可以认为强化学习是机器学习的第三个门类，另外值得一提的是，机器学习的门类可能远不止目前的三个门类。

强化学习遇到的其中一个重要问题是，如何在`exploration`和`exploitation`之间寻求平衡。

- 为了获得更多的`reward`，`agent`自然会优先选择曾经尝试过的并产生了大量`reward`的决策
- 上述决策的基础是，必须曾尝试过这个没有选择过的决策

因此`agent`必须要利用其已有的经验来获得`reward`，但也必须进行探索，以便在未来做出更好的行动选择。`agent`必须尝试各种决策，并逐步支持那些看起来最好的行为决策，与此同时也要继续对其他决策进行探索。在一个随机任务中，每一个行为都需要经历多次尝试，以获得随机行为的可靠估计。数十年前数学家就对这个所谓的`exploration–exploitation dilemma`进行过研究，然而并没有什么可靠的解决方案。上述的两种机器学习方法也都还没解决这个问题。

强化学习的另一个关键特征是它明确地考虑了目标导向的`agent`与不确定环境相互作用的整个问题。这与很多只考虑子问题而不考虑该如何将它们在更大范围内实现的方法形成对比，例如监督学习方法。这种只关心子问题的方法存在明显限制，例如它们只学到了如何对指定数据集进行回归，而不知道面对回归问题时该如何进行操作。（不知道这个理解对不对）（这个**子问题（Subproblems）**实在是不懂在说啥）

目前强化学习与许多其他工程和科学学科已经有了富有成效的结合，也反过来促进心理学与神经科学的发展。这些内容在书中最后部分。

至于发现机器智能的一般原则的过程中，强化学习所做的贡献也是十分有限的，目前投入的研究资源也很少，在这条道路上还得不出结论。

> Modern artiﬁcial intelligence now includes much research looking for general principles of learning, search, and decision making, as well as trying to incorporate vast amounts of domain knowledge. It is not clear how far back the pendulum will swing, but reinforcement learning research is certainly part of the swing back toward simpler and fewer general principles of artiﬁcial intelligence.

简单举例几个强化学习应用以后我就直接进入主题。

- 象棋高手的每一步决策
- 自适应控制器实时调整某项参数
- 新生动物在出生后的几分钟内学会站立（站起来！萌萌！站起来！）
- 扫地机器人基于当前电量决策继续进行垃圾收集还是还是回到充电站。

这些都涉及一个积极的决策`agent`与其环境的相互作用，尽管环境可能并不确定，但是`agent`仍然寻求一个目标。`agent`的行为决策可能会影响未来的环境，例如上述的象棋决策，从而会进一步影响`agent`以后可能获得的选择与机会。

>Correct choice requires taking into account indirect, delayed consequences of actions, and thus may require foresight or planning.

### 基本元素

除了`agent`和`environment`，我们还可以识别强化学习系统的四个主要基本元素，策略`policy`，奖励信号`reward signal`，价值函数`value function`，（可选）环境模型`model`

- `policy`定义了`agent`在指定时间的行为方式，即从环境感知到的状态与决策之间的映射关系。一般来说，`policy`可能是随机的
- `reward`是强化学习问题的目标。在每个时间步骤中，环境都向`agent`发送一个数值，称为`reward`。`agent`就是在长远角度来看尽可能地获取最大的`total reward`。一般来说，奖励信号可能是环境状态和所采取行动的随机函数。
- `value function`指定的是`agent`期望的总`reward`。`reward`可以说是短期的收益，而`value`才是长期收益。
    - 从某种意义上说，`reward`是首要的，而`value`会次要一些。但是我们实际采取决策的时候，`value`才是最重要的因素，因为我们的目标是长远来看获取最大收益。不过确定`value`比确定`reward`要困难得多，对于`value`的估计也是强化学习算法中最重要的部分，过去数十年的研究也是基于此。
- `environment model`是对环境行为进行模拟的模型。具体内容在第8章才涉及



### 限制

强化学习过于依赖`state`这个概念。在RNN中`state`是灵魂，强化学习本不应是类似的模型，但是却一样地依赖于状态。强化学习中的每一个元素都可以看做是一个状态。我们主要关心的问题不是如何设计状态信号，而是在任意状态信号可用时该如何采取行动。



**这里本该有一个Tic-Tac-Toe的示例，此处先行略去**

## 第二章 Multi-armed Bandit Problem

多臂赌博机问题。









# Reference

[1] [Reinforcement Learning: An Introduction](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLIntro2nd2018.pdf)

