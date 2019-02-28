---
layout: post
comments: true
title:  "Reinforcement Learning: The Multi-Armed Bandit"
excerpt: "-"
date:   2019-02-24 14:42:24 +0000
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

`2019-2-28 15:30:06`重写

前面提到的强化学习有别于**监督学习方法**和**非监督学习方法**，其主要区别在于，强化学习通过训练数据、信息去评价每一次`action`，而非通过正确标注的输入输出来指导系统的改善。但是也正是如此，我们需要积极的`exploration`，即明确的对正确`action`的探索。这里我们先简单介绍两个`feedback`，一个是`evaluation feedback`和`instruction feedback`。前者可以直译为“**评价反馈**”，是完全依赖于`action`的一种反馈，真实反馈目前`action`的水平；后者是“**指导反馈**”，完全独立于采取的`action`，反馈的是当前环境水平。目前的监督学习方法实际上使用的是这类反馈。

本节通过最简单的多臂赌博机问题，介绍最简单的强化学习模型，并且在此基础上理解上述两类反馈的思想，结合两类反馈进行实现。

### K - Armed Bandit

K臂赌博机问题实际上是一个循环决策问题，每个时间步面对一个`K-option`问题，每个选项都依一定的概率分布返回一个`reward`。该问题的目标就是最大化累计`reward`，是一个决策问题。

对于最简单的赌博机模型，即`Bernoulli Bandit`模型，赌博机会根据预先设定的概率返回`reward`为$$1$$，否则返回$$0$$。但是这里并没有采用这种简单的情况，而是采用依正态分布的方式返回`reward`。一方面能让问题稍显复杂，也能更加接近实际的决策问题。

最简单的赌博机模型如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/bern_bandit.png" style="display: inline-block;" width="500"/>
</div>

这个模型可以类比很多现实的决策问题，例如赌博问题，医生对病人治疗方案选择问题等。对本模型的介绍，就从`bandit`问题开始。在这个问题中，我们每一个时间步都会做一个唯一的决策`action`（即选择一台赌博机），每一个`action`都会得到一个对应的`reward`（赌博机给出收益）。我们用$$A_t,R_t$$分别表示$$t$$时刻的`action`和`reward`。对于每一台赌博机，我们**首先**都认为存在一个固定的期望收益（这种情况被称为`stationary`情况，大多数强化学习遇到的问题实际上都是`non-stationary`问题，这里只是最简化模型），用$$q_*(a)$$来表示，
$$
\begin{equation*}
\begin{split}
q_*(a)=\mathbb{E}[R_t\mid A_t=a]
\end{split}
\tag{1}
\end{equation*}
$$
即每一个`action`的收益期望。

假如我们知道了每个行动的期望收益，那我们的决策就变得简单了——只需要选择期望收益最大的那个决策，在足够长的时间后一定是这个选项带来的平均收益最大（逼近期望值）。

可是对于一般的决策问题，我们是肯定对这个数值一无所知的。我们一开始对各台赌博机没有任何先验知识，那么只能通过我们的行为收集环境信息，以此来估计各个赌博机的收益期望。这个估计的结果我们标记为$$\hat Q_t(a)$$，即时间步$$t$$时我们对各个选项的收益期望的估计。当然为了长期收益达到最高，我们希望这个估计值能够尽可能得逼近真实值。

如果我们持续对$$q$$进行估计，那么每个时刻总是至少有一个$$Q_t(a)$$值是最大的，这个值我们称为`greedy value`，一般的我们会根据$$\hat Q$$进行决策，这种决策方法就是`greedy`决策，对应的行动为`greedy action`。当我们依据这个策略进行决策的时候，我们就被称为 进行`exploit`。而相反的，如果我们不采用`greedy action`的话，就被称为`explore`。

那么到此为止，我们已经发现了一个最为重要的问题，即**Exploration vs Exploitation**的问题。

### Exploration vs Exploitation

如上方所说，这两个不同类型的`action`是需要我们进行选择的，因为如果我们持续`exploit`，我们就只会选择$$\hat Q$$中最大的一个`action`。但是这种情况下$$\hat Q_t(a)$$的估计结果是相当不准确的，因为我们只重复尝试其中的一个或几个看起来短期`reward`比较高的`action`，但是却不知道那些从来没有选过的`action`是否会带来更多的收益。但是一直进行`explore`会导致我们总是进行相对随机的`action`，我们一直在探索，最终可能将$$\hat Q$$估计到十分接近$$q_*(a)$$的程度。但是这样我们很少选择最优决策方案，这就会导致总收益不高。这里面临的问题就是如何进行`explore`和`exploit`来保证累计`reward`达到最高。

UCB介绍这个问题时采用的例子是，我们每天去吃饭都需要选择餐厅。假设我们面前有两个选择，一个是我们经常去的餐厅，另一个是我们从没有去过的餐厅。对于我们每天都去的餐厅，我们对于吃什么东西是非常了解的，因此我们一般来讲会吃的比较满意；但是对于从未去过的餐厅，这里可能并不好吃，但是也有可能有更佳选择。我们一直选择其中的一个都不是最佳决策，因此需要在两个选项之间权衡。相似的，



<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/exploration_vs_exploitation.png" style="display: inline-block;" width="500"/>
</div>
正是因为两个选择在一个时间步内只能选择其中的一个，两者的关系经常被形容为`conflict`

在各种条件下，两者的取舍是取决于当前的估计$$\hat Q$$、`uncertainty`、剩余时间步数等参数的复杂相互关系的。两者之间的用于平衡取舍的算法很多，甚至有很多十分精细复杂的算法，但是绝大多数都对应用情境与先验知识做了强有力的假设，要么就不能在实际中验证，要么就不能在全面考虑问题时不可验证。当假设不成立的时候，实际效果也不明显，最优性、有界损失等难以令人满意。这里首先从最简单的开始说起。

### Action - Value Methods

#### Epsilon Greedy

简单的看几个估值算法。一个`action`的真正`value`前面用$$q_*(a)$$进行表示，代表的是收益的期望。一种最简单的方式就是在计算的每一个时间步进行如下计算
$$
\begin{equation*}
\begin{split}
Q_t(a)= & \frac{sum\:of\:rewards\:when\:a\:is\:taken}{number\:of\:times\:when\:a\:is\:taken} \\= & \frac{\sum\limits_{i=1}^{t-1}R_i \mathbb{1}(A_i=a)}{\sum\limits_{i=1}^{t-1}\mathbb{1}(A_i=a)}
\end{split}
\tag{2}
\end{equation*}
$$

其中的函数$$\mathbb{1}(\cdot)$$表示逻辑函数，当括号内为真时返回1，否则返回0。一般的初始值$$Q_1(a)$$均为$$0$$，当次数无穷多时，根据大数定律，被选择了无穷多次的`action`的`value`估计值$$\hat Q$$就会逼近$$q$$，即真值。这种方法被称为**采样平均法**（`sample-average method`）

显然这种方法并不是最优方法，但是接下来我们就先利用这个方法进行`action`的选择。一般的为了短期最优`reward`回收，会采用一个`greedy action`，即估计值最高的`action`，如果此时同时存在若干相同取值的就随机在值最大的`action`中选取一个
$$
\begin{equation}
\begin{split}
A_t=\underset{a}{\operatorname{argmax}}Q_t(a)
\end{split}
\tag{3}
\end{equation}
$$
这种方法就被称为`greedy action selection`。这种方法利用了现有的知识去最大化`immediate reward`

当然以上操作只做了`exploit`，但是很明显这个方法是不好的，不做`exploration`的话我们可能永远也不会知道最好的那个选项是哪个。那么对上述算法的一点小小改进就是，在决策时依某一小概率$$\varepsilon$$进行选择`exploration`

这样在大部分时候（$$1-\varepsilon$$）我们都进行`exploit`，利用现有知识进行决策，而少部分时候（$$\varepsilon$$）我们进行`explore`，探索是否存在更好的解决方案。这个方法被称为$$\varepsilon-greedy$$方法，其优点在于，在时间无限长的条件下，每一个选项都有无穷多被选中的次数，也因此$$\hat Q$$能够依**大数定律**正确地收敛到$$q$$。这也保证了我们能够在有限的时间内让结果更加接近最优解。

当然这个结果也是一个渐近保证，实际应用中由于收敛速度较慢，优势也并不高。

我们来简单看一下结果，代码在我的GitHub中有，过几天加个链接。以下测试结果是在一个`10 armed testbed`上进行的，这是一个十臂赌博机（或者说是十个赌博机），每个赌博机的输出都是以某个均值正态分布的，方差为1；每个赌博机输出的均值是零均值高斯分布的，我们最终选出的结果为

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/10armedtestbed.png" style="display: inline-block;" width="500"/>
</div>

这个是我们的赌博机配置，然后就要设计程序在没有先验知识的条件下，更快更好地找出最优决策方案。我们实验是基于$$1000$$个时间步进行的，然后借助各态历经性的思想，我们进行2000次反复试验，并将结果进行平均，得到结果如下

<div style="text-align:center"><img alt="" src="https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/10armedbanditresult.png" style="display: inline-block;" width="500"/>
</div>
显然







$$
\begin{equation*}
\begin{split}
\theta^*=Q(a^*)=\max\limits_{a\in A}Q(a)=\max\limits_{1\leq i\leq k}\theta_i
\end{split}
\tag{1}
\end{equation*}
$$

损失函数`loss function`是我们所有时间步$$T$$的没有选择最优决策而可能产生的总遗憾`total regret`

$$
\begin{equation*}
\begin{split}
L_T=\mathbb E \left[ \sum\limits_{t=1}^T\left(\theta^*-Q\left(a_t\right)\right) \right]
\end{split}
\tag{2}
\end{equation*}
$$

#### Bandit Strategies

基于我们`exploration`方法的不同，我们有很多种方法解决多臂赌博机问题

- 不进行`exploration`
- 随机`exploration`
- 对于不确定性进行优先`exploration`

### $\varepsilon -Greedy\:Algorithm$

$$\varepsilon$$贪婪算法大多数时间里能够取得最佳`action`，但是偶尔会进行随机`exploration`。我们通过以往的经验**估计**该`action`的`Q-val`的方法就是加权平均，通过返回$$1$$和$$0$$的数量，我们可以大致估计某一个`action`的收益。

$$
\begin{equation*}
\begin{split}
\hat Q_t(a)=\frac{1}{N_t(a)}\sum\limits_{\tau=1}^t r_\tau 1[a_r=a]
\end{split}
\tag{3}
\end{equation*}
$$

其中$$1(\cdot)$$是一个函数，这个函数会返回一个操作的概率，这个被称为`indicator function`；$$N_t(a)$$是至今为止采取`action`的总次数，$$N_t(a)=\sum\limits_{\tau=1}^t1[a_\tau=a]$$

根据$$\varepsilon$$贪婪算法，我们依一个很小的概率$$\varepsilon$$进行采取一个随机`action`，而其他情况下我们都采用目前已知的最佳决策方法（即依概率$$1-\varepsilon$$），即$$\hat a_t^* =arg\max\limits_{a\in A}\hat Q_t(a)​$$









### Upper Confidence Bounds(UCB)

上述的随机探索让我们有机会尝试我们不了解的选项，但是正是由于随机性，我们很有可能尝试了一个已经证实的并不优秀的`action`。为了避免这样的低效率探索，一种方法是随时间减少参数$$\varepsilon$$，另一种方法就是对具有高度不确定度的选项持乐观态度，因此更倾向于尝试我们目前还不能完全确定估计收益的选项。换句话说我们应该更倾向于探索具有更大价值潜力的行动。

`2019-2-25 15:51:38`

> 隔了一天，我们先来复习一下。首先是赌博机`bandit`，依一定概率$$\theta_i$$给出`reward=1`或相反依概率$$1-\theta_i$$给出`reward=0`
>
> 然后我们在对环境不够了解的情况下会使用$$Q$$来估计$$\theta$$，前面给了详细的计算方式，$$Q(a)=\mathbb{E}[r\mid a]=\theta$$，这里说的$$Q$$是对一个操作选择时的依据，我们对概率的估计就是$$Q$$
>
> 目标是最大化累计收益，或者是最小化总`regret`
>
> 决策时有`exploration`和`exploitation`两个选项，最简单的方法就是$$\varepsilon$$贪婪算法，依某概率$$\varepsilon$$进行其中某一个操作，否则进行另一个。
>
> 对该算法的改进是随时间降低$$\varepsilon$$，还有一些其他方法，接下来就要讲这些。

`UCB`算法通过`reward`值的上置信度来表达这个决策方案的价值潜力，因此真实值依一个很高的概率是低于该置信上界的。

> 这里简单看了一天上置信界相关算法，感觉本科白学。
>
> 这个标题的正统翻译方法是“**上置信边界**”算法，或“**置信上界**”算法。这个算法克服了基于`exploration`策略的所有局限性，包括了解水平和次优性差距。根据噪声分布的假设，该算法有多种不同的形式。
>
> 该算法基于面对不确定性时的乐观原则，选择行动时相信未知的环境是好的。选择这样的原则的直接原因是，当我们乐观地行动时，以下两件事总有一件会发生，即要么乐观是对的，在这种情况下，学习者的未知环境是确实是优于已知部分的；另一种则是乐观并不正当，未知的环境往往更加险恶。在后面这种情况下，`agent`选择一个可能会得到很高`reward`的`action`，然而实际上可能并没有。当这种情况经常发生的时候，我们的决策者就会发现这个收益并不高，进而不再选择。
>
> 这个算法可能会得到一个较好的结果（eventually get things right），但是并不能直观看出这个算法将得到最优结果（actually be a good algorithm among all consistent ones）。
>
> 首先我们回忆一下，如果一组随机变量$$X_1,X_2,...,X_n$$是相互独立的零均值1-亚高斯变量（这里说到的亚高斯分布到底有啥用我是不知道的，听都是第一次听说这个词），我们估计他们的均值为
> $$
> \begin{equation*}
> \begin{split}
> \hat\mu=\sum\limits_{t=1}^n\frac{X_t}{n}
> \end{split}
> \tag{4.1}
> \end{equation*}
> $$
> 那么
> $$
> \begin{equation*}
> \begin{split}
> \mathbb P(\hat\mu\geq \varepsilon)\leq e^{-\frac{n\varepsilon^2}{2}}
> \end{split}
> \tag{4.2}
> \end{equation*}
> $$
> 我们令等式右侧为$$\delta$$，并解出$$\varepsilon$$得到
> $$
> \begin{equation*}
> \begin{split}
> \mathbb P\left(\hat\mu\geq \sqrt{\frac{2}{n}\log\left(\frac{1}{\delta}\right)}\right)\leq \delta
> \end{split}
> \tag{4.3}
> \end{equation*}
> $$
> 当我们的













# Reference

[1]  [Reinforcement Learning: An Introduction](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RLIntro2nd2018.pdf)

[2] [UCB Algorithm](http://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)