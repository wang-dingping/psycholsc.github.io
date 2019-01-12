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

