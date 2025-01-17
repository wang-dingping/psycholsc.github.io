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

其实经过了一小段时间的研究后我发现，电信这边常用的方法与模型，都是将问题进行完整的建模后运算推导并简化这样的思路，但是凡是与机器学习与深度学习这样十分依赖于优化的学科有关的运算，主要思想都是找出一个理论上可行的传递系统族，然后调整系统的参数使系统逼近最优系统。由于这些问题本身确实是难以求解最优解的，因此采用系统调整的方法也不失为一种可行方法。

另外本次选择的方向（P.S.电子信息类研究计划政策要求）是一个大方向，因为有些理论进行深究会发现，不仅有完善的数学体系，而且在短时间内很难完全理解。里面确实免不了出现这种内容，我的理解过于浅显也是一件不可避免的事情。

以下是几个前几年兴起于NLP的NN模型，这里进行理论介绍与复现。

话说回来，这可能是目前网上能看到的最浅显也最完整的RNN说明了吧。除了简单介绍了几种RNN模型以外，还连带着说了一下机器学习的相关理论。




## Vanilla RNN

Vanilla原义是香草，此处含义是`without fancy stuff`，即最原初的版本。

本来想参考GitHub上的某个项目写，结果这个项目的主人本身也不懂什么是vanilla RNN，甚至于Gradient Decent的时候求导计算都是错的。下面还是通过简单计算推导来说明一下Vanilla RNN。

首先是RNN的最基础模型。RNN是Recurrent Neural Network，国内说法很多，有的地方翻译为循环神经网络，有的地方翻译为递归神经网络。稍加调研后发现，国内网络翻译存在较强的误导性。较为标准的可参考翻译为，循环神经网络`Recurrent NN`和递归神经网络`Recursive NN`。一般的，并不建议将递归网络缩写为`RNN`（*Deep Learning 10.6*），以下所说的RNN均为循环网络。

RNN是一种结构稍显特殊的网络，相比传统的深度网络和卷积网络这样的结构，RNN的结构并不是简单的层状结构，而是一种类似循环的结构，相比于普通的全连接网络或者CNN，RNN更适合于处理连续模型或时序模型。

其基础结构如下

![](https://raw.githubusercontent.com/psycholsc/psycholsc.github.io/master/assets/RNN1.png)

左侧是折叠后的网络结构，右侧是展开后的结构。这个结构是目前来讲最为基础的一种。如图，下方的$$x^t$$表示的是该时刻的输入，这个输入往往是一个向量，因此写成小写的向量形式；中间始终向右传递的$$h^t$$是`hidden state`，隐藏状态，这个值在不同的`RNN Cell`中不断传递并更新。每一个Cell的计算结果经过一个**激活函数**后将得到一个输出，这个输出与我们期望的输出进行比较，用`Loss function`来描述两者的**误差**。本文中可能将误差描述为损失或错误(**E**rror)。

图中的几个加粗大写字母是矩阵，但是使用这些矩阵难免造成混淆，因此此处对矩阵进行重命名。输入后的矩阵$$U$$命名为$$W_{hx}$$，状态间转化矩阵$$W$$命名为$$W_{hh}$$，输出矩阵$$V$$命名为$$W_{hy}$$。

该过程涉及如下计算

隐藏状态$$h$$更新

$$h^t=f(W_{hx}x^{t}+W_{hh}h^{t-1}+b_h)\tag{1}$$



输出计算

$$y^t=g(W_{hy}h^t+b_y) \tag{2}$$



以上计算与个人习惯有关，运算可以理解为，每一个`cell`输入前一时刻的状态$$h^{t-1}$$，利用输入$$x^{t}$$，通过一个非线性关系计算该时刻的状态$$h^t$$，然后利用该时刻状态计算输出$$y^t$$。此过程建立了输入与输出的联系，并且对其中循环更新的隐藏状态$$h$$进行更新。

最简单的vanilla RNN应用是文字的预测，此处以字母为输入，依照输入指定长度的字符序列，预测接下来输出的字符序列。

字符的输入需要经过编码，否则不能直接参与运算。字符编码的方式很多，一般对编码的要求不同的字符有不同的编码即可。但是在较大的数据量下，编码方式会有更多要求。常见编码如下

>1. 独热码（one-hot-coding）。此编码会对字典内的所有字符进行一个编码，编码方式为，首先生成一个长度为字典大小$$m$$的零向量，然后将互不相同的一位编码为1。这样一来不但计算速度快，而且每个字符之间都是正交的。因此采用此类编码方式在本文所示的项目中十分合适。
>
>    独热码的另一个好处是有利于分类模型。这个在后面会有讨论。
>
>2. 词向量法（word2vec）。此编码方式通过一定的算法将单词向量化，最小处理单位是词。这种方法生成的词向量，在同义词之间有很强的相关性（表现为长度与方向的近似，以及相加时可以将词的表意相加而组成词组），而且是指定长度（维度）的，在字典超大的条件下可以降低内存使用。

这样，该模型输入$$x^t$$就可以用向量的方式来表达。按照上图所示，假设模型输入是$$m$$维的向量，例如

$$ x^t=\left[ \begin{matrix} x_0  \\ x_1  \\ x_2 \\ x_3 \\ ... \\ x_m  \end{matrix} \right] _{m \times 1}$$ 



由于编码为独热码，因此可以写作

$$ x^t=\left[ \begin{matrix} 0  \\ ...  \\ 1 \\ ... \\  0  \end{matrix} \right] _{m \times 1}$$

对于上面图中表示的RNN模型，输入时每一个时序输入一个向量即可。

> 举例，假设字典中共有40个字符，则矩阵长度就是40，并将40个不同的字符对应不同位置为1的向量。假设第一个字符是A，则A编码如下

$$ x_A=\left[ \begin{matrix} 1  \\ 0  \\ ... \\  0  \end{matrix} \right] _{m \times 1}$$

另外假设隐藏状态$$h$$的长度为$$n$$，则计算过程可以写成矩阵

$$ h^t=\left[ \begin{matrix} h_1^t  \\ h_2^t   \\ ... \\ ... \\ h_n^t  \end{matrix} \right] _{n \times 1}=\mathfrak F\left( \left[ \begin{matrix} ... & W_{hx}(1,j) & ... \\... & ... & ...   \\ ... & W_{hx}(i,j) & ...  \\ ... & ... & ...  \\  ... & W_{hx}(n,j) & ...   \end{matrix} \right] _{n \times m} \left[ \begin{matrix} 0  \\ ...  \\ 1 \\ ... \\  0  \end{matrix} \right] _{m \times 1}+    \left[ \begin{matrix} ... & W_{hh}(1,j) & ... \\... & ... & ...   \\ ... & W_{hh}(i,j) & ...  \\ ... & ... & ...  \\  ... & W_{hh}(n,j) & ...   \end{matrix} \right] _{n \times n} \left[ \begin{matrix} h_1^{t-1}  \\ h_2^{t-1}   \\ ... \\ ... \\ h_n^{t-1}  \end{matrix} \right] _{n \times 1}        \right) $$ 

以上是对隐藏状态的更新

相似的，输出可以写作

$$ y^t=\left[ \begin{matrix} \hat y_1^t  \\ \hat y_2^t   \\ ... \\ ... \\ \hat y_m^t  \end{matrix} \right] _{m \times 1}=\mathfrak G\left( \left[ \begin{matrix} ... & W_{hy}(1,j) & ... \\... & ... & ...   \\ ... & W_{hy}(i,j) & ...  \\ ... & ... & ...  \\  ... & W_{hy}(m,j) & ...   \end{matrix} \right] _{m \times n} \left[ \begin{matrix} h_1^t  \\ h_2^t   \\ ... \\ ... \\ h_n^t  \end{matrix} \right] _{n \times 1}   \right) $$



通过上述计算，我们就完成了一个`cell`的计算。

上述操作将在不同的`cell`之间传播，整个网络相当于一个时序处理模型。现在我们需要调整系统，因为各个矩阵的权值都是初始化的时候自动生成的，因此并不能直接输出我们期待的结果。因此要设计算法，能够根据我们期待的输出结果对系统进行调整。最常见的能够动态调整的算法就是梯度下降法，这里会从梯度下降的角度介绍RNN的优化过程。

梯度下降的计算方法是，让目前的参数向错误减小的梯度方向进行一定步长的移动。首先我们需要规定一个能够衡量参数误差的函数，常命名为$$loss$$或$$E$$。简单常用的衡量方法有很多，取其中的两种。

>一、 **直接比较**，采用均方误差方法衡量，
>
>$$E=\frac{1}{2}(y-t)^2$$
>
>或
>
>$$MSE=\frac{1}{2m}\sum\limits_{i=1}^m(y-\hat y)^2$$
>
>其中$$t$$是期待的正确输出，$$y$$是系统真实输出。采用这种方法，只需要设计算法让$$E$$向$$0$$逼近即可。二次方项在求导后会消失，计算起来也很方便。这个函数也是一个凸函数，因此在区间内的局部最优解就是全局最优解。

>二、 **交叉熵比较**，衡量两个概率分布的不同（距离）
>
>$$H(p,q)=-\sum\limits_i p(x_i)log\: q(x_i)$$
>
>或写作
>
>$$loss^t=-\frac{1}{n}\sum\limits_{i} \left[y_i\: log(\hat y_i^t) + (1-y_i)\: log(1-\hat y_i^t)  \right]$$
>
>交叉熵的一般描述是这样的，在此处我们可以认为$$p\:\&\:q$$是输出变量和实际变量的概率分布。至于为什么说是概率分布，因为输出层常常会做一个`Softmax`操作，这个操作可以将数值转化为一个具有概率意义的值，也可以说是一种归一化方法。
>
>注意这里的计算结果是一个数而非矩阵或向量，求和运算是针对向量中的所有元素的。
>
>得到概率分布后可以直接和独热码进行交叉熵比较，因为独热码也是一种概率分布，但这种概率分布是唯一的，在指定输出为$$1$$，其他位置都是$$0$$。这时损失函数是可以退化的。
>
>另外在`Softmax`激活函数输出后常用的损失函数实际上就是交叉熵函数，也被称为**对数**损失函数，因为可以降低Softmax的计算难度。表达式为
>
>$$loss^t = -y_i^tlog(\hat y_i^t)$$
>
>这里的$$y^t_i$$一般是取值为$$1$$的那一项，所以也常常不写。这样一比其实可以发现对数似然损失函数和交叉熵损失函数在这种条件下等价。
>
>> 交叉熵和对数损失函数应当是同一种损失函数。我们来稍微讨论一下熵这个东西
>>
>> **熵**是给定概率分布时的**不确定性**的度量，通信领域用于衡量**信息量**。以二分类器为例，假设概率分布为$$\{0,1\}$$，显然这个分布时完全确定的，并不存在什么不确定性，我们希望这个时候熵的数值为0；假设概率分布为$$\{0.5,0.5\}$$，则对于某一时刻分类器的输出结果，不确定性是最大的。分类的结果既可能是1，也有可能是2。实际上《通信原理》、《**信息论**》等课本中已经对熵有过明确地计算与定义，即
>>
>> $$H(x)=-\sum\limits_{i=1}^N p(x_i)log\left(p(x_i) \right)$$
>>
>> ---
>>
>> 为了衡量两个时间或两组概率分布的不同，我们常采用**KL散度**的方法。
>>
>> KL散度的定义为，对于两个事件A和B，事件的差别为
>>
>> $$D_{KL}(A\mid B)=\sum\limits_i P_A(x_i)log(\frac{P_A(x_i)}{P_B(x_i)})=\sum\limits_i \left[P_A(x_i)log(P_A(x_i))-P_A(x_i)log(P_B(x_i))\right]$$
>>
>> 左侧看起来就是熵的表达式。值得一提的是$$D_{KL}(A\mid B) \neq D_{KL}(B\mid A)$$，即这个散度并不是一个对称的距离。其取值与选取参照有关。
>>
>> ---
>>
>> 交叉熵其实就是上式中靠右的那一项。
>>
>> $$D_{KL}(A\mid B)=-H(A)+H(A,B)$$
>>
>> 即
>>
>> $$H(A,B)=-\sum\limits_{i=1}^N P_A(x_i)log\left(P_B(x_i) \right)$$
>>
>> 由于机器学习中的参考分类结果是一个常量，因此KL散度和交叉熵的定义经常是等价的，描述的都是两组概率密度的“距离”。由于交叉熵表达式还要更简单一些，计算冗余小一些，因此选择交叉熵进行损失函数的表达式。
>>
>> ---
>>
>> 那么所谓的机器学习是怎么学习的呢，李航在《统计学习方法》中介绍到，一般的学习问题采用生成模型和判别模型，虽然方法不一样，但是最终目的都是通过海量数据学习到条件概率分布、联合概率分布，实际上学习的目标就是**概率分布**。
>>
>> 损失函数就是用于评价学习结果和训练数据（由于并没有真实的概率分布情况，只能使用数据集的充分统计量来代替）之间的概率分布的差异的，最终目的是最小化差异使其一致，这样就可以得到一个近似正确的概率分布结果。
>>
>> 当然这里还有一些过/欠拟合、泛化等问题，这些均不是本次讨论的重点。
>
> 

不失普遍性，假设采用激活函数$$f(x)$$为`tanh`，采用输出层激活函数$$g(x)$$为`softmax`，损失函数为交叉熵（对数似然函数）（采用`Softmax`时常用交叉熵降低求导的难度、且有较强的逻辑对应含义）。

> 说到`Softmax`就不得不说一说它的由来（禁止开花）。这个在吴恩达的课程中实际上做了详细的介绍，此处稍微整理一下相关内容。
>
> 在二分类（逻辑回归）模型中，我们采用了`Sigmoid`函数，其实目的也很简单。如果采用回归方法，对于分类模型来说，范围肯定不能超过$$(0,1)$$，否则没有意义。因此调整函数，从线性函数换成了`Sigmoid`函数。我们发现这个函数不仅范围是$$(0,1)$$，而且可以用自己来表示自己的导数，便于计算机运算，因此采用该函数作为新的回归函数。
>
> ---
>
> 由于这属于机器学习的范畴了，因此在这里不做详细说明，但是可以看出深度学习确实是机器学习的一个重要分支。另外这些内容的推导我还不是十分懂来着，吴恩达老师的`CS229`确实很有难度。
>
> 对于多分类模型，往往是采用`Softmax`函数处理结果。
>
> 可以证明机器学习中的常用函数都是可以通过`Generalized Linear Models`导出的，中文应称作**广义线性模型**。其中提到的指数函数族，则可以用于表示许多不同形态的随机变量。
>
> 指数函数族的表达式是
>
> $$p(y;\eta)=b(y)e^{\eta^TT(y)-a(\eta)}$$
>
> 其中$$\eta$$被称为分布的**特性参数**或**自然参数**，$$T(y)$$是充分统计量，$$a(\eta)$$是对数分割函数，使$$e^{-a(\eta)}$$成为一个归一化常数，规定其取值可以令$$\sum p(y;\eta)=1$$。规定了$$T,a,b$$的模型可以表示一族分布，在改变$$\eta$$时分布随之在族内改变。伯努利分布（两点分布）和高斯分布都是指数函数族内的一种特殊情况。例如两点分布
>
> $$p(y;\phi)=\phi^y(1-\phi)^{1-y}=e^{ylog(\phi)+(1-y)log(1-\phi)}=e^{ylog(\frac{\phi}{1-\phi})+log(1-\phi)}$$
>
> 以上运算时先取对数将乘性与指数项分离，然后取自然指数。其中$$\phi$$的定义显然就是$$y=1$$时的概率。
>
> 其中
>
> $$T(y)=y$$
>
> $$a(\eta)=-log(1-\phi)=log(1+e^\eta)$$
>
> $$b(y)=1$$
>
> 继续推一步我们会发现$$\phi = \frac{1}{1+e^{-\eta}}$$，这个函数就是做逻辑回归的`Sigmoid`函数。
>
> 我们尝试对逻辑回归进行GLM建模。GLM建模有三个基本**假设**（assumption）
>
> 1. $$(y\mid x;\theta)\sim ExponentialFamily(\eta) $$，即给出$$x;\theta$$时，$$y$$的分布服从指数函数族中的某种随$$\eta$$变化的分布
> 2. 给出$$x$$，我们希望预测(predict)$$T(y)$$，这个我们前面说过，是$$y$$的充分统计量。在大多数问题中我们都取$$T(y)=y$$，那么就意味着我们将预测的是$$y=h(x)$$的输出，其中$$h$$是我们算法中学习到的函数，他需要满足$$h(x)=E[y\mid x]$$，即给定$$x$$，输出$$y$$的统计平均结果。
> 3. 我们前面没能解释清楚的$$\eta$$实际上是$$\eta=\theta^Tx$$，与输出是线性关系的。
>
> 假设是一个二分类模型，$$y$$的值已经被严格规定到$$\{ 0,1 \}$$。我们很自然的会采用**伯努利分布**对输出建模。在前面我们已经算到
>
> $$\phi = \frac{1}{1+e^{-\eta}}$$
>
> 又根据伯努利分布的特点，如果
>
> $$(y\mid x;\theta)\sim Bernoulli(\phi) $$
>
> 其中$$\theta$$是系数参数，那么期望是
>
> $$E[y\mid x;\theta]=\phi$$
>
> 估计函数就可以根据上述假设第二条写作
>
> $$h_\theta (x)=E[y\mid x;\theta]=\phi=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\theta^Tx}}$$
>
> 这个正是机器学习中的最基础的二分类模型中使用的Sigmoid函数，或者叫Logistic函数，~~种群密度自然增长也是这条曲线来着~~。预测时根据输入$$x$$输出结果是规定在$$(0,1)$$内的，而且具有概率意义，因为$$\phi$$的定义即为输出$$y=1$$的概率。
>
> ---
>
> 相同的方法可以推一下多分类模型。
>
> 对于多分类问题，假设$$y$$可以分为$$k$$种不同类别，使用GLM建立模型。根据流程，首先确定一个分布。由于是多分类问题，因此可以采用多项分布（多项式分布）。
>
> >简单说明，伯努利分布是两点分布，两种状态的单次试验问题；二项分布是将两点分布推广到多次试验，而多项式分布则将二项分布推广到多种状态。
>
> 多项式分布当然也是可以使用指数函数族进行表示的。首先为了参数化分类结果，目前常用的手段是采用一个列向量作为输出结果，描述方法为
>
> $$\hat y=\left[ \begin{matrix} \phi_1^t  \\  \phi_2^t   \\ ... \\ ... \\  \phi_k^t  \end{matrix} \right]$$
>
> 我们可以看到$$\phi_i$$的定义仍然是概率意义。不过实际上这些参数冗余了。例如我们在伯努利分布中仅使用了一个参数$$\phi$$，但实际上我们是二分类器，因为第二个概率就是$$1-\phi$$。这里可以采用相同的方法，将概率表示为$$\phi_1,\phi_2,...,\phi_{k-1}$$，这样第$$k$$类就可以用$$\phi_k=1-\sum\limits_{i=1}^{k-1}\phi_i$$。这个方法是仿照伯努利分布来的，只是需要稍微说明。
>
> 这样，用于与输出比较的结果$$T(y)$$也需要进行修改，原来只需要表示为$$T(y)=0$$或$$T(y)=1$$，这里考虑到多分类模型，就需要描述为
>
> $$T(1)=\left[ \begin{matrix} 1  \\  0   \\ ... \\ ... \\  0  \end{matrix} \right];T(2)=\left[ \begin{matrix} 0  \\  1   \\ ... \\ ... \\  0  \end{matrix} \right];T(k-1)=\left[ \begin{matrix} 0 \\  0   \\ ... \\ ... \\  1  \end{matrix} \right];T(k)=\left[ \begin{matrix} 0  \\  0   \\ ... \\ ... \\  0  \end{matrix} \right]$$
>
> 严格的说这里就不再是$$T(y)=y$$了。这时候$$T(y)$$被描述为一个$$k-1$$维向量，通过上述方式对分类结果进行描述，当分类为对应位置时，该位置值为1，其他位置为0。多项式分布在$$y$$取离散值时的概率分布为
>
> $$p(y;\phi)=\phi_1^{true(y==1)} \phi_2^{true(y==2)} ...\phi_k^{true(y==k)}$$
>
> 其中$$true(y==k)$$的含义是$$y$$是否为$$k$$。若为真则取该函数为1，否则为0。根据分类结果，其实可以写作$$T(y)_i$$
>
> 根据上面的描述，最后一项是可以由前面的判断表达的，即
>
> $$p(y;\phi)=\phi_1^{true(y==1)} \phi_2^{true(y==2)} ...\phi_k^{1-\sum\limits_{i=1}^{k-1} true(y==i)}$$
>
> 类似的进行对数与指数运算操作（为了向指数函数族靠近）可以得到
>
> $$p(y;\phi)=exp\left[T(y)_1 log(\phi_1)  +T(y)_2 log(\phi_2) +...+(1-\sum\limits _{i=1}^{k-1}T(y)_i) log(\phi_k)  \right]$$
>
> 展开最右侧求和的括号可以得到
>
> $$p(y;\phi)=exp\left[T(y)_1 log(\phi_1/\phi_k)  +T(y)_2 log(\phi_2/\phi_k) +...+T(y)_{k-1} log(\phi_{k-1}/\phi_k)+ log(\phi_k）  \right]$$
>
> 将上述结果表示成指数函数族可以得到
>
> $$p(y;\phi)=b(y)exp\left(\eta^TT(y)-a(\eta)\right)$$
>
> 其中
>
> $$b(y)=1;\eta=\left[ \begin{matrix} log(\phi_1/\phi_k)  \\  log(\phi_2/\phi_k)   \\ ... \\ ... \\  log(\phi_{k-1}/\phi_k)  \end{matrix} \right];a(\eta)=-log(\phi_k)$$
>
> 根据上述结果
>
> $$\eta_i=log(\phi_i/\phi_k)$$
>
> 对于$$i=k$$我们定义$$\eta_k=0$$
>
> 现在推导概率$$\phi$$的表达式
>
> $$e^{\eta_i}=\phi_i/\phi_k$$
>
> $$\phi_k e^{\eta_i}=\phi_i$$
>
> 概率之和为1，则
>
> $$\phi_k \sum\limits_{i=1}^k e^{\eta_i}=1$$
>
> $$\phi_k =\frac{1}{ \sum\limits_{i=1}^k e^{\eta_i}}$$
>
> 回代得到
>
> $$\phi_i = \frac{e^{\eta_i}}{ \sum\limits_{j=1}^k e^{\eta_j}}$$
>
> 这个函数就被称为**Softmax**，在$$\eta=\theta^T x$$的假设3下，
>
> $$p(y=i\mid x;\theta)=\phi_i=\frac{e^{\theta_i^T x}}{ \sum\limits_{j=1}^k e^{\theta_j^T x}}$$
>
> 其他内容在此不再赘述。

以下开始反向传播推导过程。反向传播（Back Propagation）实际上是根据已知输出进行系统修正的过程，一般的需要通过链式法则进行求导计算，根据导数（梯度）来决定每一个参数应该向什么**方向**做多大**步进**的修正。经过不断的迭代修正，就能得到正确结果。最重要的步骤就是求导。

先把**前向传播**运算拉下来

---

隐藏状态$$h$$更新

$$h^t=f(W_{hx}x^{t}+W_{hh}h^{t-1}+b_h)\tag{1}$$

输出计算

$$y^t=g(W_{hy}h^t+b_y)\tag{2} $$

---

指定时序$$t$$，计算此时的反向传播参数。由于是时序模型，此时也是针对时间的反向传播。

我们定义总损失

$$E=\sum\limits_{t}E^t\tag{3}$$

最简单的梯度是输出层的梯度。根据计算公式

$$\hat y^t=g(W_{hy}h^t+b_y)=Softmax(W_{hy}h^t+b_y)$$

则输出层**偏置**的梯度为

$$\frac{\partial E}{\partial b_y}=\frac{\partial }{\partial b_y}\sum\limits_{t}E^t=\sum\limits_{t}\frac{\partial E^t}{\partial b_y}$$

根据链式法则

$$\sum\limits_{t}\frac{\partial E^t}{\partial b_y}=\sum\limits_{t}\frac{\partial E^t}{\partial \hat y^t} \frac{\partial \hat y^t}{\partial z^t}\frac{\partial z^t}{\partial b_y}$$

第一个求导是关于损失函数的求导，第二个求导是关于`Softmax`的求导，第三层就是Softmax内部的求导过程。这样根据链式法则分别求导可以得到

$$\sum\limits_{t}\frac{\partial E^t}{\partial \hat y^t} \frac{\partial \hat y^t}{\partial z^t}\frac{\partial z^t}{\partial b_y}=\sum\limits_{t}-\frac{1}{\hat y^t} \frac{\partial \hat y^t}{\partial z^t}\frac{\partial z^t}{\partial b_y}=\sum\limits_{t}\left[ -\frac{1}{\hat y^t} \frac{e^{z_i}\sum\limits_j e^{z_j} -(e^{z_i})^2}{(\sum\limits_j e^{z_j})^2}·1\right]$$

最终可以求得向量化导数为

$$\sum\limits_t (\hat y^t -y^t)\tag{4}$$

这里求导只有一个需要注意的，就是求导过程中**分母不能只当做是一个常数**，而是应该注意到分母的**求和项中也有我们的自变量项**，因此求导时需要使用除法求导的法则。

> 注意到RNN参数共享，此处$$b_y$$并不需要写作$$b_y^t$$，倒是可以在需要的时候写$$(b_y)_i$$，但很多模型中由于考虑到计算方便，往往不会将$$b_y$$设计为向量，而是常数。

同理可以计算关于$$W_{hy}$$的导数，此时的不同点只在上述求导中的第三项

$$\sum\limits_{t}\frac{\partial E^t}{\partial \hat y^t} \frac{\partial \hat y^t}{\partial z^t}\frac{\partial z^t}{\partial W_{hy}}=\sum\limits_{t}-\frac{1}{\hat y^t} \frac{\partial \hat y^t}{\partial z^t}\frac{\partial z^t}{\partial W_{hy}}=\sum\limits_{t}\left[ -\frac{1}{\hat y^t} \frac{e^{z_i}\sum\limits_j e^{z_j} -(e^{z_i})^2}{(\sum\limits_j e^{z_j})^2}·(h^t)^T\right]$$

向量化结果为

$$\sum\limits_t (\hat y^t -y^t)·(h^t)^T\tag{5}$$



---

这里分页，因为后面的计算变得复杂了许多。观察计算表达式可以发现，如果计算关于$$W_{hh}$$和$$W_{hx}$$的表达式的导数，就会发现，对于$$t$$时刻的输出结果，应由两部分组成，分别为$$t$$时刻的梯度与$$t+1$$时刻的梯度。这是因为在前向传播计算时，$$t$$时刻状态$$h^t$$不仅影响该时刻的输出$$\hat y^t$$，还会影响下一时刻的输出$$\hat y^{t+1}$$。当然这还都只是从表达式直接得到的，考虑到状态在网络中不断传递，实际上影响是不断传递的。计算时根据前向传播表达式计算的结果就可以直接推导整个过程。

我们在这里把两个表达式写出来，一个是该时刻的损失

$$E^t=-log\left[ Softmax\left( W_{hy}tanh(W_{hx}x^t +W_{hh}h^{t-1} +b_h )  +b_y \right) \right]\tag{6}$$

为方便起见，此处写作

$$E^t=-log\left[ Softmax\left( W_{hy}h^t +b_y \right) \right]\tag{6}$$

一个是下一时刻的损失

$$E^{t+1}=-log\left[ Softmax\left( W_{hy}tanh(W_{hx}x^{t+1} +W_{hh}h^{t} +b_h )  +b_y \right) \right]\tag{7}$$

可以看出总损失$$E$$与两个方向的导数有关，因此计算的时候可能会考虑到两个方向的计算。首先计算较为简单的$$b_h$$的导数。$$b_h$$就是$$h^t$$计算中的一个共享参数，计算时采用链式法则可以写作

$$\frac{\partial E}{\partial b_h}=\frac{\partial \sum\limits_{t}E^t}{\partial b_h}=\sum\limits_{t}\frac{\partial E^t}{\partial b_h}=\sum\limits_{t}\frac{\partial E^t}{\partial h^t}\frac{\partial h^t}{\partial b_h} $$



这里比较复杂的就是$$h^t$$部分的处理，前面说了存在两种方法，计算时主要是考虑$$\frac{\partial E}{\partial h^t}$$的特殊性。

这里假设一个

$$\delta^{t}=\frac{\partial E}{\partial h^t}=\frac{\partial E^t}{\partial h^t}+\frac{\partial E^{t+1}}{\partial h^{t+1}}\frac{\partial h^{t+1}}{\partial h^{t}}\tag{8}$$

由于每一次计算时刻$$t$$的导数时都会导致两项结果，因此可以通过这种递推方式求得所有层的结果。

另外，实际上效果应该是无穷传递的，但是这样下去不会得到解析解，因此求导的假设是其它层已知时求导该层。前面一项前面求过了

$$\frac{\partial E^t}{\partial h^t}=W_{hy}^T (\hat y^t-y^t)\tag{9}$$

后面一项计算时有些需要考虑的内容，一个是$$tanh$$的导数，另一个是`Hadamard Product`问题。

按照求导规则，激活函数的导数一般是与之前结果做**哈达玛积**，`tanh`的导数一般也采用其本身来表示，即$$(tanh(x))'=1-(tanh(x))^2$$，计算结果就可以表示为

$$\frac{\partial E^{t+1}}{\partial h^{t+1}}\frac{\partial h^{t+1}}{\partial h^{t}}=W_{hh}^T \frac{\partial E^{t+1}}{\partial h^{t+1}}\odot(1-(h^{t+1})^2)$$

哈达玛积在解决问题时比较复杂，我们直接写成矩阵的乘积比较方便，处理方法就是将两个做哈达玛积的$$n\times 1$$向量的前者写成对角阵，即$$diag(...)$$

$$W_{hh}^T diag\left((1-(h^{t+1})^2)\right)\frac{\partial E^{t+1}}{\partial h^{t+1}}=W_{hh}^T diag\left((1-(h^{t+1})^2)\right)\delta ^{t+1} \tag{10}$$

所以

$$\delta^t=W_{hy}^T (\hat y^t-y^t)+W_{hh}^T diag\left((1-(h^{t+1})^2)\right)\delta ^{t+1}\tag{11}$$



*对于序列中的最后一个，由于其后不再有其他序列的存在，因此它的导数中不包含上述的第二项。*

这样计算时就可以用$$\delta^t$$做简化，例如

$$\frac{\partial E}{\partial b_h}=\sum\limits_{t}\frac{\partial E^t}{\partial h^t}\frac{\partial h^t}{\partial b_h}=\sum\limits_{t}diag\left((1-(h^{t+1})^2)\right)\delta^{(t)} \tag{12}$$

$$\frac{\partial E}{\partial W_{hx}}=\sum\limits_{t}\frac{\partial E^t}{\partial h^t}\frac{\partial h^t}{\partial W_{hx}}=\sum\limits_{t}diag\left((1-(h^{t+1})^2)\right)\delta^t (x^t)^T \tag{13}$$

$$\frac{\partial E}{\partial W_{hh}}=\sum\limits_{t}\frac{\partial E^t}{\partial h^t}\frac{\partial h^t}{\partial W_{hx}}=\sum\limits_{t}diag\left((1-(h^{t+1})^2)\right)\delta^t (h^{t-1})^T \tag{14}$$

至此所有的梯度都求解完成。

---

理论上RNN可以很有效完成序列的建模与数据处理，但是实际上，从我们上面的推导（算式11）也容易看出，其梯度计算时总是前后依赖的，这样就导致计算结果会逐层减小（累乘），最终导致梯度消失（`vanish`）或爆炸（`explode`）。如果将梯度的数值分为大于1和小于1的部分，高于1的梯度累乘就会极大增加，小于1的梯度相乘就会逐渐消失。Bengio发现在维持记忆和对小扰动保持鲁棒性的条件下，必然会导致RNN参数进入参数空间中的梯度消失区，而实践证明，如果增大需要捕获的依赖关系跨度，基于梯度的优化就变得越来越困难，随机梯度下降在长度为10-20的短序列训练中会迅速归零。

另外，vanilla RNN是一种没有门控的RNN结构，也是最简单的RNN结构。相比目前常用的LSTM等模型，这种模型的“遗忘”效应较难处理。

对于处理长期依赖问题而言，人们提出的解决方法主要有

1. 时间维度的跳跃连接
2. 设置线性自连接单元（渗漏单元，`Leaky Unit`）
3. 删除连接

以上内容我还没有仔细看，并不很懂。

在本文完成之时调研的所有结果来看，实际应用中最为有效的模型是门控模型，包络基于长短期记忆（Long Short Term Memory）的`LSTM`模型和门控循环单元（Gated Recurrent Unit）`GRU`模型。



## LSTM

想了想果然还是应该在下一篇文章里写呢

---

`2019-1-22 15:34:41`

看不懂，还是看不懂，啥都看不懂，他们好厉害啊怎么啥都会。

## Reference

[1] 图源《Deep Learning》

[2] CS229 

[3] Gradient 《Deep Learning》

[4] 刘建平Pinard blog