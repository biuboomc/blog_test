# 以循环为桥：循环Transformers能否弥合输出与表征的差距?

#### 引言：表达滞后思维，认知滞后感知

你是否发现，大模型总是存在一种“表达滞后”的现象？

具体而言，如果我们把大模型的能力拆解开来，通常会看到三个有趣的层级：

对策略模型 $\pi$，一个任务 ($T$) 和验证其是否完成的验证器 ($V$) 有：

1. **任务执行 (Task Performance, $P_{TP}(\pi) \triangleq \Pr_{A\sim\pi(\cdot| T)}[V(T,A)=1]$)**：让模型做一道题，它可能做错了。
2. **自我验证 (Self-Verification, $P_{SV}(\pi,A,s) \triangleq \Pr[SV_s(T,A)=V(T,A)]$, $s$ 是某种验证策略)**：虽然它做错了，但如果你让它检查自己的答案，它往往能指出“这不对”。这说明它的**语言验证能力 (SV)** 往往强于 **执行能力 (A)**。
3. **潜意识/内部表征 (Representation readout, $P_{RR}(\pi,A,l,g) \triangleq \Pr[ RR_l(T,A)=V(T,A) ]$，其中 $RR_l(T,A) \triangleq g(h_l(T,A))$ 为基于第 $l$ 层表征训练的监控器 $g$ 的验证结果)**：如果我们跳过语言输出，直接把探针（Probe）插进模型的神经元里看它的激活状态，会发现它“心里”其实知道正确答案的相关特征。

现有的研究几乎都在印证一个不等式：

$$
\sup_{\pi} P_{TP}(\pi) \le \sup_{\pi,A,s} P_{SV}(\pi,A,s) \le \sup_{\pi,A,l,g} P_{RR}(\pi,A,l,g)
$$

|      ![loop](./loop_cn.svg)      |
| :------------------------------: |
| 语言模型实践中三个层次的性能间存在差异 |

换句话说，模型是一个**“心里有数” (RR)，但“嘴上说不明白” (SV)，导致“手头做不对” (TP)** 的矛盾体。推理与思维链的出现，给出了一条让 $P_{TP}$ 逐渐追上 $P_{SV}$ 的路线。**但谁来解决 $P_{SV}$ 和 $P_{RR}$ 之间的脱节？** 为什么模型深层的直觉无法完美映射到显性的语言上？


这正是我们试图探索的焦点。

为了缩小“直觉”与“表达”的鸿沟，我们把目光投向了 **Looped Transformer**。



#### 破局者：Looped Transformer 的“内省”潜力

不同于传统 Transformer “一条路走到黑”的前馈结构，Looped Transformer 允许输入在层与层之间循环迭代。你可以把它想象成模型在开口说话前，把这句话在脑子里**“反刍”**了几遍。

> [!TIP]
>
> **什么是Looped Transformer**
>
> 如果说 CoT 是通过**序列长度**换取思考时间，Scaling Laws 是通过**参数规模与数据**堆砌智能，那么 Looped Transformer 则开启了 Scaling 的另一个维度：**深度**。
>
> 其核心理念在于**“权重共享与递归处理”**。它不再像传统模型那样急于将每一层的表征解码为离散的 Token，而是将中间层的隐状态再次扔回模型内部，使用相同的权重参数进行多轮循环（Loop）。
>
> 你可以将其类比为人类的“深思”：在开口说话（解码）之前，通过在脑海中反复迭代同一个概念，利用有限的脑容量（参数），换取更深层的推理质量。
>
> 目前对该架构的探索百花齐放，研究重心主要聚焦于两个维度：**“怎么Loop”**与**“何时Loop/Loop几次”**。
>
> **怎么Loop**
>
> - **PonderLM** [1,2] 将原本的离散语言预测转化为基于预测概率对所有 Token 的 Embedding 进行加权求和，以此迭代改进预测结果。
> - **Retrofitting-Recurrencet** [3] 选择了一条改造的路线，它改造现有模型，仅让中间的若干层变成循环层，而保持浅层和深层用于编码与解码。
> - **THINK-AT-HARD** [4] 在迭代过程中引入 LoRA 适配器与 Duo-causal attention，手把手地“教”模型学会如何利用循环带来的额外计算量。
>
> **在哪 Loop？**
>
> > 并不是所有的词都值得“深思熟虑”。像“的”、“是”这样的停用词不需要 Loop，而复杂的逻辑节点则需要多次迭代。 
>
> - **Google 的 MoR** [5] 引入了路由机制，动态学习每个 Token 应该分配多少计算预算（即循环几次）。
> - **THINK-AT-HARD** 使用 Oracle Iteration Policy，根据 Base Model 的 SFT 变体是否预测正确，来倒推该 Token 是否需要进入循环模式。
> - **SEED 的 OURO 模型** [7] 引入了**“早停机制”**。它在预训练阶段就对最大深度内的每次 Loop 进行了训练，利用熵正则化损失和专门设计的自适应门控训练，让模型学会自主判断输出时机。
>
> 在本次实验中，我们选择了 **OURO** 模型作为研究对象。这不仅是因为它在训练规模和性能上的优异表现，更因为其对 **vLLM** 推理框架的良好支持，使我们能够更高效地在验证关于“直觉”与“表达”的假设。

我们期待：这种 Loop 过程不仅仅是计算量的堆叠，也可以是一种**内化了的“探针”**。它迫使模型对自己的表征流进行额外的非线性处理，从而赋予模型一种**“内省”**的能力——即敏锐地捕捉自身思维流的细微变化，并将深层的隐性表征（RR）更好地“翻译”为显性的语言输出（SV）。

带着这个愿景，我们使用 Looped Transformer 架构进行了实验，结果既令人振奋，又充满困惑。



#### 发现一：Gap 确实在缩小，但代价是什么？

我们在数学和安全场景下，对比了“语言监控器（Language Monitor）”和“表征监控器（Representation Monitor）”的性能差异。

我们期待看到的是：随着 Loop 次数的增加，模型想得越久，语言表达（SV）就越接近内部表征（LA）。

**实验结果印证了一半：** 随着 Loop 增加，二者之间的 Gap（表征探针性能 - 语言判断性能）确实在缓慢下降。

- **好消息：** 语言验证的准确率确实随着 Loop 呈上升趋势。模型确实因为“深思熟虑”而变得更能用语言解释清楚问题。

|   ![cot](./cot.svg "语言验证能力随 Loop 增加而增长")   |
| :----------------------------------------------------: |
|             语言验证能力随 Loop 增加而增长             |
|       ![gap](./gap.svg "Gap 随 Loop 增加而下降")       |
| Gap 随 Loop 增加而下降 |

- **坏消息（或者说有趣的代价）：** Gap 的缩小，部分原因竟然是因为**表征监控的性能下降了**。

| ![lp Logo](./lp.svg "表征监控的性能随 Loop 增加而下降") |
| :-----------------------------------------------------: |
|            表征监控的性能随 Loop 增加而下降             |

这似乎暗示了一个残酷的权衡：**Loop 的过程虽然整理了思路，但也造成了原始信息的熵减或丢失。** 如果“想得太久”的代价是磨损了“直觉”的敏锐度，这是否得不偿失？这是我们需要深思的问题。



#### 发现二：薛定谔的“内省”——它真的知道自己在想什么吗？

为了验证模型是否真的具备“内省”能力，我们参考 Anthropic 的实验设置 [8]，做了一个**“思维植入”测试**。

我们在模型“思考”的过程中，强行注入一个特定的词向量（Concept Vector），然后看模型能不能意识到：“嘿，刚才有个奇怪的概念钻进我脑子里了。”

结果非常反直觉：

| ![identify](./identify.svg "模型在接近输出时才能更好的识别注入的概念") |
| :----------------------------------------------------------: |
|           模型在接近输出时才能更好的识别注入的概念           |

- 在前几次 Loop 中，无论我们注入什么向量，模型都视而不见，仿佛处于“无意识”的状态。
- **只有在最后一次 Loop 中**，被注入的表征才能被模型识别出来。

这与我们期待的“连续内省”完全不符。这说明，即使架构上是循环的，模型对内在语义的处理范围似乎仍然是局部的、短视的。它并没有在每一次循环中都审视自我，而只是在最终输出的关头才“醒”过来。



#### 写在最后

我们的实验仅针对 Looped Transformers 的一种特定实现方式进行了初步探索。因此，所观察到的局限性，例如表示的退化或对表示的缺乏持续监测，不应被解读为本身普遍存在的缺陷。相反，我们仍然认为Looped Transformers为模型语言行为与内部表征的对齐提供了一个极具潜力的方向。

然而我们的实验的确揭示了语言与表征之间复杂的关系：**Loop 确实能让模型“说”得更好，但未必能让它“想”得更清楚。** 这种“表达”与“表征”的非同步进化，或许正是通往更好对齐以及更高级智能的道路上必须跨越的障碍。我们希望本报告中的实证观察结果能为该领域的未来研究提供有价值的启发，随着训练目标的改进和架构的优化，我们相信未来的迭代发展能够克服这些障碍。



关于对问题的建模、实验的详细数据等，我们在以下[Report](https://github.com/biuboomc/L-A-B/blob/main/Loop_blog_report_0114.pdf)中进行了更完整的讨论。



@article{,

  title={},

  author={},

  journal={URL https://github.com/biuboomc/L-A-B/blob/main/Loop_blog_report_0114.pdf}

}



[1] Zeng B, Song S, Huang S, et al. Pretraining Language Models to Ponder in Continuous Space[J]. arXiv preprint arXiv:2505.20674, 2025.

[2] Zeng B, Li H, Song S, et al. PonderLM-2: Pretraining LLM with Latent Thoughts in Continuous Space[J]. arXiv preprint arXiv:2509.23184, 2025.

[3] McLeish S, Li A, Kirchenbauer J, et al. Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence[J]. arXiv preprint arXiv:2511.07384, 2025.

[4] Fu T, You Y, Chen Z, et al. Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models[J]. arXiv preprint arXiv:2511.08577, 2025.

[5] Bae S, Kim Y, Bayat R, et al. Mixture-of-recursions: Learning dynamic recursive depths for adaptive token-level computation[J]. arXiv preprint arXiv:2507.10524, 2025.

[7] Zhu R J, Wang Z, Hua K, et al. Scaling latent reasoning via looped language models[J]. arXiv preprint arXiv:2510.25741, 2025.

[8] Lindsey J. Emergent introspective awareness in large language models[J]. arXiv preprint arXiv:2601.01828, 2026.




