# 隐马尔可夫
## 示例
隐马尔可夫(Hidden Markov Model, HMM)是一个包含隐含未知参数的马尔可夫的统计模型。**马尔可夫过程：给定当前状态和过去所有状态的条件下，其下一个状态的条件概率分布仅依赖于当前状态，通常具备离散状态的马尔可夫过程称之为马尔可夫链。** 因此马尔可夫链可以理解为一个有限状态机，给定了当前状态为 $S_i$, 下一时刻状态为 $S_j$ 的概率，**不同状态之间的变换的概率称之为转移概率。** 下图描述了3个状态 $S_a, S_b, S_c$ 之间转换状态的马尔可夫链。

![image.png](C:\Users\great\OneDrive\图片\MarkDown\微信截图_20220408111911.png)

## 简介
隐马尔可夫模型中包含两种序列：随机生成的状态构成的序列称之为状态序列(state sequence)，状态序列是不可被观测到的；每个状态对应的观测值组成的序列称之为观测序列(observation sequence)。
- $I = (i_1, i_2, \cdots, i_T)$ 为状态序列，其中 $i_t$ 为第 $t$ 时刻系统的状态值
- $O = (o_1, o_2, \cdots, o_T)$ 为观测序列，其中 $o_t$ 为第 $t$ 时刻系统的观测值
- 系统的所有可能的状态集合为 $Q = \{q_1, q_2, \cdots, q_N\}$
- 所有可能的观测集合为 $V = \{v_1, v_2, \cdots, v_M\}$。  

隐马尔可夫模型主要有三组参数构成：
### 状态转移矩阵
$$
A = [a_{ij}]_{N \times N} \tag{1}
$$

其中，

$$
a_{ij} = P(i_{t+1} = q_j|i_t = q_i), 1 \leq i, j \leq N, 1 \leq t \leq (T-1) \tag{2}
$$
表示 $t$ 时刻状态为 $q_i$ 的情况下，在 $t+1$ 时刻状态转移到 $q_j$ 的概率。

### 观测概率矩阵
$$
B = [b_j(k)]_{N \times M} \tag{3}
$$

其中，

$$
b_j(k) = P(o_t = v_k|i_t = q_j), 1 \leq k \leq M, 1 \leq j \leq N, 1 \leq t \leq (T-1) \tag{4}
$$
表示 $t$ 时刻状态为 $q_i$ 的情况下，观测值为 $v_k$ 的概率。

### 初始状态概率向量
$$
\pi = (\pi_i) \tag{5}
$$

其中，

$$
\pi = P(i_1 = q_i), i=1,2,\cdots,N \tag{6}
$$
表示$t=1$时刻，系统处于状态$q_i$的概率。

### 生成步骤
**初始状态概率向量 $\pi$ 和状态转移矩阵 $A$ 决定了状态序列，观测概率矩阵 $B$ 决定了状态序列对应的观测序列** ，因此马尔可夫模型可以表示为：

$$
\lambda = (A,B,\pi) \tag{7}
$$
对于马尔可夫模型 $\lambda = (A,B,\pi)$, 通过如下步骤生成观测序列 $\{o_1, o_2, \cdots, o_T \}$:

1. 按照初始状态分布 $\pi$ 产生状态 $i_1$。
2. 令 $t = 1$。
3. 按照状态 $i_t$ 的观测概率分布 $b_{i_t}(k)$ 生成 $O_t$
4. 按照状态 $i_t$ 的状态转移概率分布 $a_{i_t,i_{t+1}}$ 产生状态 $i_{t+1}, i_{t+1} = 1,2,\cdots,N$
5. 令 $t = t+1$，如果 $t < T$，则转步骤3；否则，终止

### 3个问题
#### 概率问题
给定模型 $\lambda = (A, B, \pi)$ 和观测序列 $O = \{o_1,o_2,\cdots,o_T\}$, 计算在观测序列 $O$ 出现的概率 $P(O|\lambda)$。
#### 学习问题
已知观测序列 $O = \{o_1, o_2, \cdots, o_T\}$, 估计模型 $\lambda = (A, B, \pi)$ 参数，使得在该模型下观测序列概率 $P(X|\lambda)$ 最大。即用极大似然估计的方法估计参数。
#### 预测问题
也称为解码(decoding)问题。已知模型 $\lambda = (A, B, \pi)$ 和观测序列 $O = \{o_1,o_2,\cdots,o_T\}$, 求对给定观测序列条件概率 $P(I|O)$ 最大的状态序列 $I = \{i_1,i_2,\cdots,i_T\}$。即给定观测序列，求最有可能的对应的状态序列。

## 概率问题计算
### 直接计算法
给定模型 $\lambda = (A, B, \pi)$ 和观测序列 $O = \{o_1, o_2, \cdots, o_T\}$，计算在在模型参数 $\lambda$ 下观测序列 $O$ 出现的概率 $P(O|\lambda)$。最简单的办法就是列举出左右可能的状态序列 $I = \{i_1, i_2, \cdots, i_n\}$，然后再根据观测矩阵 $B$，计算每种状态序列对应的联合概率 $P(O, I|\lambda)$，对其进行求和得到概率 $P(O|\lambda)$。  

状态序列 $I = \{i_1, i_2, \cdots, i_T\}$ 的概率是：

$$
P(I|\lambda) = \pi_{i_1} \prod_{t=1}^{T-1}a_{i_t, i_{t+1}} \tag{8}
$$
对于固定的状态序列 $I = \{i_1, i_2, \cdots, i_T\}$，观测序列 $O = \{o_1, o_2, \cdots, o_T\}$ 的概率是：

$$
P(O|I, \lambda) = \prod_{t=1}^T b_{i_t}(o_t) \tag{9}
$$
$O$ 和 $I$ 同时出现的联合概率为：

$$
\begin{align}
P(O, I|\lambda) &= P(I|\lambda)P(O|I, \lambda) \\
&= \pi_{i_1} \prod_{t=1}^{T-1} a_{i_t, i_{t+1}} \prod_{t=1}^T b_{i_t}(o_t)
\end{align} \tag{10}
$$

然后，对于所有可能的状态序列 $I$ 求和，得到观测序列 $O$ 的概率 $P(O|\lambda)$，即：

$$
\begin{align}
P(O|\lambda) &= \sum_I P(I|\lambda) P(O|I, \lambda) \\
&= \sum_{i_1, i_2, \cdots, i_T|_{N^T}} \pi_{i_1} \prod_{t=1}^{T-1} a_{i_t, i_{t+1}} \prod_{t=1}^T b_{i_t}(o_t)
\end{align} \tag{11}
$$

但利用上式的计算量很大，是 $O(N^T \times T)$ 阶的，这种计算不可行

### 前向算法
**前向概率：** 给定隐马尔可夫模型 $\lambda$, 给定到时刻 $t$ 部分观测序列为 $o_1, o_2, \cdots, o_t$, 且状态为 $q_i$的概率为前向概率，记作：

$$
\alpha_t(i) = P(o_1, o_2, \cdots, o_t, i_t=q_i|\lambda) \tag{12}
$$

1. **初值**

$$
\alpha_1(i) = \pi_i b_i(o_1), i \in [1, N] \tag{13}
$$

2. **递推**：对 $t = 1,2,\cdots,T-1$

$$
\alpha_{t+1}(i) = \left[ \sum_{j=1}^N \alpha_t(j) \cdot a_{ji} \right] b_i(o_{t+1}), i \in [1, N] \tag{14}
$$

3. **终止**：

$$
P(O|\lambda) = \sum_{i=1}^N{\alpha_T(i)} \tag{15}
$$

### 后向算法
**后向概率**: 给定隐马尔可夫模型 $\lambda$，给定在时刻 $t$ 状态为 $q_i$ 的条件下，从 $t+1$ 到 $T$ 的部分观测序列为 $o_{t+1}, o_{t+2}, \cdots, o_T$ 的概率为后向概率，记作：

$$
\beta_t(i) = P(o_{t+1}, o_{t+2}, \cdots, o_T|i_t=q_i, \lambda) \tag{16}
$$
可以递推的求得后向概率 $\beta_t(i)$ 及观测序列概率 $P(O|\lambda)$，后向算法如下：  



1. **初值**：

$$
\beta_T(i) = 1, i \in [1, N] \tag{17}
$$

2. **递推**：对于 t = T-1,T-2,\cdots,1

$$
\beta_t(i) = \sum_{j=1}^N a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j), i  \in [1, N] \tag{18}$$
$$

3. **终止**：

$$
P(O|\lambda) = \sum_{i=1}^N \pi_i \cdot b_i(o_1) \cdot \beta_1(i) \tag{19}
$$

## 学习问题计算
### 监督学习算法
**假设给定训练数据值包含 $S$ 个长度为 $T$ 的观测序列和状态序列 $\{(O_1, I_1), (O_2, I_2), \cdots, (O_S, I_S)\}$ ($S$ 个样本)**, 那么可以利用极大似然估计法来估计隐马尔可夫模型的参数。  

设样本从时刻 $t$ 状态 $i$ 转移到时刻 $t+1$ 状态 $j$ 的**频数为 $A_{ij}$**, 那么**转移概率 $a_{ij}$** 的估计是：

$$
\hat{a}_{ij} = \frac{A_{ij}}{\sum_{j\_ = 1}^N A_{ij\_}}, i \in [1,N], j\_ \in [1,N] \tag{20}
$$
设样本中状态为 $j$ 并观测为 $k$ 的 **频数为 $B_{jk}$**, 那么**状态为 $j$ 观测为 $k$ 的概率 $b_j(k)$** 的估计是：

$$
\hat{b}_j(k) = \frac{B_{jk}}{\sum_{k\_=1}^{M}B_{jk\_}}, j \in [1,N], k\_ \in [1,M] \tag{21}
$$
**初始状态概率 $\pi_i$** 估计 $\hat{\pi}_i$ 为 $S$ 个样本中初始状态为 $q_i$ 的频率。

### 无监督算法
**假设给定训练数据值包含 $S$ 个长度为 $T$ 的观测序列 $\{O_1, O_2, \cdots, O_S\}$ (S个样本)** 而没有对应的状态序列，目标是学习隐马尔可夫模型 $\lambda = (A, B, \pi)$ 的参数。我们将观测序列看作是观测数据 $O$，状态序列数据看作不可观测的隐数据 $I$, 那么马尔可夫模型事实上是一个含有隐变量的概率模型：
$$
P(O|\lambda) = \sum_I P(O|I,\lambda)P(I|\lambda) \tag{22}
$$
它的参数学习可以由 $EM$ 算法实现。$EM$ 算法在隐马尔可夫模型学习中的具体实现为 **$Baum-Welch$ 算法**：

#### 单个序列
1. **设置临时变量 $\gamma_t(\bar{i})$：在给定观察序列 $O$ 和 参数 $\bar{\lambda}$ 的情况下，时间为 $t$、状态为 $q_\bar{i}$的可能性：**

$$
\begin{align}
\gamma_t(\bar{i}) &= P(\bar{i}_t=q_\bar{i}|O, \bar{\lambda}) \\
&= \frac{P(\bar{i}_t=q_\bar{i},O|\bar{\lambda})}{P(O|\bar{\lambda})} \\
&= \frac{\alpha_t(\bar{i}) \cdot \beta_t(\bar{i})} {P(O|\bar{\lambda})} \\
&= \frac{\alpha_t(\bar{i}) \cdot \beta_t(\bar{i})} {\sum_{\bar{j}=1}^{N}{\alpha_t(\bar{j}) \cdot \beta_t(\bar{j})}}
\end{align}
\tag{23}
$$

2. **设置临时变量 $\xi_{t}(\bar{i}\bar{j})$：在给定观察序列 $O$ 和 参数 $\bar{\lambda}$ 的情况下，时间为 $t$、状态为 $q_\bar{i}$、时间为 $t+1$、状态为 $q_\bar{j}$ 的可能性**

$$
\begin{align}
\xi_t(\bar{i}\bar{j}) &= P(\bar{i}_t=q_\bar{i}, \bar{j}_{t+1}=q_\bar{j}|O, \bar{\lambda}) \\
&= \frac{P(\bar{i}_t=q_\bar{i}, \bar{j}_{t+1}=q_\bar{j}, O|\bar{\lambda})}{P(O|\bar{\lambda})} \\
&= \frac{\alpha_t(\bar{i}) \cdot a_{\bar{i}\bar{j}} \cdot b_{\bar{j}}(o_{t+1}) \cdot \beta_{t+1}(\bar{j})} {P(O|\bar{\lambda})} \\
&= \frac{\alpha_t(\bar{i}) \cdot a_{\bar{i}\bar{j}} \cdot b_{\bar{j}}(o_{t+1}) \cdot \beta_{t+1}(\bar{j})} 
{\sum_{\bar{i}=1}^N \sum_{\bar{j}=1}^N \alpha_t(\bar{i}) \cdot a_{\bar{i}\bar{j}} \cdot b_\bar{j}(o_{t+1}) \beta_{t+1}(\bar{j})}
\end{align}
\tag{24}
$$



3. **将式子 $(13)、(14)、(17)、(18)$ 代入式子 $(23)、(24)$ 可得隐马尔可夫的参数**

$$
\pi_i = \gamma_1(\bar{i}) \tag{25}
$$

$$
a_{ij} = \frac{\sum_{t=1}^{T-1} {\xi_t(\bar{i}\bar{j})}} {\sum_{t=1}^{T-1} \gamma_t(\bar{i})} \tag{26}
$$

$$
\begin{align}
b_i(k) &= P(O_t=v_k|i_t=q_i) \\
&= \frac{\sum_{t=1}^T 1_{o_t=v_k} \cdot \gamma_t(\bar{i})} {\sum_{t=1}^T \gamma_t(\bar{i})}
\end{align}
\tag{27}
$$


$$
1_{o_t = v_k} = 
\begin{cases}
1 & if o_t = v_k \\
0 &othrewise
\end{cases}
\tag{28}
$$

4. **备注：**
    - **公式 $28$ 是一个指标函数，并且 $b_i(k)$ 是在状态处于 $q_i$ 时观测值等于 $v_k$ 的次数 与 状态处于 $q_i$ 的总次数的比值**
    - **$\gamma_t(\bar{i})$ 和 $\xi_t(\bar{i}\bar{j})$ 的分母是相同的，它们代表在给定参数 $\bar{\lambda}$ 的情况下，观测序列为 $O$ 的可能性**

#### 多个序列
在目前为止描述的算法是假设一个观察到的序列 $O=(o_1, o_2, \cdots, o_T)$；但是在许多的情况下，可以观察到以下几个序列 $\{O_1, O_2, \cdots, O_S\}$。在这种情况下，必须在更新参数时使用来自所有观察到的序列的信息对参数 $A, B, \pi$ 进行更新。假设你已经对每个序列 $\{O_s，s \in [1, S]\}$ 计算了 $\gamma_t(\bar{i} s)$ 和 $\xi_t(\bar{i} \bar{j} s)$，现在可以更新参数：

