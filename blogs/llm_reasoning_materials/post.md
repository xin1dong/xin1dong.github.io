# Learn LLMs Only with Good Materials (Reasoning)


Upated on July 24, 2025 | Author: Xin Dong

--- 
## Framework 

[veRL](https://github.com/volcengine/verl/tree/main) 
I use veRL a lot in my research. It is a mature RL framework which handles most of infrastructure issues for you. For example, it helps you to allocate the distributed GPU resources for different models involved in the RL training and it abstracts the RL training logic as a simple sequential execution. 
This [video](https://appodzjvyp51702.xet.citv.cn/v3/course/alive/l_6821f9cee4b0694c5ad2531b?app_id=appodzjvyp51702&alive_mode=0&pro_id=&type=2) (in Chinese) given a good introduction to veRL. 

## Whether Base Models (w/o RL Training) Can Reason?

First of all, we need to define what is reasoning. 
In the context of LLMs, reasoning is the behavior of LLMs to generate a sequence of *intermediate* tokens to better answer the question. The *intermediate* tokens can be interpretable or not or even non-human-readable [[Zhu et al., 2025](https://arxiv.org/pdf/2505.12514)]. 
The answer is probably **Yes**, which is not a surprise. 

If one compares the difference between pre-training and reinforcement learning (RL) training, one of the main differences is that the RL is a free-form gradient descent (or ascent, depending on the sample's reward) since it calculates the gradient with free-form self-generated data while pre-training uses pre-defined data. As a result, as long as the pre-training corpus contains some reasoning-related data, the base model should learn to reason at certain level. 

Danny Zhou gave a good talk on this topic in the [Large Language Model Reasoning](https://dennyzhou.github.io/LLM-Reasoning-Stanford-CS-25.pdf) at [Stanford CS25](https://www.youtube.com/watch?v=ebnX5Ur1hBk). 






## Whether RL Training Really Improves Reasoning?


## What is the RL Training Really Optimizing?


## Is GRPO Optimal?

GRPO [[DeepSeek-AI, 2024](https://arxiv.org/abs/2402.03300)], popularized by DeepSeek-R1 is widely used in the community as the de facto RL training method. 

<figure style="text-align: center; margin: 20px auto;">
  <img src="grpo-1.png" alt="GRPO-1" width="700" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;"></figcaption>
</figure>

<figure style="text-align: center; margin: 20px auto;">
  <img src="grpo-2.png" alt="GRPO-1" width="550" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;"></figcaption>
</figure>

$\pi_\theta(y_{i,t}|x,y_{i,<t})$ is the token-wise likelihood on one training sample $(x,y)$ where $x$ is the query and $y$ is the model's response. This term is *the same as* the pre-training loss. The different is that GRPO involves two additional scaling factors: $\frac{1}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$ and $\hat{A}_{i,t}$.


### Token-wise Impact on Training

#### Importance Sampling and Clipping
The theoretical reason for $\frac{1}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$ is importance sampling given that $(x, y_i)$ is sampled from $\pi_{\theta_\text{old}}$ but used to update $\pi_\theta$. However, the possible value of $\frac{1}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$ ranges from 1 to $\infty$, which may introduce huge variance of the gradient. 

To mitigate this issue, GRPO uses a clipping technique to limit the value of $w_{i,t}(\theta) > (1 - \epsilon)$ to a reasonable range. The impact of clipping is more like a token filtering mechanism. For example, when $\hat{A}_{i,t}>0$ and $w_{i,t}(\theta) > (1 - \epsilon)$ where a typical value of $\epsilon$ is 0.2, the token $y_{i,t}$'s loss $\pi_\theta(y_{i,t}|x,y_{i,<t})$ will be not used in the gradient calculation. 

In practice, this clipping technique introduces an extra hyper-parameter $\epsilon$. DAPO [[ByteDance Seed, 2025](https://arxiv.org/pdf/2503.14476)] finds that using a higher clipping upper bound $1 + \epsilon=1.28$ can encourage the model to explore more aggressively and improve the reasoning performance. It makes sense because this will result in filtering fewer tokens that $\pi_\theta$ is different from $\pi_{\theta_\text{old}}$, which is usually termed as *exploration* in RL.

#### Low-Probability Tokens Over-Dominate
Interestingly, although GRPO filters out some tokens by clipping importance sampling ratio $w_{i,t}(\theta)$, tokens with low probability still dominate the gradient as revealed by [[Yang et al., 2025]](https://arxiv.org/pdf/2505.12929). 

<figure style="text-align: center; margin: 20px auto;">
  <img src="low_prob_overdominate.png" alt="GRPO-1" width="600" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;">Do Not Let Low-Probability Tokens
Over-Dominate in RL for LLMs, Yang et al., 2025</figcaption>
</figure>

Although low-probability tokens only account for 19.22\% of the total tokens (a), their gradient norm makes up most of the total gradient norm (d). Another evidence is that if we only use the low-probability tokens' gradient to update the model, the model's change is still quite similar to the one using all the tokens' gradient (e) while only using the high-probability tokens' gradient will lead to very different model updates (f). 


What if the gradient of low-probability tokens over-dominates? The concern is that tokens with high probability cannot be updated towards the right direction effectively. A simple fix proposed by [[Yang et al., 2025]](https://arxiv.org/pdf/2505.12929) is to seperate low- and high-probability tokens into two groups and apply their gradients individually.

#### Sequence-wise Importance Sampling 

In GRPO, we scale the gradient of each token by $\frac{1}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$, which could be problematic according to the above analysis. Then, a straightforward fix is to scale all tokens in a sequence with the averaged $\frac{1}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$ across all tokens in that sequence. Suppose we ignore the advantage and clipping, the loss of GRPO is 

\[
    \mathcal{J}_{\text{GRPO}} = \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}
    \]

GSPO [[Zheng et al., 2025](https://arxiv.org/pdf/2507.18071)] proposed to change it into 

\[  
    \begin{aligned}
    \mathcal{J}_{\text{GSPO}} &= \left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}} \\
    & = \left(\frac{\Pi_{t=1}^{|y_i|} \pi_\theta(y_{i,t}|x,y_{i,<t})}{\Pi_{t=1}^{|y_i|} \pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}\right)^{\frac{1}{|y_i|}} \\
    & = \exp\left(\frac{1}{|y_i|}\log\left(
    \frac{\Pi_{t=1}^{|y_i|} \pi_\theta(y_{i,t}|x,y_{i,<t})}{\Pi_{t=1}^{|y_i|} \pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}
    \right)\right) \\
    & = \exp\left(\frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log\left(
    \frac{ \pi_\theta(y_{i,t}|x,y_{i,<t})}{ \pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}
    \right)\right)
    \end{aligned}
    \]


By comparing the loss of GRPO and GSPO, we can see that 


- GRPO is doing arithmetic mean of $\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$.

- GSPO is doing geometric mean of $\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_\text{old}}(y_{i,t}|x,y_{i,<t})}$.

Comparing to arithmetic mean, geometric mean is less sensitive to extreme outliers. 

The true loss of GSPO is 


<figure style="text-align: center; margin: 20px auto;">
  <img src="gspo_loss.png" alt="GRPO-1" width="600" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;">Group Sequence Policy Optimization, Zheng et al., 2025</figcaption>
</figure>

When we put the gradient of GRPO and GSPO together, we can see 



<figure style="text-align: center; margin: 20px auto;">
  <img src="gspo_gradient.png" alt="GRPO-1" width="600" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;">Group Sequence Policy Optimization, Zheng et al., 2025</figcaption>
</figure>

<figure style="text-align: center; margin: 20px auto;">
  <img src="grpo_gradient.png" alt="GRPO-1" width="600" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;">Group Sequence Policy Optimization, Zheng et al., 2025</figcaption>
</figure>

where all tokens in a sequence share the same scaling factor $\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}\right)^{\frac{1}{|y_i|}}$ in GSPO. 

#### High-Entropy Tokens Matter
Another line of work studies impact of tokens from the perspective of token entropy. Note that the entropy of a token is defined as

\[
    H(p)=-\sum_{i=1}^N p_i\log p_i,
    \]

given the output probability of this token is $p\in\mathbb{R}^N$ where $N$ is the vocabulary size. 

If a token has 100% probability at the ground-truth index, its entropy is 0. However, if a token has low probability at the ground-truth index, its entropy can be high or low depending on the probability distribution of the other indices. 

Many studies [[Wang et al., 2025](https://arxiv.org/pdf/2506.01939), [Wang et al., 2025](https://arxiv.org/pdf/2507.15778v1)] find that high-entropy tokens contribute significantly to the success of RL training.
Both studies find that high-entropy tokens are usually words that connect reasoning steps and guide the model toward continuing the reasoning process.

<figure style="text-align: center; margin: 20px auto;">
  <img src="20_80_words.png" alt="GRPO-1" width="600" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;">Beyond the 80/20 Rule, Wang et al., 2025</figcaption>
</figure>

These tokens that have high entropy and steer reasoning path are called forking tokens.
[Wang et al., 2025](https://arxiv.org/pdf/2506.01939) found that utilizing only 20\% of the tokens with high entropy get even better performance than full-gradient updates and the performance gap is more significant when the model is larger. In the same time, this method also increases the average response length probably because of the emphasis on forking tokens. 

[Wang et al., 2025](https://arxiv.org/pdf/2507.15778v1) found a very similar phenomenon. 


<figure style="text-align: center; margin: 20px auto;">
  <img src="kuaishou_words.png" alt="GRPO-1" width="600" style="display: block; margin: 0 auto; max-width: 100%;">
  <figcaption style="color: #666; font-size: 0.9em; margin-top: 10px;">Stabilizing Knowledge, Promoting Reasoning: Dual-Token Constraints for RLVR, Wang et al., 2025</figcaption>
</figure>

They call high-entropy tokens as reasoning-related tokens because these tokens act as logical coonectors and call low-entropy tokens as knowledge-related tokens because these tokens are usually words tied to factual knowledge although I am not sure if this is a good definition of 'knowledge'. 
The solution they proposed is to deal with these two types of tokens differently by using two sets of hyper-parameters. Specifically, they use a larger $\epsilon$ and smaller coefficient of $\mathcal{D}_{\text{KL}}(\pi_\theta\|\pi_{\theta_\text{ref}})$ for for reasoning-related tokens.