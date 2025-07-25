# Learn LLMs Only with Good Materials (Reasoning)


Upated on July 24, 2025 | Author: Xin Dong

--- 
## Framework 

[veRL](https://github.com/volcengine/verl/tree/main) 
I use veRL a lot in my research. It is a mature RL framework which handles most of infrastructure issues for you. For example, it helps you to allocate the distributed GPU resources for different models involved in the RL training and it abstracts the RL training logic as a simple sequential execution. 
This [video](https://appodzjvyp51702.xet.citv.cn/v3/course/alive/l_6821f9cee4b0694c5ad2531b?app_id=appodzjvyp51702&alive_mode=0&pro_id=&type=2) (in Chinese) given a good introduction to veRL. 

## Whether Base Models (w/o RL Training) Can Reason?

First of all, we need to define what is reasoning. 
In the context of LLMs, reasoning is the behavior of LLMs to generate a sequence of *intermediate* tokens to better answer the question. The *intermediate* tokens can be interpretable or not or even non-human-readable. 
The answer is probably **Yes**, which is not a surprise. 

If one compares the difference between pre-training and reinforcement learning (RL) training, one of the main differences is that the RL is a free-form gradient descent (or ascent, depending on the sample's reward) since it calculates the gradient with free-form self-generated data while pre-training uses pre-defined data. As a result, as long as the pre-training corpus contains some reasoning-related data, the base model should learn to reason at certain level. 

Danny Zhou gave a good talk on this topic in the [Large Language Model Reasoning](https://dennyzhou.github.io/LLM-Reasoning-Stanford-CS-25.pdf) at [Stanford CS25](https://www.youtube.com/watch?v=ebnX5Ur1hBk). 






## Whether RL Training Really Improves Reasoning?


## What is the RL Training Really Optimizing?





