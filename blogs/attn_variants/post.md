# Attention Variants

Updated on Aug 3, 2025 | Author: Xin Dong

--- 
## Sharing Makes Us Happy
### Group Query Attention (GQA)

Simply put, GQA is a variant of multi-head attention mechanism that shares KV pairs across multiple heads. 
Suppose $\frac{h}{g}$ heads share one set of KV pairs, then what we need to do is copy the KV pair $\frac{h}{g}$ times, where $h$ is the total number of heads (or number of Q heads) and $g$ is the number of groups (or number of KV heads).

In linear algebra, sharing and copying can be represented as a low-rank matrix factorization,

\[
    c^{KV} = x W^{DKV}, \quad kv = W^{UKV} c^{KV}
\]

Let $d_h$ be the dimension of one head (usually $d_h = 128$, for example, in Llama2). The hidden state is represented as $x\in\mathbb{R}^{d_h\times h}$. 

- If using standard multi-head attention (MHA), then $c^{KV}\in\mathbb{R}^{d_h\times 2h}$ and $W^{DKV}$ is the identity matrix. 
- If using group query attention (GQA), then $c^{KV}\in\mathbb{R}^{d_h\times 2g}$ and $W^{DKV}$ is a special [copying matrix](https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices) that duplicates $c^{KV}$ for $\frac{h}{g}$ times. 

Let's call the space of $c^{KV}$ the latent head space. The smaller $g$ is, the smaller the rank of the latent head space, and the more we compress the KV pairs. 

In implementation, we only need to save $c^{KV}\in\mathbb{R}^{d_h\times 2g}$ which is $\frac{h}{g}$ times smaller than the original $c^{KV}\in\mathbb{R}^{d_h\times 2h}$. The second multiplication is just duplication which does not involve any parameters or computation. 
This is why GQA is more memory efficient than MHA. 

### Multi-head Latent Attention (MLA)

MLA, proposed in [[DeepSeek-V2]](https://arxiv.org/pdf/2405.04434), is a variant of MHA that leverages low-rank joint compression of keys and values. 

Since we already analyzed that GQA is in a form of low-rank matrix factorization, we can also derive MLA with the same equation. 

- In GQA, the matrix $W^{DKV}$ is a special copying matrix that duplicates $c^{KV}$ for $\frac{h}{g}$ times. 
- In MLA, why don't we remove the constraint on matrix $W^{DKV}$ as a special copying matrix and make it a general matrix instead? 

Congratulations! You have just discovered MLA!

However, MLA makes this change with some price to pay. 

In GQA, we just save $c^{KV}\in\mathbb{R}^{d_h\times 2g}$ to save memory. During computation, we can just copy $c^{KV}$ for $\frac{h}{g}$ times. However, in MLA, if we save $c^{KV}\in\mathbb{R}^{d_h\times 2g}$, then extra computation is needed to compute $kv = W^{UKV} c^{KV}$. This extra computation is expensive.


Let's revisit standard multi-head attention (MHA) and find a solution for this problem. 

For a single head in standard MHA, we have 

\[
q_i \cdot k_j = x_i W_i^Q (x_j W_j^K)^T = x_i \left( W_i^Q \left( W_j^K \right)^T \right) x_j^T     \tag{1}
\]

But why does nobody compute $x_i \left( W_i^Q \left( W_j^K \right)^T \right) x_j^T$? 

- One reason is that $q_i \cdot k_j$ has flops $O(2 \cdot n \cdot d_\text{model} \cdot d_h + n^2 \cdot d_h)$, where $n$ is the sequence length. While $x_i \left( W_i^Q \left( W_j^K \right)^T \right) x_j^T$ has flops $O(d_\text{model}^2 \cdot d_h + n^2 \cdot d_\text{model})$. Note that $d_\text{model}=hd_h$ is usually much larger than $d_h$. 

- If we compute $x_i \left( W_i^Q \left( W_j^K \right)^T \right) x_j^T$, then we need to save $x$ as context. If we compute $q_i \cdot k_j$, then we only need to save $q$ and $k$ as context. The former memory size is $O(n \cdot hd_h)$, while the latter is $O(n \cdot 2d_h)$. 

Although we discovered a useless trick, we now know that **you can fold the linear projection on K into Q's projection matrix**. 

In MLA, if we isolate the second projection on K, 

\[
k = W^{UK} c^{KV}
\]

We can fold $W^{UK}$ into $W^{Q}$. Similarly, we can fold $W^{UV}$ into the output projection $W^O$. 

As a result, we can just save $c^{KV}\in\mathbb{R}^{d_h\times 2g}$ without any extra computation. 


So far, we derived the MLA by connecting it to the GQA variant. 

In DeepSeek-V2, $d_\text{model}=5120$, $d_c=512$,

- If we use MHA, we usually set per-head dimension $d_h=128$ and have $h=\frac{5120}{128}=40$ heads. **So the KV cache size for one token is $2\times 5120$.**
- If we use MLA, we still set $d_h=128$. We only have one latent KV head and the latent head dimension is $d_c=512$. **So the KV cache size for one token is $1\times 512$.** Then, the single latent KV head is projected to $h=128$ heads.

You may wonder why DeepSeek-V2 uses $h=128$ heads instead of $h=40$ heads. The reason is that DeepSeek-V2's cache size only depends on $d_c$ and not on the number of heads $h$. As a result, it can freely set $h=128$ to increase compute. 

```latex
In summary, since MLA shares KV pairs in a compressed low-rank space, it allows one to increase the number of heads to increase compute intensity and control cache size by setting the latent head dimension. 
```

An additional trick that DeepSeek-V2 uses is that they preserve a few individual dimensions in query and key to add RoPE and hold positional information. The reason for using extra dimensions is that RoPE is not compatible with equation (1). DeepSeek-V2 split head dimension into two parts: one part is using MLA without RoPE, and the other part is using MQA with RoPE. The second part is much smaller than the first part but is sufficient to model positional information. 

This can be treated as partial RoPE. Partial RoPE [[BarBero et al., 2025]](https://openreview.net/pdf?id=GtvuNrk58a) actually can achieve better performance than full RoPE even for standard MHA. The intuition is that RoPE, as an injection of positional information, occupies the capacity for semantic information modeling. If we only apply RoPE to part of the dimensions, then we can free other dimensions to better capture semantic information. 





