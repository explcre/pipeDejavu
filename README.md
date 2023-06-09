# pipeDejavu: Hardware-aware Latency Predictable, Differentiable Search for Faster Config and Convergence of Distributed ML Pipeline Parallelism

## How to run the code
1. Predictive Model saving pre-profiling time of Alpa.

    To check the content in the `prof_database.pkl` file
    You can run
    ```
    python ./pickle_display_result.py
    ```

    To check the predictive model(fitting a linear regression) for pre-profiling database,
    You can run
    ```
    python ./pickle_find_pattern.py
    ```

2. Differentiable Search Space

    To check the result of Differentiable Search method comparing with Dynamic Programming , please check:

    https://github.com/explcre/pipeDejavu/blob/main/auto_pipeline_slicing_dp_ipynb-copy.ipynb



3. Parallel Random Initialization

    To run the simulation program for parallel random initialization, please run:
    ```
    python ./simulate_random_initialization_parallel.py --to_demo False --num_workers 8 --epochs 100 --demo_epochs 20
    ```


# hardware-aware scheduling
notion: https://www.notion.so/hardware-aware-scheduling-487793b3a1874389b3f8a57c848b23ae

googledoc:[https://docs.google.com/document/d/1qON658QVbS2n_xLyuYeIiXB8Oqp0kjT4eKmmuKapV2A/edit?usp=sharing](https://docs.google.com/document/d/1qON658QVbS2n_xLyuYeIiXB8Oqp0kjT4eKmmuKapV2A/edit?usp=sharing)

overleaf: [https://www.overleaf.com/6194344811knzgsftfhnhm](https://www.overleaf.com/6194344811knzgsftfhnhm)

github: [explcre/pipeDejavu (github.com)](https://github.com/explcre/pipeDejavu)

gpu use plan: [https://docs.google.com/document/d/1vT9A8O0NoQhAmFYVq_GjZyjY7yW2SIqbmUGdQxoBLOU/edit?usp=sharing](https://docs.google.com/document/d/1vT9A8O0NoQhAmFYVq_GjZyjY7yW2SIqbmUGdQxoBLOU/edit?usp=sharing)

1. **As for heterogenous pipeline parallelism(predict latency model)**, we find we can look into simulator in the flexflow, make it one step predict model , based on communication cost and other parameters. And perhaps we can also plug it into alpa.(while alpa only assume network inside machine is much larger than between machines, we can use this simulator to meet situation where network doesn’t follow this condition)
(another difference is search algorithm, flexflow use mcmc, alpa use dp)
    
    
    “HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism” Maybe propose a way for heterogenous circumstances.
    Indy opinion: take a step back and think of new ways of combining pipeline parallelism and data parallelism.
    
    1. [Learned TPU Cost Model for XLA Tensor Programs – Google Research](https://research.google/pubs/pub49859/) [learned_tpu_kaufman_2019.pdf (mlforsystems.org)](http://mlforsystems.org/assets/papers/neurips2019/learned_tpu_kaufman_2019.pdf)
        
        xla: 6 Future Work
        While early results are promising, this is a work in progress. For the next steps, we would like to try the following ideas. First, train and evaluate on a broader dataset of XLA graphs to improve accuracy and better understand the model’s ability to generalize. Second, experiment with modifications to the model that encourage generalization between kernels with the same computation graph but different tensor shapes (similar to the approach used in Halide to learn coefficients for hand-engineered performance counters [2]). Third, extend the model to evaluate an XLA graph that varies along other
        axes, such as kernel’s tile size and layout assignment. Finally, we would like to improve performance on large kernels in particular.
        
    2. [osdi22-unger.pdf (usenix.org)](https://www.usenix.org/system/files/osdi22-unger.pdf)Unity: Accelerating DNN Training Through Joint
    Optimization of Algebraic Transformations and Parallelization这是flexflow最新paper, 把代数变换也和parallel策略一起优化
    3. [Beyond Data and Model Parallelism for Deep Neural Networks (stanford.edu)](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf)可以借鉴flexflow里面simulator
    4. [用ILP和DP自动探索DL分布式策略——Alpa - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/487588274)
    5. [Alpa: Automated Model-Parallel Deep Learning – Google AI Blog (googleblog.com)](https://ai.googleblog.com/2022/05/alpa-automated-model-parallel-deep.html)
    6. [https://alpa.ai/tutorials/perf_tuning_guide.html](https://alpa.ai/tutorials/perf_tuning_guide.html) 这里最后说https://github.com/alpa-projects/alpa/blob/main/tests/runtime/test_debug_info.py可以看他的runtime跑的情况debug，之后也许会用到。但貌似他说他没有一个很好visualization tool
    7. 在dp之前直接传给他profiling结果的是alpa/pipeline_parallel/stage_profiling.py里面的get_compute_cost()函数，如果要优化profiling可能就是仔细看他里面 tic toc,时间测了哪些。感觉可以仔细看看alpa/pipeline_parallel/stage_profiling.py里面的get_compute_cost()函数，如果能单机跑alpa的话，可以测一下这块跑了多久时间，如果不去profiling直接赋值，会节省多少时间
2. **differentiable parallel configuration search space**
Can we use differentiable search algorithm like “DARTS: Differential Neural Architecture Search” to search the optimal of parallel plan?
The existing optimization of auto parallel strategy is mostly MCMC or Dynamic Programming.
I was thinking that there is a differentiable method DARTS in Neural Architecture Search to search for the minimum value. 
That is, the n alternative discrete max of the original search for the optimal network structure becomes a differentiable softmax approximation, and then derives the derivative. The gradient of the train and the gradient of finding the optimal network structure are combined into one formula. Each time the gradient drops one step, the optimal network structure is found while training. If the distribution training is based on a calculation graph (like the latest flexflow 22 article on unity), can this similar method be used to search for the minimum value. Find the optimal parallel configuration while training. Now the search is generally dp/mcmc or something. Personally, I feel that this method should be quite insightful, not a small repair, but whether it can really work depends on whether it can be implemented.
    
    
    Indy’s: Intriguing idea. May not beat existing algorithm at first when implement at first time. Can start implement small part, do some experiments. Show some minimum doable results that it may work.
    
    1. [Darts代码解读 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/143574526)
    2. Differentiable Dynamic Programming for
    Structured Prediction and Attention [1802.03676.pdf (arxiv.org)](https://arxiv.org/pdf/1802.03676.pdf)
    3. [Intuitive Explanation of Differentiable Architecture Search (DARTS) | by Shivam Kaushik | Towards Data Science](https://towardsdatascience.com/intuitive-explanation-of-differentiable-architecture-search-darts-692bdadcc69c)
    4. Differentiable Neural Architecture Search in Equivalent Space with Exploration Enhancement [9a96a2c73c0d477ff2a6da3bf538f4f4-Paper.pdf (neurips.cc)](https://proceedings.neurips.cc/paper/2020/file/9a96a2c73c0d477ff2a6da3bf538f4f4-Paper.pdf)
    5. 给gpt4喂了stage_construction.py代码以及一堆darts相关资料之后他给的回答，写了初步代码[Shared Conversation (chatgpt4google.com)](https://webapp.chatgpt4google.com/s/NTEzMTI5)
    6. gpt4 输入了我们differentiable search space之后对时间复杂度的分析[Shared Conversation (chatgpt4google.com)](https://webapp.chatgpt4google.com/s/MTU3Mjg2)
    7. alpa/pipeline_parallel/stage_construction.py我理解是dp在的地方。里面616行开始，AutoStageOption,就是他，自动去搜的代码应该是。311行training_dp()就是里面具体dp算法。235行training_dp_imp()更核心一点
    8. [Euphoria16/Shapley-NAS: update README.md (github.com)](https://github.com/euphoria16/shapley-nas) [Shapley-NAS: Discovering Operation Contribution for Neural Architecture Search (thecvf.com)](https://openaccess.thecvf.com/content/CVPR2022/papers/Xiao_Shapley-NAS_Discovering_Operation_Contribution_for_Neural_Architecture_Search_CVPR_2022_paper.pdf)  这个是NAS 的SOTA,用game theory里面sharpley value 去预测哪个operation贡献大
    9. [krzysztofrusek/net2vec: This repository is a collection of machine learning models for computer networks. (github.com)](https://github.com/krzysztofrusek/net2vec)  
        1. [Message Passing](https://github.com/krzysztofrusek/net2vec/blob/master/mpnn)vanilla Graph Neural Network
        2. [RouteNet](https://github.com/krzysztofrusek/net2vec/blob/master/routenet) - A new neural architecture designed for neural understanding of routing in the network.
        3. [Routing by Backprop](https://github.com/krzysztofrusek/net2vec/blob/master/routing_by_backprop) - Differentiable surrogate for Dijkstra algorithm.
        4. [HeterPS: Distributed Deep Learning With Reinforcement Learning Based Scheduling in Heterogeneous Environments (arxiv.org)](https://arxiv.org/pdf/2111.10635.pdf)
    10. [HeterPS: Distributed Deep Learning With Reinforcement Learning Based Scheduling in Heterogeneous Environments (arxiv.org)](https://arxiv.org/pdf/2111.10635.pdf)
    11. [2209.10380.pdf (arxiv.org)](https://arxiv.org/pdf/2209.10380.pdf)Fast Traffic Engineering by Gradient Descent with
    Learned Differentiable Routing
    12. [2110.02781v1.pdf (arxiv.org)](https://arxiv.org/pdf/2110.02781v1.pdf) FTPipeHD: A Fault-Tolerant Pipeline-Parallel
    Distributed Training Framework
    for Heterogeneous Edge Devices
    13. [2110.14895.pdf (arxiv.org)](https://arxiv.org/pdf/2110.14895.pdf) Pipeline Parallelism for Inference on Heterogeneous Edge Computing
    14. [osdi20-jiang.pdf (usenix.org)](https://www.usenix.org/system/files/osdi20-jiang.pdf) A Unified Architecture for Accelerating Distributed
    DNN Training in Heterogeneous GPU/CPU Clusters
    15. [Optimizing Distributed Training Deployment in Heterogeneous GPU Clusters (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3386367.3432728)   heterog gnn   https://github.com/eval-submissions/HeteroG 用GNN学习placement,超过了hetpipe
        1. [atc20-park.pdf (usenix.org)](https://www.usenix.org/system/files/atc20-park.pdf)HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism 
        2. 
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ec3835b6-3a02-47e6-b485-9f17ebefd3ce/Untitled.png)
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67bbce77-bd45-45c9-aae4-83515e13fc3b/Untitled.png)
            
    16. [1906.01736.pdf (arxiv.org)](https://arxiv.org/pdf/1906.01736.pdf)  Distributed Training with Heterogeneous Data:
    Bridging Median- and Mean-Based Algorithms
    
    []()
    
3. **Another dimension: parallel random initialization for faster training loss convergence**
I also thought of a problem, distribution can also be viewed from another angle, the initial randomization. Gradient descent can be imagined as a rugged hill looking for the minimum value. Gradient descent is the ball rolling down the steepest place. The intuition of data parallel is to assign n workers to look at n small directions at a random initial point, then combine the gradients and then gradient descent. 
But at some points of the initial randomization, the loss may be very low at some points at the beginning, and the gradient descent at those points may quickly reach the minimum. Now the general training of neural networks is just randomization once at the beginning instead of sampling many times. At the beginning of parallel, we can sample those randomization points in parallel at the beginning. We can select the ones with small loss and large gradients while sampling. 
After filtering out some points, we can continue to sample in a new smaller scope. After reaching a certain level, we can focus on the gradient descent of a few points. You can start with a rough grid with a large interval, and then find the one with a small loss inside, and then perform finer-grained sampling near this point. I was still wondering if we could catch up with this year's 5/10 deadline nips if we do experiment quickly. 
Also, if it works, we could put this parallel randomization into the best existing distributed training algorithm

Indy’s opinion: Intriguing as well. Simpler than previous idea. Be careful that loss have steep at the beginning of training. Make sure we filter out the initialization when we reach plateau of loss. Also, don’t make sampling strategy as complex as neural network.
    
    reference:
    
    1. [Gradient descent with random initialization: fast global convergence for nonconvex phase retrieval | SpringerLink](https://link.springer.com/article/10.1007/s10107-019-01363-6)
    2. Understanding the difficulty of training deep feedforward neural networks:[http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi)  This paper introduces xavier init.
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/76b41343-6699-4af3-947b-b7282c76e218/Untitled.png)
        
    3. ****Kaiming Initialization****Introduced by He et al. in [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://paperswithcode.com/paper/delving-deep-into-rectifiers-surpassing-human)
        1. **Kaiming Initialization**, or **He Initialization**, is an initialization method for neural networks that takes into account the non-linearity of activation functions, such as [ReLU](https://paperswithcode.com/method/relu) activations.
            
            A proper initialization method should avoid reducing or magnifying the magnitudes of input signals exponentially. Using a derivation they work out that the condition to stop this happening is:
            
            $$
            \frac{1}{2}n_{l}\text{Var}\left[w_{l}\right] = 1
            $$
            
            This implies an initialization scheme of:
            
            $$
            w_{l} \sim \mathcal{N}\left(0,  2/n_{l}\right)
            $$
            
            That is, a zero-centered Gaussian with standard deviation of  
            
            $$
            \sqrt{2/{n}_{l}}
            $$
            
            (variance shown in equation above). Biases are initialized at 0.
            
            Source:
            
            [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852v1)
            
        
        d. [torch.nn.init — PyTorch 2.0 documentation](https://pytorch.org/docs/stable/nn.init.html) 这些有很多种init方法，后续可以并行搜索一下对于模型哪个更好
        
        e. random initialization gpt4的回答帮助写demo代码    [https://webapp.chatgpt4google.com/s/ODQ4NzIz](https://webapp.chatgpt4google.com/s/ODQ4NzIz)
        
        f. [Gradient Descent in 3D Visualization | Machine Learning | Data Science - YouTube](https://www.youtube.com/watch?v=My7buNDH50k)
        
        g.[Visualizing Gradient Descent in 3D | Kaggle](https://www.kaggle.com/code/christianwittmann/visualizing-gradient-descent-in-3d)
        
        h. [Entropy | Free Full-Text | A Neural Network MCMC Sampler That Maximizes Proposal Entropy (mdpi.com)](https://www.mdpi.com/1099-4300/23/3/269#:~:text=Markov%20Chain%20Monte%20Carlo%20%28MCMC%29%20methods%20sample%20from,with%20neural%20networks%20can%20potentially%20improve%20their%20efficiency.)
        
        1. if network is deep enough, all the local minima have similar loss function alue as the global loss function value [[1605.07110] Deep Learning without Poor Local Minima (arxiv.org)](https://arxiv.org/abs/1605.07110)
        
        j. can we use evolution algorithm to filter the bad loss state/hyperparameter/randomized initialization
        
        k. how to split data, if data splited on each machine is alike, a node for cat , a node for dog, can the gradient descent be less or more conflict?
        
        l.[755acd0c7c07180d78959b6d89768207-Paper-Conference.pdf (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2022/file/755acd0c7c07180d78959b6d89768207-Paper-Conference.pdf)
        
        On Scrambling Phenomena
        for Randomly Initialized Recurrent Networks
        
        m.[2010.01179.pdf (arxiv.org)](https://arxiv.org/pdf/2010.01179.pdf)
        
        The Surprising Power of Graph Neural Networks with
        Random Node Initialization
        
        n. parallel tune batch size parameter
        
        o. use umap to visualize the distribution of random initialization , move the points ,then inverse.
        
4. **Fault-tolerance for model parallel/pipeline parallel , alpa**
    
    [Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance (arxiv.org)](https://arxiv.org/pdf/1805.10032.pdf) This paper provide a algorithm when faulty node is possible in parameter server distributed training, which is data parallel.  It analyzed the Stochastic gradient descent.
    
    While is there any similar fault-tolerance analysis for model/pipeline parallel, and alpa like strategy?
    
    1. [PipeDream: Fast and Efficient Pipeline Parallel DNN Training -0.22in (arxiv.org)](https://arxiv.org/pdf/1806.03377.pdf)
    2. [https://arxiv.org/pdf/1805.10032.pdf](https://arxiv.org/pdf/1805.10032.pdf) 之前说的zeno，他假设data parallel会有些机器传的有问题的梯度，然后他提出一种方法我理解主要思想就是把梯度对现在的影响从小到大排列取影响最小的k个取均值梯度下降。我感觉现在没怎么看到model parallel/pipeline parallel的fault tolerance的分析。感觉也可以类似提出，对于分布在某个机器的那几层模型，如果有坏的梯度的假设，可以设计容错性算法，更新对那一层模型从小到大排列影响最小的k个取平均值梯度下降
    3. [Shared Conversation (chatgpt4google.com)](https://webapp.chatgpt4google.com/s/MzY0MDg3)
    4. 
    
    Algorithm
    
    1. Initialize the weights of each machine $ w_i=1 / N $.
    2. Compute the local gradients $g_i$ on each machine.
    3. Compute the overall gradient $g$ as the weighted sum of the local gradients:
    $$
    g=\sum_{i=1}^N w_i g_i
    $$
    4. Compute the suspicion level of each machine $s_i$ :
    $$
    s_i=\left\|g_i-g\right\|
    $$
    5. Update the weight of each machine according to its suspicion level:
    $$
    w_i=\frac{\alpha}{\alpha+s_i}
    $$
    6. Sort the machines based on their weight in ascending order.
    7. Identify the $\mathrm{k}$ machines with the least influence on the overall gradient.
    8. Replace the gradients of the identified machines with the average of the remaining gradients:
    $$
    \hat{g}*i=\frac{\sum*{j=1}^{N-k} g_j}{N-k}
    $$
    9. Compute the overall gradient $\$ g \$$ using the updated gradients:
    $$
    g=\sum_{i=1}^N w_i \hat{g}_i
    $$
    10. Update the model parameters using the overall gradient.
    11. Repeat steps 2-10 until convergence.
    
5. **Fault-tolerance for Neural Network**
    1. assumption: neural network have k number of nodes/layer/data/…, **added/deleted/go wrong**, whether it will keep its performance/accuracy.
    2. [IEEE Xplore Full-Text PDF:](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8013784)Fault and Error Tolerance in Neural Networks: A Review
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e95a51cd-0f51-4a27-b6f5-3f86cb9f70ac/Untitled.png)
        
    
6. How to distribute data/model:
    1. if the policy to distribute a batch of data is carefully designed, or similar label or different label,(using some clustering method), whether we can decrease the gradient conflict between different batches of data or better fit into the data parallel situation? Whether we can use a metric based on model loss and data distribution to decide how to distribute the data. Will we have gain from just randomly assigned data?
        1. positive/negetive label distribution. multi-label distribution. regression data value distribution.
    2. data distribution can change, a set of data can be piggy backed to another machine, if it find this way can better converge during distributed training. The data distribution policy can evolve through time, based on some metric.
    3. 
        
        Distributed Training with Heterogeneous Data:
        Bridging Median- and Mean-Based Algorithms
        
        [2109.03194.pdf (arxiv.org)](https://arxiv.org/pdf/2109.03194.pdf)
        
        On the Convergence of Decentralized Adaptive Gradient
        Methods
        
    4. [pan-etal.arxiv2016cyclades1.pdf (berkeley.edu)](https://amplab.cs.berkeley.edu/wp-content/uploads/2016/08/pan-etal.arxiv2016cyclades1.pdf)  CYCLADES: Conflict-free Asynchronous Machine Learning
        1. 把计算图没冲突的分组。在dl中，前向后向传播作为一个epoch的计算图，是否也能找出计算图无冲突或者冲突比较小的分组。
    5. use umap to visualize the distribution of data distribution , move the points ,then inverse. assign data with more class/umap diversity.
    6. [621461af90cadfdaf0e8d4cc25129f91-Paper.pdf (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf#:~:text=Deep%20learning%20algorithms%20can%20fare%20poorly%20when%20the,and%20their%20combination%20achieves%20even%20better%20performance%20gains1.)Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss
    7. [1906.01736.pdf (arxiv.org)](https://arxiv.org/pdf/1906.01736.pdf)  Distributed Training with Heterogeneous Data:
    Bridging Median- and Mean-Based Algorithms
7. Eventually consistent→eventually converge:
    1. the machine can fail or change, the network topology or congestion may change when training, but eventually this model will converge. How to prove/ realize this feature? Can we use similar thoughts from eventually consistent?
8. Another communication topology:
    1. similar to parameter server, but server is rolling.
        1. a→{b,c,d,e,f},    {a,c,d,e,f}→ b  , b→{a,c,d,e,f} , {a,b,d,e,f}→ c. 
    2. It can avoid single point of failure. when detect failure, skip that machine  
    3. [分布式训练 – 第3篇 - 分布式训练常用的集合通信及其通信原语 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/493092647?utm_id=0)
9. gossip style distributed training
    1. randomly choose k nodes to send gradient or gather gradient.
10. heterogeneous,if gpu memory is not same, computational power not same, how to schedule?
    
    A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters
    
    [Heterogeneity-Aware Distributed Machine Learning Training via Partial Reduce (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3448016.3452773)
    
    [Optimizing Distributed Training Deployment in Heterogeneous GPU Clusters (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3386367.3432728) 用GNN学习placement,超过了hetpipe
    
    [atc20-park.pdf (usenix.org)](https://www.usenix.org/system/files/atc20-park.pdf)
    
    HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism 
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8a022957-1746-4ef6-8199-81c895259f59/Untitled.png)
    
11. distributed ml simulation playground
    1. it taks resourses to simulate the distributed training, whether we can provide a simulation playground so that the distributed training experiment can be done on one machine.
12. reading code, nlp, to predict performance

### Kaiyang Chen

these days the computation cost is hard to estimate or complicated to explicitly define because:

1. the cost are affected by user behavior like selection of optimization strategy (sharding or not, etc). And in different strategy, the relative score ranking of same machine can be different.
2. The factors that need to be consider is complex, not only GPU memory, but also CPU memory and computation power when using strategy like ZeRO-offload (even NVMe when choosing ZeRO-Infinite), and the network condition since the communication can be the bottleneck in some settings.
3. Hard to verify since we will need larger number of heterogeneous clusters in order to comprehensively verify the practicality and benefits of our solution. (cost is high…)

But i still think it’s worthwhile to do hardware-aware scheduling. At least i can think of one scenario that can be beneficial. When doing data (sharding) & pipeline combined strategy, since the network size of each layer(pipeline block) is different, at least we can put the larger size block to those cluster with high-speed network so that the average training time can be shorter because(faster communication).

One good thing is we already have great code base, DeepSpeed and ColossalAI

Another question is i am not sure what’s the logic for above existing system to perform scheduling (need to check the source code), i think it’s quite different with Beachi, which generate the placement directly.

**Pengcheng Xu**

idea:

predict 1.execution time 2. resource uses

based on: 1. ML model 2. hardware 3.code 4. meta-data 5. user behaviour data

prediction can be analytical formula,

like GPT model,first layer on A100 GPU , mean execution time 1h, stddev:10min

searching algorithm:

dp

mcmc+simulator

which one better, on which senerio?

alpa没建模communication的cost，他们假定机器内通讯远大于机器之间。
○我们可以有一个小出发点可不可以简单sample/或者直接get一下communication速度，然后加入到他们的方程之中（貌似他们默认机器间就inter-op,机器内就intra-op,相当于他们锁了一个更小搜索空间。有没有可能如果网络速度变奇怪，机器间大于机器内，他们alpa估计还是这样分配。如果加入这个communication cost也许能改进这个情况）。

**alpa profiling**

alpa/pipeline_parallel/stage_profiling.py

alpa/mesh_profiling.py

Flexflow [flexflow/FlexFlow: A distributed deep learning framework that supports flexible parallelization strategies. (github.com)](https://github.com/flexflow/FlexFlow)

他们19年那篇paper是通过mcmc采样来找最优分配(alpa是dp)，还有个simulatior模块去一步计算一种采样的分配算多久（可能简单算每个operator时间）。我看的19年MLsys他们文章和视频。
○这个simulator能不能用到alpa的profile里

**simulator:**

Zhihao Jia, Sina Lin, Charles R. Qi, and Alex Aiken. Exploring hidden dimensions in accelerating convolutional
neural networks. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of
Proceedings of Machine Learning Research. PMLR,
2018. 1, 10

Zhihao Jia, Matei Zaharia, and Alex Aiken. Beyond
data and model parallelism for deep neural networks.
In Proceedings of the 2nd Conference on Systems and
Machine Learning, SysML’19, 2019. 1, 2, 3, 10, 11, 13

Keshav Santhanam, Siddharth Krishna, Ryota Tomioka,
Tim Harris, and Matei Zaharia. DistIR: An Intermediate
Representation and Simulator for Efficient Neural Network Distribution. arXiv:2111.05426 [cs], November
2021. 7, 10, 14

unity: unite the algebra transformation and parallel

然后flexflow的github可以看到22年有个osdi直接广义把并行和代数变换建一个图，考虑了计算/并行/通讯。 [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/conference/osdi22/presentation/unger). In Proceedings of the Symposium on Operating Systems Design and Implementation (OSDI), July 2022.

**Can we use NAS related differentiable search algorithm to search the optimal of parallel plan?**

DART gradient descent based search algorithm. Make search sapce differentiable.(→softmax)

I was thinking that there is a differentiable method DARTS in Neural Architecture Search to search for the minimum value. That is, the n alternative discrete max of the original search for the optimal network structure becomes a differentiable softmax approximation, and then derives the derivative. The gradient of the train and the gradient of finding the optimal network structure are combined into one formula. Each time the gradient drops one step, the optimal network structure is found while training. If the distribution training is based on a calculation graph (like the latest flexflow 22 article on unity), can this similar method be used to search for the minimum value. Find the optimal parallel configuration while training. Now the search is generally dp/mcmc or something. Personally, I feel that this method should be quite insightful, not a small repair, but whether it can really work depends on whether it can be done.

我在想，Neural Architecture Search 里面有一个可微分的方法DARTS去搜最小值。就是原来搜索最优网络结构的n个备选的离散的max变成可微分的softmax近似，然后求导。train的梯度和找最优网络结构的梯度合并成一个式子，每次梯度下降一步就是，一边训练一边找最优网络结构。如果分布训练基于一个计算图的话（像最新flexflow22年那篇unity），是不是也可以用这个类似的方法搜最小值。一边训练一边找最优并行配置。现在搜索普遍是dp/mcmc还是什么。个人感觉这种方法应该挺有insight，不是小修补，但能不能真的效果好得看能不能做出来。

**resource-aware scheduling**

[https://www.usenix.org/system/files/osdi22-mohan.pdf](https://www.usenix.org/system/files/osdi22-mohan.pdf) 

我发现这篇文章Looking Beyond GPUs for DNN Scheduling on Multi-Tenant Clusters他貌似就是resource-aware的动态DNN scheduling。有个视频https://www.bilibili.com/video/BV1U24y1p78t/?share_source=copy_web&vd_source=986cc4a343d48f7717cede6222a5413a

**heterogeneous**

HetPipe: Enabling Large DNN Training on
(Whimpy) Heterogeneous GPU Clusters
through Integration of Pipelined Model
Parallelism and Data Parallelism

1. [OSDI'20] "[A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters](https://www.usenix.org/conference/osdi20/presentation/jiang)". Yimin Jiang, Yibo Zhu, Chang Lan, Bairen Yi, Yong Cui, Chuanxiong Guo.

[https://github.com/guanh01/CS692-mlsys](https://github.com/guanh01/CS692-mlsys)

[https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning)

[Awesome-System-for-Machine-Learning/training.md at master · HuaizhengZhang/Awesome-System-for-Machine-Learning (github.com)](https://github.com/HuaizhengZhang/Awesome-System-for-Machine-Learning/blob/master/training.md#training-system)

**question:**

pipedream is asynchronous, gradient descent is not synchronous, how to restore the accuracy loss?

papers:

1.

paper: [osdi22-zheng-lianmin.pdf (usenix.org)](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)

github: [alpa-projects/alpa: Training and serving large-scale neural networks (github.com)](https://github.com/alpa-projects/alpa)

OSDI talk slides:

[https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit#slide=id.g136a86a0982_0_105](https://docs.google.com/presentation/d/1CQ4S1ff8yURk9XmL5lpQOoMMlsjw4m0zPS6zYDcyp7Y/edit#slide=id.g136a86a0982_0_105)

document:

[Install Alpa — Alpa 0.2.2.dev53 documentation](https://alpa.ai/install.html)

issue:

[https://github.com/alpa-projects/alpa/issues/792](https://github.com/alpa-projects/alpa/issues/792)

[https://github.com/alpa-projects/alpa/issues/792](https://github.com/alpa-projects/alpa/issues/792)

question: 

The mapping between stage and device mesh seems not hard-ware aware, If that is the case, some device mesh with heavier workload might run on less powerful device mesh while some light workload might run on resourceful device mesh, which lead to larger idle time (resource waste) in Inter-Op Parallel pipeline. so i can wondering the factor that contributes to the current orchestration.

主要要research一下有没有什么办法来预测resource consumption 能做的话再去看怎么加

2.

paper: AAAI 2022

[[2112.08761v1] DISTREAL: Distributed Resource-Aware Learning in Heterogeneous Systems (arxiv.org)](https://arxiv.org/abs/2112.08761v1)

abstract:

We study the problem of distributed training of neural networks (NNs) on devices with heterogeneous, limited, and time-varying availability of computational resources. We present an adaptive, resource-aware, on-device learning mechanism, DISTREAL, which is able to fully and efficiently utilize the available resources on devices in a distributed manner, increasing the convergence speed. This is achieved with a dropout mechanism that dynamically adjusts the computational complexity of training an NN by randomly dropping filters of convolutional layers of the model. Our main contribution is the introduction of a design space exploration (DSE) technique, which finds Pareto-optimal per-layer dropout vectors with respect to resource requirements and convergence speed of the training. Applying this technique, each device is able to dynamically select the dropout vector that fits its available resource without requiring any assistance from the server. We implement our solution in a federated learning (FL) system, where the availability of computational resources varies both between devices and over time, and show through extensive evaluation that we are able to significantly increase the convergence speed over the state of the art without compromising on the final accuracy.

我们研究了在具有异构、有限和时变可用性的计算资源的设备上神经网络 (NN) 的分布式训练问题。 我们提出了一种自适应的、资源感知的设备上学习机制 DISTREAL，它能够以分布式方式充分有效地利用设备上的可用资源，从而提高收敛速度。 这是通过丢弃机制(dropout mechanism)实现的，该机制通过随机丢弃模型卷积层的过滤器来动态调整训练 NN 的计算复杂性。 我们的主要贡献是引入了设计空间探索 (DSE) 技术，该技术根据资源需求和训练收敛速度找到帕累托最优(Pareto-optimal)的每层 dropout 向量。 应用这种技术，每个设备都能够动态选择适合其可用资源的丢失向量，而无需服务器的任何帮助。 我们在联合学习 (FL) 系统中实施我们的解决方案，其中计算资源的可用性在设备之间和随着时间的推移而变化，并通过广泛的评估表明我们能够显着提高收敛速度而不影响现有技术 关于最终的准确性。

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1745e70c-1bd6-4632-b73f-f6d87c6dada1/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/75c412c6-cab7-421c-8de9-eb7520c53815/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/272e963c-edcf-4ef0-aea5-d5cee836ca74/Untitled.png)

1. Journal of Parallel and Distributed Computing 2022

[Evaluating execution time predictions on GPU kernels using an analytical model and machine learning techniques](https://www.sciencedirect.com/science/article/pii/S0743731522001903?casa_token=Kqi7243-EJsAAAAA:P9NoXtKwokKF11fN8OgGYys5r0gaF_gVweA7ZyMfYvSMkUkGQ37oMASLi49cnXp_0F9Oogk3Fg)

Predicting the performance of applications executed on GPUs is a great challenge and is essential
for efficient job schedulers. There are different approaches to do this, namely analytical modeling
and machine learning (ML) techniques. Machine learning requires large training sets and reliable
features, nevertheless it can capture the interactions between architecture and software without manual intervention.
In this paper, we compared a BSP-based analytical model to predict the time of execution of kernels
executed over GPUs. The comparison was made using three different ML techniques. The analytical model is based on the number of computations and memory accesses of the GPU, with additional information on cache usage obtained from profiling. The ML techniques Linear Regression, Support Vector Machine, and Random Forest were evaluated over two scenarios: first, data input or features for ML techniques were the same as the analytical model and, second, using a process of feature extraction, which used correlation analysis and hierarchical clustering. Our experiments were conducted with 20 CUDA kernels, 11 of which belonged to 6 real-world applications of the Rodinia benchmark suite, and the other were classical matrix-vector applications commonly used for benchmarking. We collected data over 9 NVIDIA GPUs in different machines.
We show that the analytical model performs better at predicting when applications scale regularly. For the analytical model a single parameter λ is capable of adjusting the predictions, minimizing the complex analysis in the applications. We show also that ML techniques obtained high accuracy when a process of feature extraction is implemented. Sets of 5 and 10 features were tested in two different ways, for unknown GPUs and for unknown Kernels. For ML experiments with a process of feature extractions, we got errors around 1.54% and 2.71%, for unknown GPUs and for unknown Kernels, respectively

预测在 GPU 上执行的应用程序的性能是一项巨大的挑战，也是必不可少的
用于高效的作业调度程序。 有不同的方法可以做到这一点，即分析建模
和机器学习 (ML) 技术。 机器学习需要大量的训练集和可靠的
功能，尽管如此，它无需手动即可捕获体系结构和软件之间的交互
干涉。
在本文中，我们比较了基于 BSP 的分析模型来预测内核的执行时间
在 GPU 上执行。 使用三种不同的 ML 技术进行了比较。 分析模型
基于 GPU 的计算和内存访问的数量，以及附加信息
关于从分析中获得的缓存使用情况。 机器学习技术线性回归、支持向量机、
和随机森林在两种情况下进行了评估：首先，ML 技术的数据输入或特征
与分析模型相同，其次，使用特征提取过程，该过程使用
相关分析和层次聚类。 我们的实验是用 20 个 CUDA 内核进行的，
其中 11 个属于 Rodinia 基准套件的 6 个实际应用，另一个是
通常用于基准测试的经典矩阵向量应用程序。 我们收集了超过 9 个 NVIDIA 的数据
不同机器中的 GPU。
我们表明，分析模型在预测应用程序何时定期扩展方面表现更好。 为了
单个参数 λ 的分析模型能够调整预测，最小化复杂度
应用中的分析。 我们还表明，ML 技术在处理过程中获得了高精度
实现了特征提取。 以两种不同的方式测试了 5 组和 10 组特征，因为
未知的 GPU 和未知的内核。 对于具有特征提取过程的 ML 实验，我们
对于未知的 GPU 和未知的内核，分别有大约 1.54% 和 2.71% 的错误

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bb5fcd6a-60e0-4920-9fb3-9cc7637b7b55/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d37a6252-908c-417f-a49f-cd9a57025ef9/Untitled.png)

1. 

Predicting Workflow Task Execution Time
in the Cloud Using A Two-Stage Machine
Learning Approach

1. [IEEE Xplore Full-Text PDF:](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8013738)

Abstract—Many techniques such as scheduling and resource provisioning rely on performance prediction of workflow tasks for varying input data. However, such estimates are difficult to generate in the cloud. This paper introduces a novel two-stage machine learning approach for predicting workflow task execution times for varying input data in the cloud. In order to achieve high accuracy predictions, our approach relies on parameters reflecting runtime information and two stages of predictions. Empirical results for four real world workflow applications and several commercial cloud providers demonstrate that our approach outperforms existing prediction methods. In our experiments, our approach respectively achieves a best-case and worst-case estimation error of 1.6 and 12.2 percent, while existing methods achieved errors beyond 20 percent (for some cases even over 50 percent) in more than 75 percent of the evaluated workflow tasks. In addition, we show that the models predicted by our approach for a specific cloud can be ported with low effort to new clouds with low errors by requiring only a small number of executions.

调度和资源供应等许多技术都依赖于对不同输入数据的工作流任务的性能预测。 然而，这样的估计很难在云中生成。 本文介绍了一种新颖的两阶段机器学习方法，用于预测云中不同输入数据的工作流任务执行时间。 为了实现高精度预测，我们的方法依赖于反映运行时信息和两个预测阶段的参数。 四个真实世界工作流应用程序和几个商业云提供商的实证结果表明，我们的方法优于现有的预测方法。 在我们的实验中，我们的方法分别实现了 1.6% 和 12.2% 的最佳情况和最坏情况估计误差，而现有方法在超过 75% 的评估工作流中实现了超过 20%（在某些情况下甚至超过 50%）的误差 任务。 此外，我们表明，我们的方法针对特定云预测的模型只需少量执行即可轻松移植到错误率较低的新云中。

5.Ali data center use data

fast.ai

pytorch lightning

### Yuanrui Zhang

预测model的gpu consumption

[https://www.microsoft.com/en-us/research/uploads/prod/2020/09/dnnmem.pdf](https://www.microsoft.com/en-us/research/uploads/prod/2020/09/dnnmem.pdf)

**Plan:**

get 2 machine gpu/cpu

do distributed data parallel, pipeline parallel,

zero(data parallel)/lightning(pipeline)

experience setting 

combine pipeline(within same node)/data(across nodes)

pretrain-finetune

only train last few layer

BERT

resnet, batch size

because care about training time.

pytorch white paper:

**Progress:**

1.BERT/GPT for CPU code:

[transformers/examples/pytorch/language-modeling at main · huggingface/transformers · GitHub](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling)

[Train 1 trillion+ parameter models — PyTorch Lightning 1.9.2 documentation (pytorch-lightning.readthedocs.io)](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/cdf4c22a-b3d9-49f4-8ff5-367f4faeaea9/Untitled.png)

1. Alibaba cluster data collected from production clusters in Alibaba for cluster management research

[https://github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata)

R**eference:**

Discussion on tooling for Distributed ML on hn front page today:

[https://news.ycombinator.com/item?id=34752489](https://news.ycombinator.com/item?id=34752489)

1. PyTorch's Distributed Data Parallel: [https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#combine-ddp-with-model-parallelism](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#combine-ddp-with-model-parallelism)
    - There is a section on combining with Model Parallel too
2. Pipeline parallelism: [https://pytorch.org/docs/stable/pipeline.html](https://pytorch.org/docs/stable/pipeline.html)
    1. Well known papers: PipeDream:[PipeDream: Generalized Pipeline Parallelism for DNN Training (microsoft.com)](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/pipedream.pdf), GPipe
3. ZeRO technique: Reduces redundant replication in Data parallel - [https://arxiv.org/pdf/1910.02054.pdf](https://arxiv.org/pdf/1910.02054.pdf)
4. PyTorch Lightening has a number of wrappers to enable distributed training
    - Eg: Advance model parallel feature using ZeRO: [https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html)
5. Some basic summary blogs:
    1. [https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)
    2. [https://siboehm.com/articles/22/pipeline-parallel-training](https://siboehm.com/articles/22/pipeline-parallel-training) (github of the project: [https://github.com/siboehm/ShallowSpeed](https://github.com/siboehm/ShallowSpeed))
6. Other topics we discussed today:
    1. Fault tolerance in training (Suppose we are training across 8 GPUs using any of the above techniques and one OOMs/fails what happens then?)
    2. Elasticity, autoscaling (Can you automatically reconfigure training to run on more/fewer GPUs? )
    3. DNN training jobs scheduling
7. Projects to explore: Horovod, Ray project ([https://github.com/ray-project/ray](https://github.com/ray-project/ray))

[Multi node PyTorch Distributed Training Guide For People In A Hurry (lambdalabs.com)](https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide)

[Optimize training performance with Reduction Server on Vertex AI | Google Cloud Blog](https://cloud.google.com/blog/topics/developers-practitioners/optimize-training-performance-reduction-server-vertex-ai)

[Faster distributed training with Google Cloud’s Reduction Server | Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/faster-distributed-training-with-google-clouds-reduction-server?hl=en)

[分布式训练  |  Vertex AI  |  Google Cloud](https://cloud.google.com/vertex-ai/docs/training/distributed-training?hl=zh-cn)

[Scalable multi-node deep learning training using GPUs in the AWS Cloud  | AWS Machine Learning Blog (amazon.com)](https://aws.amazon.com/cn/blogs/machine-learning/scalable-multi-node-deep-learning-training-using-gpus-in-the-aws-cloud/)

**computational graph representation**

In order to represent the parallel search space as a computational graph, we can model it using a directed acyclic graph (DAG) where each vertex represents a stage of computation and the edges denote the flow of data or model parameters between the stages. The nodes in this graph can be partitioned into different parallelization strategies, such as data parallelism, model parallelism, and pipeline parallelism.

For example, let's consider a simple graph G(V, E), where V is the set of vertices, and E is the set of edges. We can represent a parallel search space for a two-layer neural network as follows:

1. Data Parallelism (DP): Each layer is replicated across all available devices, and the input data is split into equal partitions. In this case, we can represent the parallel search space as two sets of nodes, where each set contains nodes for each device:

```

DP_1 -> DP_2 -> ... -> DP_n
 |
 v
DP_1'-> DP_2'-> ... -> DP_n'

```

Here, DP_i and DP_i' are nodes representing the computation of layer 1 and layer 2 on the i-th device, respectively.

1. Pipeline Parallelism (PP): Each layer is assigned to a different device, and the input data is processed in a pipelined manner across the devices. In this case, the parallel search space can be represented as a single set of nodes, where each node corresponds to a layer:

```

PP_1 -> PP_2

```

Here, PP_1 and PP_2 are nodes representing the computation of layer 1 and layer 2 on separate devices, respectively.

To represent both parallelization strategies in the same graph, we can create a graph-like structure with multiple layers and vertices:

```

  DP_1 -> DP_2 -> ... -> DP_n
   |      |            |
   v      v            v
  DP_1'-> DP_2'-> ... -> DP_n'
   |
   v
  PP_1 -> PP_2

```

We can then use the softmax function to assign probabilities to the edges between nodes and layers, which represent the likelihood of selecting a specific parallelization strategy. This continuous representation enables us to differentiate the parallelization search space and optimize the parallel configuration along with the model parameters.

**differentiable parallelization search space**

To use a computational graph representation for the differentiable parallelization search space, let's first define a graph G(V, E), where V is the set of vertices and E is the set of edges. In this graph, each vertex represents a computation stage, and the edges denote the flow of data or model parameters between the stages.

For each vertex v_i in V, we associate a set of N parallelization strategies, represented as α_i1, α_i2, ..., α_iN. We can then represent the search space as a matrix A of size |V| x N, where A_ij denotes the discrete choice for strategy j at stage i. We can transform this search space into a differentiable problem by applying the softmax function to each row of A:

softmax(A)_ij = exp(A_ij) / Σ(exp(A_ik)) for k in [1, N]

Here, the continuous approximation of the discrete choice is given by softmax(A)_ij for each strategy j at stage i.

Now, we can use this continuous representation of the search space in the context of a computational graph to describe the algebra transformations and parallel strategies. Each vertex v_i in the graph can be associated with a continuous approximation of its parallelization strategy, given by the row softmax(A)_i. These continuous approximations can then be used to compute the forward and backward passes in the computational graph while considering the different parallel strategies.

The rest of the optimization process, including gradient computation and end-to-end optimization, remains the same as in the previous response, with the parallel configuration matrix A being updated during the optimization.

This approach allows us to incorporate a graph representation of the parallelization search space into the differentiable optimization process, enabling the joint optimization of model training and parallelization strategies.

**bilevel optimization**

In the DARTS paper, the authors propose a bilevel optimization problem where they optimize the model's architecture and its weights simultaneously. To adapt this approach for parallelization strategies, we'll introduce a loss function that jointly evaluates the parallel strategy and the training loss.

Let's denote the model's weights by θ and the parallelization strategy represented as a matrix A. We'll define the loss function as L(θ, A), which is a combination of the training loss and the evaluation of the parallel strategy. We can write the joint optimization problem as:

minimize L(θ, A)
with respect to θ, A

This bilevel optimization problem can be approximated using gradient-based optimization. The gradients required for updating θ and A are as follows:

∇_θ L(θ, A) = dL(θ, A) / dθ
∇_A L(θ, A) = dL(θ, A) / dA

Here, ∇_θ L(θ, A) is the gradient with respect to the model parameters θ, while ∇_A L(θ, A) is the gradient with respect to the parallelization strategy matrix A.

To perform end-to-end optimization, we'll update both the model parameters and the parallelization strategy using gradient descent:

θ = θ - η_θ ∇_θ L(θ, A)
A = A - η_A ∇_A L(θ, A)

where η_θ and η_A are the learning rates for the model parameters and the parallelization strategy matrix A, respectively.

This approach allows us to jointly optimize the model training and the parallelization strategies using a gradient-based optimization method similar to the one used in the DARTS paper. Note that in practice, the loss function L(θ, A) should be designed to effectively balance the trade-offs between training loss and the evaluation of the parallel strategy.

_______________________________________________________________________________________

## Computational Graph Representation

To represent the parallel search space as a computational graph, we can model it using a directed acyclic graph (DAG) where each vertex represents a stage of computation and the edges denote the flow of data or model parameters between the stages. The nodes in this graph can be partitioned into different parallelization strategies, such as data parallelism, model parallelism, and pipeline parallelism.

For example, let's consider a simple graph G(V, E), where V is the set of vertices, and E is the set of edges. We can represent a parallel search space for a two-layer neural network as follows:

- Data Parallelism (DP): Each layer is replicated across all available devices, and the input data is split into equal partitions. In this case, we can represent the parallel search space as two sets of nodes, where each set contains nodes for each device:

```
DP_1 -> DP_2 -> ... -> DP_n
 |
 v
DP_1'-> DP_2'-> ... -> DP_n'

```

Here, DP_i and DP_i' are nodes representing the computation of layer 1 and layer 2 on the i-th device, respectively.

- Pipeline Parallelism (PP): Each layer is assigned to a different device, and the input data is processed in a pipelined manner across the devices. In this case, the parallel search space can be represented as a single set of nodes, where each node corresponds to a layer:

```
PP_1 -> PP_2

```

Here, PP_1 and PP_2 are nodes representing the computation of layer 1 and layer 2 on separate devices, respectively.

To represent both parallelization strategies in the same graph, we can create a graph-like structure with multiple layers and vertices:

```
  DP_1 -> DP_2 -> ... -> DP_n
   |      |            |
   v      v            v
  DP_1'-> DP_2'-> ... -> DP_n'
   |
   v
  PP_1 -> PP_2

```

We can then use the softmax function to assign probabilities to the edges between nodes and layers, which represent the likelihood of selecting a specific parallelization strategy. This continuous representation enables us to differentiate the parallelization search space and optimize the parallel configuration along with the model parameters.

## Differentiable Parallelization Search Space

To use a computational graph representation for the differentiable parallelization search space, let's first define a graph G(V, E), where V is the set of vertices and E is the set of edges. In this graph, each vertex represents a computation stage, and the edges denote the flow of data or model parameters between the stages.

For each vertex v_i in V, we associate a set of N parallelization strategies, represented as α_i1, α_i2, ..., α_iN. We can then represent the search space as a matrix A of size |V| x N, where A_ij denotes the discrete choice for strategy j at stage i. We can transform this search space into a differentiable problem by applying the softmax function to each row of A:

softmax(A)_ij = exp(A_ij) / Σ(exp(A_ik)) for k in [1, N]

Here, the continuous approximation of the discrete choice is given by softmax(A)_ij for each strategy j at stage i.

Now, we can use this continuous representation of the search space in the context of a computational graph to describe the algebra transformations and parallel strategies. Each vertex v_i in the graph can be associated with a continuous approximation of its parallelization strategy, given by the row softmax(A)_i. These continuous approximations can then be used to compute the forward and backward passes in the computational graph while considering the different parallel strategies.

The rest of the optimization process, including gradient computation and end-to-end optimization, remains the same as in the previous response, with the parallel configuration matrix A being updated during the optimization.

This approach allows us to incorporate a graph representation of the parallelization search space into the differentiable optimization process, enabling the joint optimization of model training and parallelization strategies.

## Bilevel Optimization

In the DARTS paper, the authors propose a bilevel optimization problem where they optimize the model's architecture and its weights simultaneously. To adapt this approach for parallelization strategies, we'll introduce a loss function that jointly evaluates the parallel strategy and the training loss.

Let's denote the model's weights by θ and the parallelization strategy represented as a matrix A. We'll define the loss function as L(θ, A), which is a combination of the training loss and the evaluation of the parallel strategy. We can write the joint optimization problem as:

minimize L(θ, A)
with respect to θ, A

This bilevel optimization problem can be approximated using gradient-based optimization. The gradients required for updating θ and A are as follows:

∇_θ L(θ, A) = dL(θ, A) / dθ
∇_A L(θ, A) = dL(θ, A) / dA

Here, ∇_θ L(θ, A) is the gradient with respect to the model parameters θ, while ∇_A L(θ, A) is the gradient with respect to the parallelization strategy matrix A.

To perform end-to-end optimization, we'll update both the model parameters and the parallelization strategy using gradient descent:

θ = θ - η_θ ∇_θ L(θ, A)
A = A - η_A ∇_A L(θ, A)

where η_θ and η_A are the learning rates for the model parameters and the parallelization strategy matrix A, respectively.

This approach allows us to jointly optimize the model training and the parallelization strategies using a gradient-based optimization method similar to the one used in the DARTS paper. Note that in practice, the loss function L(θ, A) should be designed to effectively balance the trade-offs between training loss and the evaluation of the parallel strategy.
