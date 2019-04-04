# Knowledge Graph Reasoning Papers

## (h, r, ?)
Predict the missing tail entity and corresponding supporting paths in one triple.

1. **DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning.** *Wenhan Xiong, Thien Hoang, William Yang Wang.* EMNLP  2017. [paper](https://www.aclweb.org/anthology/D17-1060) [code](https://github.com/xwhan/DeepPath)
    > They describe a novel reinforcement learning framework for learning multi-hop relational paths: we use a policy-based agent with continuous states based on knowledge graph embeddings, which reasons in a KG vector-space by sampling the most promising relation to extend its path.

1. **Differentiable Learning of Logical Rules for Knowledge Base Reasoning.** *Fan Yang, Zhilin Yang, William W. Cohen.* NIPS 2017. [paper](https://papers.nips.cc/paper/6826-differentiable-learning-of-logical-rules-for-knowledge-base-reasoning.pdf) [code](https://github.com/fanyangxyz/Neural-LP)
    > They propose a framework, Neural Logic Programming, that combines the parameter and structure learning of first-order logical rules in an end-to-end differentiable model. They design a neural controller system that learns to compose differentiable reasoning operations.

1. **Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning.** *Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy,  Alex Smola, Andrew McCallum.* ICLR 2018. [paper](https://arxiv.org/pdf/1711.05851.pdf) [code](https://github.com/shehzaadzd/MINERVA)
    > MINERVA addresses the much more difficult and practical task of answering questions where the relation is known, but only one entity. 

1. **M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search.** *Yelong Shen, Jianshu Chen, Po-Sen Huang, Yuqing Guo, Jianfeng Gao.* NIPS 2018. [paper](https://papers.nips.cc/paper/7912-m-walk-learning-to-walk-over-graphs-using-monte-carlo-tree-search.pdf) [code](https://github.com/yelongshen/GraphWalk)
    > M-Walk learns to walk over a graph towards a desired target node for given input query and source nodes. Specifically, this paper proposes a novel neural architecture that encodes the state into a vector representation, and maps it to Q-values and a policy.
    
1. **Multi-Hop Knowledge Graph Reasoning with Reward Shaping.** *Xi Victoria Lin, Richard Socher, Caiming Xiong.* EMNLP 2018. [paper](https://aclweb.org/anthology/D18-1362) [code](https://github.com/salesforce/MultiHopKG)
    > This paper proposes two modeling advances for end-to-end RL-based knowledge graph query answering: (1) reward shaping via graph completion and (2) action dropout. 


## (h, ?, t)
Given head and tail entity and paths between them, predict the missing relation.

1. **Random walk inference and learning in a large scale knowledge base.** *Ni Lao, Tom Mitchell, William W. Cohen.* EMNLP 2011. [paper](https://www.cs.cmu.edu/~tom/pubs/lao-emnlp11.pdf) 
    > This paper shows that the system can learn to infer different target relations by tuning the weights associated with random walks that follow different paths through the graph, using a version of the Path Ranking Algorithm.
    
1. **Compositional vector space models for knowledge base inference.** *Arvind Neelakantan, Benjamin Roth, Andrew McCallum.* ACL 2015. [paper](https://www.aclweb.org/anthology/P15-1016) 
    > This paper presents an approach that reasons about conjunctions of multi-hop relations non-atomically, composing the implications of a path using a recurrent neural network (RNN) that takes as inputs vector embeddings of the binary relation in the path.

1. **Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks.** *Rajarshi Das, Arvind Neelakantan, David Belanger, Andrew McCallum.* EACL 2017. [paper](https://www.aclweb.org/anthology/E17-1013) [code](https://rajarshd.github.io/ChainsofReasoning/)
    > This paper proposes three significant modeling advances: (1) they learn to jointly reason about relations, entities, and entity-types; (2) they use neural attention modeling to incorporate multiple paths; (3) they learn to share strength in a single RNN that represents logical composition across all relations.
    
1. **Variational Knowledge Graph Reasoning.** *Wenhu Chen, Wenhan Xiong, Xifeng Yan, William Yang Wang.* NAACL 2018. [paper](https://aclweb.org/anthology/N18-1165) 
    > This paper tackles apractical query answering task involving predicting the relation of a given entity pair. They frame this prediction problem as an inference problem in a probabilistic graphical model andaim at resolving it from a variational inference perspective.


## Rules Learning

1. **Differentiable Learning of Logical Rules for Knowledge Base Reasoning.** *Fan Yang, Zhilin Yang, William W. Cohen.* NIPS 2017. [paper](https://papers.nips.cc/paper/6826-differentiable-learning-of-logical-rules-for-knowledge-base-reasoning.pdf) [code](https://github.com/fanyangxyz/Neural-LP)
    > They propose a framework, Neural Logic Programming, that combines the parameter and structure learning of first-order logical rules in an end-to-end differentiable model. They design a neural controller system that learns to compose differentiable reasoning operations.

1. **Scalable Rule Learning via Learning Representation.** *Pouya Ghiasnezhad Omran, Kewen Wang, Zhe Wang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0297.pdf)
    > This paper presents a new approach RLvLR to learning rules from KGs by using the technique of embedding in representation learning together with a new sampling method. For massive KGs with hundreds of predicates and over 10M facts, RLvLR is much faster and can learn much more quality rules than major systems.

1. **Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning.** *Wen Zhang, Bibek Paudel, Liang Wang, Jiaoyan Chen, Hai Zhu, Wei Zhang, Abraham Bernstein, Huajun Chen.* WWW 2019. [paper](https://arxiv.org/pdf/1903.08948.pdf)
    > This paper explores how embedding and rule learning can be combined together and complement each other’s difficulties with their advantages.

1. **RUGE: Knowledge Graph Embedding with Iterative Guidance from Soft Rules.** *Shu Guo, Quan Wang, Lihong Wang, Bin Wang, Li Guo.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.11231.pdf) [code](https://github.com/iieir-km/RUGE)
    > RUGE is the first work that models interactions between embedding learning and logical inference in a principled framework. It enables an embedding model to learn simultaneously from labeled triples, unlabeled triples and soft rules in an iterative manner.

1. **Rule Learning from Knowledge Graphs Guided by Embedding Models.** *V. Thinh Ho, D. Stepanova, M. Gad-Elrab, E. Kharlamov, G. Weikum.* ISWC 2018. [paper](https://people.mpi-inf.mpg.de/~dstepano/conferences/ISWC2018/paper/ISWC2018paper.pdf)
    > They propose a rule learning method that utilizes probabilistic representations of missing facts. In particular, they iteratively extend rules induced from a KG by relying on feedback from a precomputed embedding model over the KG and external information sources including text corpora.





## Complex Natural Language Query

1. (Dataset: WikiTableQuestions) **Compositional Semantic Parsing on Semi-Structured Tables.** *Panupong Pasupat, Percy Liang.* ACL 2015. [paper](https://aclweb.org/anthology/P15-1142) [code](https://github.com/ppasupat/WikiTableQuestions)
    > This paper creates a dataset of 22,033 complex questions on Wikipedia tables. They propose a logical-form driven parsing algorithm guided by strong typing constraints.

1. (Dataset: MetaQA) **Variational Reasoning for Question Answering with Knowledge Graph.** *Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, Le Song.* AAAI 2018. [paper](https://arxiv.org/pdf/1709.04071.pdf) [data](https://github.com/yuyuz/MetaQA)
    > This paper proposes an end-to-end variational learning algorithm which can handle noise in questions, and learn multi-hop reasoning simultaneously. Besides, they derive a series of new benchmark datasets named MetaQA, including questions for multi-hop reasoning, questions paraphrased by neural translation model, and questions in human voice.





## Others

1. **Embedding Logical Queries on Knowledge Graphs.** *William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, Jure Leskovec.* NIPS 2018. [paper](http://papers.nips.cc/paper/7473-embedding-logical-queries-on-knowledge-graphs.pdf)
    > This paper aims to develop techniques that can go beyond simple edge prediction and handle more complex logical queries, which might involve multiple unobserved edges, entities, and variables. They introduce a framework to efficiently make predictions about conjunctive logical queries --— a flexible but tractable subset of first-order logic —-- on incomplete knowledge graphs.
