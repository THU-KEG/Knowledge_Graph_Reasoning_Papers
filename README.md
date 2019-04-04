# Knowledge Graph Reasoning Papers

## (h, r, ?)
Predict the missing tail entity and corresponding supporting paths in one triple.

1. **Multi-Hop Knowledge Graph Reasoning with Reward Shaping.** *Xi Victoria Lin, Richard Socher, Caiming Xiong.* EMNLP 2018. [paper](https://aclweb.org/anthology/D18-1362) [code](https://github.com/salesforce/MultiHopKG)
> Multi-hop knowledge graph reasoning learned via policy gradient with reward shaping and action dropout.

1. **GO FOR A WALK AND ARRIVE AT THE ANSWER: REASONING OVER PATHS IN KNOWLEDGE BASES USING REINFORCEMENT LEARNING.**

1. **M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search.**

1. **Differentiable Learning of Logical Rules for Knowledge Base Reasoning.**

1. **DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning.**







## (h, ?, t)
1. **Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks.**

1. **Variational Knowledge Graph Reasoning.**

1. **Random walk inference and learning in a large scale knowledge base.**

1. **Compositional vector space models for knowledge base inference.**





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
