# Knowledge Graph Reasoning Papers

## Multi-Hop Reasoning (h, r, ?)
Predict the missing tail entity and corresponding supporting paths in one triple.

1. **Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning.** *Rajarshi Das, Shehzaad Dhuliawala, Manzil Zaheer, Luke Vilnis, Ishan Durugkar, Akshay Krishnamurthy,  Alex Smola, Andrew McCallum.* ICLR 2018. [paper](https://arxiv.org/pdf/1711.05851.pdf) [code](https://github.com/shehzaadzd/MINERVA)

2. **M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search.** *Yelong Shen, Jianshu Chen, Po-Sen Huang, Yuqing Guo, Jianfeng Gao.* NeurIPS 2018. [paper](https://papers.nips.cc/paper/7912-m-walk-learning-to-walk-over-graphs-using-monte-carlo-tree-search.pdf) [code](https://github.com/yelongshen/GraphWalk)
    
3. **Multi-Hop Knowledge Graph Reasoning with Reward Shaping.** *Xi Victoria Lin, Richard Socher, Caiming Xiong.* EMNLP 2018. [paper](https://aclweb.org/anthology/D18-1362) [code](https://github.com/salesforce/MultiHopKG)

4. **Adapting Meta Knowledge Graph Information for Multi-Hop Reasoning over Few-Shot Relations.** *Xin Lv, Yuxian Gu, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu.* EMNLP 2019. [paper](https://www.aclweb.org/anthology/D19-1334.pdf) [code](https://github.com/THU-KEG/MetaKGR)

5. **DIVINE: A Generative Adversarial Imitation Learning Framework for Knowledge Graph Reasoning.** *Ruiping Li, Xiang Cheng.* EMNLP 2019. [paper](https://www.aclweb.org/anthology/D19-1266.pdf) [code](https://github.com/BUPT-Data-Intelligence-Lab/DIVINE)

6. **Reasoning on Knowledge Graphs with Debate Dynamics.** *Marcel Hildebrandt, Jorge Andres Quintero Serna, Yunpu Ma, Martin Ringsquandl, Mitchell Joblin, Volker Tresp.* AAAI 2020. [paper](https://ojs.aaai.org/index.php/AAAI/article/download/6600/6454) [code](https://github.com/m-hildebrandt/R2D2)

7. **Reasoning Like Human: Hierarchical Reinforcement Learning for Knowledge Graph Reasoning.** *Guojia Wan, Shirui Pan, Chen Gong, Chuan Zhou, Gholamreza Haffari.* IJCAI 2020. [paper](https://www.ijcai.org/Proceedings/2020/0267.pdf) 

8. **Learning Collaborative Agents with Rule Guidance for Knowledge Graph Reasoning.** *Deren Lei1, Gangrong Jiang, Xiaotao Gu, Kexuan Sun, Yuning Mao, Xiang Ren.* EMNLP 2020. [paper](https://www.aclweb.org/anthology/2020.emnlp-main.688.pdf) [code](https://github.com/m-hildebrandt/R2D2)

9. **Dynamic Anticipation and Completion for Multi-Hop Reasoning over Sparse Knowledge Graph.** *Xin Lv, Xu Han, Lei Hou, Juanzi Li, Zhiyuan Liu, Wei Zhang, Yichi Zhang, Hao Kong, Suhui Wu.* EMNLP 2020. [paper](https://www.aclweb.org/anthology/2020.emnlp-main.459.pdf) [code](https://github.com/THU-KEG/DacKGR)

## Multi-Hop Reasoning (h, ?, t)
Given head and tail entity and paths between them, predict the missing relation.

1. **Random walk inference and learning in a large scale knowledge base.** *Ni Lao, Tom Mitchell, William W. Cohen.* EMNLP 2011. [paper](https://www.cs.cmu.edu/~tom/pubs/lao-emnlp11.pdf) 
    
2. **Compositional vector space models for knowledge base inference.** *Arvind Neelakantan, Benjamin Roth, Andrew McCallum.* ACL 2015. [paper](https://www.aclweb.org/anthology/P15-1016) 

3. **Chains of Reasoning over Entities, Relations, and Text using Recurrent Neural Networks.** *Rajarshi Das, Arvind Neelakantan, David Belanger, Andrew McCallum.* EACL 2017. [paper](https://www.aclweb.org/anthology/E17-1013) [code](https://rajarshd.github.io/ChainsofReasoning/)

4. **DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning.** *Wenhan Xiong, Thien Hoang, William Yang Wang.* EMNLP  2017. [paper](https://www.aclweb.org/anthology/D17-1060) [code](https://github.com/xwhan/DeepPath)
    
5. **Variational Knowledge Graph Reasoning.** *Wenhu Chen, Wenhan Xiong, Xifeng Yan, William Yang Wang.* NAACL 2018. [paper](https://aclweb.org/anthology/N18-1165) 

5. **Incorporating Graph Attention Mechanism into Knowledge Graph Reasoning Based on Deep Reinforcement Learning.** *Heng Wang, Shuangyin Li, Rong Pan, Mingzhi Mao.* EMNLP 2019. [paper](https://www.aclweb.org/anthology/D19-1264/) [code](https://github.com/jimmywangheng/AttnPath)

## Rule Mining/Learning

1. **Fast rule mining in ontological knowledge bases with AMIE+.** *Luis Galárraga, Christina Teflioudi, Katja Hose, Fabian M. Suchanek.* VLDB Journal 2015. [paper](https://link.springer.com/article/10.1007/s00778-015-0394-1) [code](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie)

2. **Scalable Rule Learning via Learning Representation.** *Pouya Ghiasnezhad Omran, Kewen Wang, Zhe Wang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0297.pdf) [code](https://www.ict.griffith.edu.au/aist/RLvLR/)

3. **Rule Learning from Knowledge Graphs Guided by Embedding Models.** *V. Thinh Ho, D. Stepanova, M. Gad-Elrab, E. Kharlamov, G. Weikum.* ISWC 2018. [paper](https://people.mpi-inf.mpg.de/~dstepano/conferences/ISWC2018/paper/ISWC2018paper.pdf) [code](http://people.mpi-inf.mpg.de/~gadelrab/RuLES/)

4. **Anytime Bottom-Up Rule Learning for Knowledge Graph Completion.** *Christian Meilicke, Melisachew Wudage Chekol, Daniel Ruffinelli, Heiner Stuckenschmidt.* IJCAI 2019. [paper](https://www.ijcai.org/Proceedings/2019/0435.pdf) [code](http://web.informatik.uni-mannheim.de/AnyBURL/)

## Rule-based Reasoning

1. **Differentiable Learning of Logical Rules for Knowledge Base Reasoning.** *Fan Yang, Zhilin Yang, William W. Cohen.* NeurIPS 2017. [paper](https://papers.nips.cc/paper/6826-differentiable-learning-of-logical-rules-for-knowledge-base-reasoning.pdf) [code](https://github.com/fanyangxyz/Neural-LP)

2. **End-to-End Differentiable Proving.** *Tim Rocktäschel, Sebastian Riedel.* NeurIPS 2017. [paper](https://arxiv.org/pdf/1705.11040.pdf) [code](https://github.com/uclnlp/ntp)

3. **DRUM: End-To-End Differentiable Rule Mining On Knowledge Graphs.** *Ali Sadeghian, Mohammadreza Armandpour, Patrick Ding, Daisy Zhe Wang.* NeurIPS 2019. [paper](https://papers.nips.cc/paper/2019/file/0c72cb7ee1512f800abe27823a792d03-Paper.pdf) [code](https://github.com/alisadeghian/DRUM)

4. **RNNLogic: Learning Logic Rules for Reasoning on Knowledge Graphs.** *Meng Qu, Junkun Chen, Louis-Pascal Xhonneux, Yoshua Bengio, Jian Tang.* Arxiv 2020. [paper](https://openreview.net/pdf?id=tGZu6DlbreV) 

4. **Learning Reasoning Strategies in End-to-End Differentiable Proving.** *Pasquale Minervini, Sebastian Riedel, Pontus Stenetorp, Edward Grefenstette, Tim Rocktäschel.* ICML 2020. [paper](https://proceedings.icml.cc/static/paper_files/icml/2020/3569-Paper.pdf) [code](https://github.com/uclnlp/ctp)


## Rule-enhanced Knowledge Graph Embedding

1. **RUGE: Knowledge Graph Embedding with Iterative Guidance from Soft Rules.** *Shu Guo, Quan Wang, Lihong Wang, Bin Wang, Li Guo.* AAAI 2018. [paper](https://arxiv.org/pdf/1711.11231.pdf) [code](https://github.com/iieir-km/RUGE)

2. **Iteratively Learning Embeddings and Rules for Knowledge Graph Reasoning.** *Wen Zhang, Bibek Paudel, Liang Wang, Jiaoyan Chen, Hai Zhu, Wei Zhang, Abraham Bernstein, Huajun Chen.* WWW 2019. [paper](https://arxiv.org/pdf/1903.08948.pdf)

3. **Probabilistic Logic Neural Networks for Reasoning.** *Meng Qu, Jian Tang.* NeurIPS 2019. [paper](https://papers.nips.cc/paper/2019/file/13e5ebb0fa112fe1b31a1067962d74a7-Paper.pdf) [code](https://github.com/DeepGraphLearning/pLogicNet)


## Complex Logic Query

1. **Embedding Logical Queries on Knowledge Graphs.** *William L. Hamilton, Payal Bajaj, Marinka Zitnik, Dan Jurafsky, Jure Leskovec.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7473-embedding-logical-queries-on-knowledge-graphs.pdf) [code](https://github.com/williamleif/graphqembed)

1. **Embedding Logical Queries on Knowledge Graphs.** *Hongyu Ren, Weihua Hu, Jure Leskovec.* ICLR 2019. [paper](https://openreview.net/forum?id=BJgr4kSFDS) [code](https://github.com/hyren/query2box)

1. **Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs.** *Hongyu Ren, Jure Leskovec.* NeurIPS 2020. [paper](https://papers.nips.cc/paper/2020/file/e43739bba7cdb577e9e3e4e42447f5a5-Paper.pdf) [code](http://snap.stanford.edu/betae)

## Complex Natural Language Query

1. (Dataset: WikiTableQuestions) **Compositional Semantic Parsing on Semi-Structured Tables.** *Panupong Pasupat, Percy Liang.* ACL 2015. [paper](https://aclweb.org/anthology/P15-1142) [code](https://github.com/ppasupat/WikiTableQuestions)

2. (Dataset: MetaQA) **Variational Reasoning for Question Answering with Knowledge Graph.** *Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, Le Song.* AAAI 2018. [paper](https://arxiv.org/pdf/1709.04071.pdf) [data](https://github.com/yuyuz/MetaQA)