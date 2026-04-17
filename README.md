# COTA: Cluster Optimal Transport Alignment for Cross-Domain Recommendation

## Abstract

Cross-domain recommendation (CDR) has emerged as a promising solution to the user cold-start problem by transferring preference knowledge from an information-rich source domain to a data-scarce target domain. A central challenge in CDR is generating accurate recommendations for nonoverlapping users, those who interact only in the source domain and have no interaction history in the target domain. Existing mapping-based CDR methods primarily rely on overlapping users and source-domain interaction sequences to learn a preference transfer function, leaving the rich structural information encoded in pre-trained target-domain embeddings largely unexploited. In this paper, we propose COTA (Cluster-based Optimal Transport Alignment), a novel framework for cross-domain cold-start recommendation. COTA consists of two dedicated modules. First, the Prototype Optimal Transport Alignment (POTA) module independently clusters both the source and target domains from their pre-trained embeddings, projects source prototype representations into the target space via a multi-layer perceptron, and employs an optimal transport (OT) solver to learn a principled prototype-to-prototype alignment. Gumbel-Softmax sampling over the resulting transport plan provides each source user with a differentiable target-domain destination, injecting target-structural priors in a fully unsupervised, end-to-end trainable manner. Second, the Target-Aware Preference Fusion (TAPF) module aggregates four complementary signals, namely the source user embedding, source prototype, OT-derived target destination, and mapped prototype representation, through a Mixture-of-Experts (MoE) architecture to produce the final cross-domain user representation. A key advantage of COTA is that it directly exploits pre-trained target user embeddings as prototype anchors, eliminating the need for item interaction sequence modeling and thereby reducing computational overhead while retaining expressive capacity. Extensive experiments on three cross-domain benchmarks derived from the real-world public datasets demonstrate that COTA consistently outperforms state-of-the-art baselines across all settings.

Keywords: Cross-domain Recommendation \sep Cold-start Problem \sep Optimal Transport \sep Mixture-of-Experts \sep Prototype Learning

## Implementation

Implementation steps of task 1 (Movie-Music, beta = 20%):
1) Assuming you already installed pytorch, install -> requirement.txt <-.
2) Goto -> variable.py <-, change the variable -> absolute_path <- to your own project root, Ray Tune worker require absolute path.
3) Run Jupyter notebook -> main-train.ipynb <-.
Note: In training process, Ray Tune save many checkpoints under path ~/ray_results/, they take a lot of space, clean them after training.
Screenshot of training process:
[Project Logo](train-example.jpg)

To implement this model to other task from the paper, you need to run preprocess and pretrain before training, read instruction below:
1) Download dataset & put in ./data/, e.g. ./data/amazon/book/reviews_Books_5.json, then register the path in -> variable.py <-.
2) Read the example usages which are commented out in -> preprocess.ipynb <-.
3) There are 3 function, each does different job, restore the commented method and run.
4) After done data preprocess, goto -> pretrain.ipynb <-, change the variables above to desire value, then run the notebook.
Note: -> preprocess.ipynb <- is only ready to process Amazon-2014 datasets, which registered folder path in -> variable.py <-.
