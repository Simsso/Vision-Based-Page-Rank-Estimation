# Vision-based Page Rank Estimation with Graph Networks

_[Timo I. Denk](https://timodenk.com/), [Samed Güner](https://twitter.com/samedguener)  
(published in May 2019)_

This repository contains the code that was written in our student research project on **vision-based page rank estimation with graph networks**. _Code_ is four different pieces of software: GraphNets library, RankPredictor, human evaluation tool, and data crawler.

The **GraphNets library** (`/graph_nets/`) is our Python implementation of the graph network (GN) framework from [Battiglia et al. (2018)](). It was developed in such a way that other people can use it in their projects too. We use the library in our research to create GNs, train them, and perform inference.

The **RankPredictor** code (`/rank_predictor/`) is a collection of Python scripts and classes which are the actual implementation of our method. The code heavily relies on the GraphNets library and implements many of its ab- stract classes. The code contains our models and can be used to reproduce our results.

The **human evaluation tool** (`/human_eval/`) is a lightweight Node.js server with a simplistic frontend that was used to determine the human performance on the page rank task. It shows pairs of web pages and asks the user to rank them. The answers are being stored so the human accuracy on pairwise page rank estimation can be output.

The **data crawler** (`/datacrawler`) served the purpose of creating the dataset. It visits web pages, takes screenshots, collects meta information, and stores the collected data. It was developed in C++ with special focus on scalability and modularity.

## Quick Links: [Report]() · [Dataset]() · [Blog Post]() · [Slides]()
