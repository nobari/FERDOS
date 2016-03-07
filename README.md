# FERDOS
### Fast Random Graph Generation

## Abstract

Today, several database applications call for the generation of random graphs. A fundamental, versatile random graph model adopted for that purpose is the Erdős–Rényi Γv,p model. This model can be used for directed, undirected, and multipartite graphs, with and without self-loops; it induces algorithms for both graph generation and sampling, hence is useful not only in applications necessitating the generation of random structures but also for simulation, sampling and in randomized algorithms. However, the commonly advocated algorithm for random graph generation under this model performs poorly when generating large graphs, and fails to make use of the parallel processing capabilities of modern hardware. In this paper, we propose PPreZER, an alternative, data parallel algorithm for random graph generation under the Erdős–Rényi model, designed and implemented in a graphics processing unit (GPU). We are led to this chief contribution of ours via a succession of seven intermediary algorithms, both sequential and parallel. Our extensive experimental study shows an average speedup of 19 for PPreZER with respect to the baseline algorithm.

## Code
The code consists of two versions, i.e. sequential and parallel. The sequential code executes on CPU and the parallel code executes on GPU and each code is inside the CPU and GPU folders, respectively.

## Please cite the paper:
[Sadegh Nobari](http://bit.ly/NOB-GS), Xuesong Lu, Panagiotis Karras, and Stéphane Bressan,
["Fast Random Graph Generation"](http://portal.acm.org/citation.cfm?id=1951406),
Proc. of the 14th Intl Conf. on Extending Database Technology (EDBT'11), Uppsala, Sweden
### BibTeX record
@inproceedings{Nobari:2011,
  author    = {Sadegh Nobari and
               Xuesong Lu and
               Panagiotis Karras and
               St{\'{e}}phane Bressan},
  title     = {Fast random graph generation},
  booktitle = {{EDBT} 2011, 14th International Conference on Extending Database Technology,
               Uppsala, Sweden, March 21-24, 2011, Proceedings},
  pages     = {331--342},
  year      = {2011},
  url       = {http://doi.acm.org/10.1145/1951365.1951406},
  doi       = {10.1145/1951365.1951406}
}
 
## Furthur reading
[Full paper](http://portal.acm.org/citation.cfm?id=1951406)

[Wikipedia](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)

## Email
s @ s q n c o . c o m 
