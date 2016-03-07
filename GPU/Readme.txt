 This code generates random Erdos Renyi graph using cuda.
 The corresponding author is Sadegh Nobari

 If use please cite:
 @inproceedings{Nobari:2011,
 author = {Nobari, Sadegh and Lu, Xuesong and Karras, Panagiotis and Bressan, St\'{e}phane},
 title = {Fast random graph generation},
 booktitle = {Proceedings of the 14th International Conference on Extending Database Technology},
 series = {EDBT/ICDT '11},
 year = {2011},
 isbn = {978-1-4503-0528-0},
 location = {Uppsala, Sweden},
 pages = {331--342},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/1951365.1951406},
 doi = {http://doi.acm.org/10.1145/1951365.1951406},
 acmid = {1951406},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Erd\H{o}s-r\'{e}nyi, Gilbert, parallel algorithm, random graphs},
 } 



After introducing the CURAND library in 2011, it is recommended to use the CURAND uniform random number generator.
please check the comments in the initialization sections for more detail.
In kernel function RND simply generetes a uniform random number.




 Last update 19 Jun 2011