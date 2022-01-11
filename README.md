# quickdraw

## Getting Started
To begin, we first have to download the data from google cloud. This can be done by running the following commands

```
chmod 777 quick_download.sh
./quick_download.sh
```
This will pull in 75k (70k train, 2.5k validation, 2.5k test) total sketches across 345 sketch classes. With the data in hand, we construct a subset of the dataset by considering 1k train, 100 validation, 100 test, as in [[1]](#1).


<a id="1">[1]</a> 
Multi-Graph Transformer for Free-Hand Sketch Recognition by Peng Xu, Chaitanya K Joshi, Xavier Bresson, ArXiv 2019
