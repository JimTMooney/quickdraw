# quickdraw

## Getting Started
To begin, we first have to download the data from google cloud. This can be done by running the following commands

```
chmod 777 quick_download.sh
./quick_download.sh
```
This will pull in 75k (70k train, 2.5k validation, 2.5k test) total sketches across 345 sketch classes. With the data in hand, we construct a subset of the dataset by considering 1k train, 100 validation, 100 test, as in [[1]](#1). Each sketch is stored as a tuple of (x_t, y_t, n_t) where x_t and y_t give the x and y offsets relative to the previous timestep and n denotes whether a new stroke has begun (if a new stroke has begun, this value is given by a 1 when a new stroke has begun and 0 otherwise). Each sketch is normalized such that the x,y coordinates of all points in the sketch are set to be between -.5 and .5. The new classes of sketches are stored under the 'data_full' directory. In addition to these sketches, the individual strokes are also stored in a separate directory. These can be found at 'data_strokes'. Both of these directories are created by running the following command from the terminal

```
python3 create_dataset.py
```

Both of these datasets also trim the size of the sketches to be sequences of no more than 150 key points.


<a id="1">[1]</a> 
Multi-Graph Transformer for Free-Hand Sketch Recognition by Peng Xu, Chaitanya K Joshi, Xavier Bresson, ArXiv 2019
