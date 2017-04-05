## Image Geocoding with PostgreSQL (IGP) ##

IGP is a python solution of Image-based Location Recognition (ILR) based on the research [1]. ILR receives as input one or more photographs and outputs their location. At the core of ILR are image features (2D patches which together best describe the scene contained within images). In an preparation stage, image features are extracted from database images and the Structure-from-Motion model (3D) is built. In an execution stage, image features are extracted from query images and compared to the database. With the SFM model available, direct 2D-3D matching is possible, accelerating the overall process of querying images and allowing pose estimation (calculus of position and orientation). However, this query process is easily hindered by outliers and accurately estimating the pose is not easily done when scaling the database into large datasets.

Research such as [1] focus on achieving the best performance at large scale by carefully selecting, organizing and prioritizing database data to relieve the complexity of successfully querying new photographs.

### Goal ###

The goal of this project is not to compete with [1]. After all, we are very interested by the top leading performance achieved by their research. What we actually want is to provide a system which benefits from a database system to relieve the runtime memory requirements of image geocoding. As a consequence, querying images will be much slower (due to disk reads). But how much slower? And how faster can we perform ILR with a database system? Our job will be to offer a comprehensive analysis on this impact and balance query speed with the available memory to compute the pose of query images.

To achieve this goal, we will divide the project into two stages: in the first we want to replicate the same accuracy achieved in [1]; in the second we will tune the overall pipeline to achieve satisfactory query speed.

### Dependencies ###
* Python 2.7
* PostgreSQL 9.5
* Psycopg2
* Numpy
* PyFLANN (Modified and available in https://github.com/ntma/flann)
* PGSEDistance (available in https://github.com/ntma/PGSEDistance)

### Related Algorithms ###
* Vocabulary Prioritized Search (VPS) [1]
* Active Search (AS) [1]
* SPRT-RANSAC (converted from ACG-Localizer c++ source code [4] to python) [3]
* Direct Linear Transform (DLT) from 6 correspondences [3]

### Datasets ###

To benchmark our code, at the moment we are using:
* a 100K generic vocabulary [5] to support the fine/coarse vocabularies required by [1];
* Dubrovnik dataset [6] which contains 6044 database images, Structure-from-Motion model and 800 additional query images with ground truth measured in meters provided by [2]. 

More datasets will be tested soon.

### Benchmarks ###

The following statistics were obtained with the current implementation:
* Success pose: 790 from 800 queries (98.75%)
* Overall time: ~2 hours
* Mean time/query: ~10 seconds
* Mean position error: ~25 meters

Note that 75% of the successful queries have an error below 8 meters. This benchmark was performed in an 
i5-4200U 1.6GHz processor with a 550MB/s read/write speed. We are currently fixing the query speed performance.

### HUGE DISCLAMER ###

Before proceeding to the installation/execution instructions, we must warn that 
this project is in a prototyping stage. Future releases of this code may not 
provide backwards compatibility.

### Installing pre-requisites ###

Download the [modified FLANN](https://github.com/ntma/flann) library and install it. This modified version is able to compute 
the parents of quantized descriptors at level L when using K-Means clustering.

```
cd flann/
mkdir build
cd build/
cmake ..
make
make install
```

Download and install the [SEDistance](https://github.com/ntma/PGSEDistance) plugin to compute euclidean distances in PostgreSQL. 
The installation target will be the PostgreSQL library folder.

```
cd sedistance/
make
make install
```

And finally compile the python plugin for faster math operations.

```
cd src/c_package/
python setup.py build_ext --inplace
```

### Setting the Database ###

Setting the database requires a super user. Only super users can create functions from 
extensions written in C. If not present, a super user can be created with:

```
createuser -h host -p port -P -s -e username
```

Now create the database with the super user.

```
createdb -h host -p port -U username database_name
```

And finally, execute both the create tables/functions scripts.

```
psql -p port -h host -d database_name -f postgres_scripts/create_tables.sql
psql -p port -h host -d database_name -f postgres_scripts/sql_functions.sql
```

### Insert the Dataset ###

To pre-process the dataset we follow a similar strategy to ACG-Localizer (although our generated files are not compatible YET).
We first generate a binary file containing all the required data from a Bundler SFM point cloud.

```
parse_dataset.py -p PATH_DATASET -o OUTPUT_BINARY
```

Then we read the outputted binary file and pre-process the point cloud. This includes, generating 
the indexes to the input visual vocabulary, computing the k-NN visibility graph and meaning descriptors 
per visual word (float representation for now). The output directory will contain several CSV files to 
speed the dataset insertion into PostgreSQL.

```
prepare_dataset.py -p BINARY_PATH -w VOCABULARY_PATH -o CSV_OUTPUT_DIRECTORY
```

Finally, the CSV are loaded into PostgreSQL.

```
insert_dataset.py -p CSV_INPUT_DIRECTORY -k PG_KEY
```

### Running benchmarks ###
* Coming soon

### References ###

1. Sattler, T., Leibe, B., & Kobbelt, L. (2016). Efficient & Effective Prioritized Matching for Large-Scale Image-Based Localization. Ieee Transactions on Pattern Analysis and Machine Intelligence, X(1). http://doi.org/10.1109/TPAMI.2016.2611662
2. Li, Y., Snavely, N., Huttenlocher, D., & Fua, P. (2012). Worldwide Pose Estimation using 3D Point Clouds. Computer Vision - ECCV, 15–29. http://doi.org/10.1007/978-3-642-33718-5_2
3. Chum, O., & Matas, J. (2008). Optimal Randomized RANSAC. IEEE Trans. Pattern Anal. Mach. Intell., 30(8), 1472–1482. http://doi.org/10.1109/TPAMI.2007.70787
4. ACG-Localizer: https://www.graphics.rwth-aachen.de/software/image-localization
5. 100K generic vocabulary: https://www.graphics.rwth-aachen.de/software/image-localization
6. Dubrovnik dataset: http://www.cs.cornell.edu/projects/p2f/