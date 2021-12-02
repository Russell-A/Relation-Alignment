This folder contains the front and the backend of the human verification system.



To run the program, there need some extra packages. Due to the limitation of filesize on GitHub, we put them on a web disk.

1. language model, please put the bin.gz file in the folder ./language_model

   link: https://pan.baidu.com/s/1oU2wK6htkE49J5-hwGTMyQ password: 2ri2

2. degree_dict, which contains every entity's degree in DBpedia. Please put it in the folder ./degree

   link: https://pan.baidu.com/s/1ibW5ljtQfcf_Mpbl4Bzr4w password: qpst

3. candidate5m, which contains every subgraph for a triple pattern in freebase. Please put the txt files in the folder ./candidate5m

   link: https://pan.baidu.com/s/1u4lC8J0DqrL_FUSw1yduuw password: hi12

4. graphvisual, which is the front of our system. You can find the index html page in ./graphvisual/dist

   link: https://pan.baidu.com/s/1Sjz-tGLtbLkbZvCgS3g0zA password: 49a0



Moreover, the backend requires some python packages.

- flask

- re

- numpy

- codecs

- joblib

- itertools

- matplotlib

- networkx

- shutil

- os

- gensim

  

After all these files are correctly placed and python packages are installed. You can run the backend of the system. 

The command is 'python ./backend/Backend.py'

Then, you can open the ./graphvisual/dist/index.html to perform your alignment. The aligned result will be put in ./result and ./log will log the statistical data of the system.





