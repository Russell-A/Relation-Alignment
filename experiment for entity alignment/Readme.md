#Entity Alignment Expirement 
##Environment
Environment setting in the packages.txt is to install necessary packages.

##Id files of DBpedia and DWY
Entity to id and Relation to id are files to map the names of entity and relation to an integer id so that they can be used to
train deep neural network. The results are store in the following files.
* dbpedia_entity2id.txt: Entity ids in DBpedia.
* dbpedia_relation2id.txt: Relation ids in DBpedia.
* dwy_entity2id.txt: Entity ids in DWY.
* dwy_relation2id.txt: Relation ids in DWY.

##Triple files of DBpeida and DWY
Respect triples of DBpedia and DWY are used to compute TransE loss separately.
* dbpedia_train.txt: Triples of DBpedia.
* dwy_train.txt: Triples of DWY.

##Files of Entity Alignment
In this experiment, the main task is entity alignment between DBpedia and DWY. We have 10k aligned entity pairs totally, and we take 30% of it for training and the rest for testing.
* entity_align_test.txt: Dataset for testing with 7k pairs of entity.
* entity_align_train.txt: Dataset for Training with 3k pair of entity.

##Files of Relation Alignment
Files of relation alignment are store in the fold named result, each of which is verified by human volunteers.

##Training and Testing
We integrate training and testing process in model.py.