# Relatin-alignment

This github repository aims to provide the dataset for relation-alignment and the project code for the paper 'Building Benchmark Datasets for Relation Alignment Between Knowledge Graphs'


### Dataset for relation alignment

The dataset for relation alignment is put in the folder 'Dataset Between DBpedia and Freebase'.

The dataset contains four files.  'Relation-patterns.txt' has the aligned structural-relation patterns we have found between DBpeida and Freebase. Every aligned pattern is separated by a blank line. The first line of an aligned pattern is a pattern from Freebase. The second line shows the aligned head entity and tail entity in DBpedia. The left lines describe the aligned pattern mined from DBpedia. All entities' types in the 'relation-patterns.txt' can be found in 'type.txt'. 'instancec.txt' contains the instances we cover using the aligned patterns we have found.

