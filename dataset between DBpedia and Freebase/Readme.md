This folder contains the dataset for relation alignment between DBpeida and Freebase.



- Relation-patterns.txt records the relation patterns passing the verification. Patterns are seperated by a blank line. The first line of each pattern is the triple pattern from Freebase. The second line of each pattern is the aligned entities for the head entity and tail entity of the triple pattern. The other lines are the aligned structural-relation pattern from DBpedia. Their type information is recorded in type.txt.
- Type.txt records the entities's type which occur in relation-patterns.txt.
- Relation-map.txt records the full name of relations occurred in relation-patterns.txt
- instance.txt is constructed by covering the subgraphs with the patterns we find.
