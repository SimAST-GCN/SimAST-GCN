# Turn-Tree-into-Graph-Automatic-Code-Review-via-Simplified-AST-Driven-Graph-Convolutional-Network
Turn Tree into Graph: Automatic Code Review via Simplified AST Driven Graph Convolutional Network


Attention! We modified the pipline.py, it didn't need large disk file. The graph generation is moved to the data_iter.py.

for the gensim module, please make sure that the version is < 4.00.

for SimAST-GCN, you can run the following commond.
1. python pipline.py
2. python SimAST-GCN.py

for astnn, we recommend you to remove the files generated by the pipline.py. Then, do the following operation.
1. python ast_pipline.py
2. python astnn.py


for dace, we all recommend you to remove the files generated by other files. Then, do the following operation.
1. python token_pre.py
2. python dace.py
