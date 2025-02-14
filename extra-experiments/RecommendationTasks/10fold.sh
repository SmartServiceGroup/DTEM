cd ./PRReviewer
python train_nn_10fold.py ./10fold_result.txt ../../GNN/HetSAGE/node_embedding/HetSAGE_node_embedding.bin model 512
cd ../RepoMaintainer
python train_nn_10fold.py ./10fold_result.txt ../../GNN/HetSAGE/node_embedding/HetSAGE_node_embedding.bin model 512
