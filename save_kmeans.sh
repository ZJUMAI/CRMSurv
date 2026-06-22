CUDA_VISIBLE_DEVICES=0 python save_kmeans.py --which_splits 5foldcv \
                                      --dataset tcga_blca \
                                      --data_root_dir your_root \
                                      --modal path \
                                      --model crmsurv \
                                      --cancer_type blca \
                                      --cluster 8 \
                                      --save_cluster_path ./Data/WSI_kmeans_8,
