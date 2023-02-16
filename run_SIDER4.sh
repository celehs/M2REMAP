#!/bin/bash

# abort entire script on error
set -e


python main.py --train_directory "data/" \
                    --train_filename "sider4_train_OneDrug_Sides_CUIs_Sider_threshld3_1757.csv" \
                    --test_directory "data/" \
                    --test_filename  "sider4_test_OneDrug_Sides_CUIs_Sider_threshld3_1757.csv"                     \
                    --save_directory  "result/" \
                    --results_filename  "SIDER4.csv"   \
                    --train_filename_assist  "label_indication.pkl"   \
                    --epochs 80 \
                    --learning_rate  0.001 \
                    --wight_embedding  0.001 \
                    --weight_emb_gan  0.001 \
                    --weight_pair_gan  0.001 \
                    --colums_max  164