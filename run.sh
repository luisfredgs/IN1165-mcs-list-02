#!/bin/sh

python main.py --no-pruning --dataset=kc2
python main.py --no-pruning --dataset=pc1

python main.py --pruning --dataset=kc2 --hardnesses=all_instances
python main.py --pruning --dataset=kc2 --hardnesses=easy
python main.py --pruning --dataset=kc2 --hardnesses=hard

python main.py --pruning --dataset=pc1 --hardnesses=all_instances
python main.py --pruning --dataset=pc1 --hardnesses=easy
python main.py --pruning --dataset=pc1 --hardnesses=hard