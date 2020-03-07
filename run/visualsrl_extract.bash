# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/visualsrl/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python3 src/tasks/visualsrl.py \
    --train train \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --batchSize 1 --optim bert --lr 5e-5 --epochs 1 \
    --tqdm --output $output ${@:3}
