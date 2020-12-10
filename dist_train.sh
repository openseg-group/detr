PYTHON="/root/miniconda3/envs/detr/bin/python"
$PYTHON -m pip install scipy

DATAPATH=$1"/coco"
BACKBONE=$2

NODE_NUM=8
MAX_EPOCH=150

OUTPUT="outputs/detr_${BACKBONE}"
LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}.txt"

$PYTHON -m torch.distributed.launch \
            --nproc_per_node=$NODE_NUM \
            --use_env main.py \
            --coco_path $DATAPATH \
            --output_dir $OUTPUT \
            --epochs $EPOCH \
            --backbone $BACKBONE \
            2>&1 | tee ${LOG}
