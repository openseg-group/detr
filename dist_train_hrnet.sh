PYTHON="/opt/conda/bin/python"
$PYTHON -m pip install scipy
$PYTHON -m pip install yacs

DATAPATH=$1"/coco"
BACKBONE=$2

NODE_NUM=8
MAX_EPOCH=50
LR_DROP=40


if [ "$3"x == "1"x ]; then
    export incre_memory_resolution=0
    export freeze_stem_stage1=0

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride32x"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride32x.txt"

    $PYTHON -m torch.distributed.launch \
                --nproc_per_node=$NODE_NUM \
                --use_env main_hrnet.py \
                --coco_path $DATAPATH \
                --output_dir $OUTPUT \
                --epochs $MAX_EPOCH \
                --lr_drop $LR_DROP \
                --backbone $BACKBONE \
                --resume auto \
                2>&1 | tee ${LOG}

elif [ "$3"x == "2"x ]; then
    export incre_memory_resolution=0
    export freeze_stem_stage1=1

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride32x_freeze_stem"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride32x_freeze_stem.txt"

    $PYTHON -m torch.distributed.launch \
                --nproc_per_node=$NODE_NUM \
                --use_env main_hrnet.py \
                --coco_path $DATAPATH \
                --output_dir $OUTPUT \
                --epochs $MAX_EPOCH \
                --lr_drop $LR_DROP \
                --backbone $BACKBONE \
                --resume auto \
                2>&1 | tee ${LOG}

elif [ "$3"x == "3"x ]; then
    export incre_memory_resolution=1
    export freeze_stem_stage1=0

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride4x"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride4x.txt"

    $PYTHON -m torch.distributed.launch \
                --nproc_per_node=$NODE_NUM \
                --use_env main_hrnet.py \
                --coco_path $DATAPATH \
                --output_dir $OUTPUT \
                --epochs $MAX_EPOCH \
                --lr_drop $LR_DROP \
                --backbone $BACKBONE \
                --resume auto \
                2>&1 | tee ${LOG}

elif [ "$3"x == "4"x ]; then
    export incre_memory_resolution=1
    export freeze_stem_stage1=1

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride4x_freeze_stem"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride4x_freeze_stem.txt"

    $PYTHON -m torch.distributed.launch \
                --nproc_per_node=$NODE_NUM \
                --use_env main_hrnet.py \
                --coco_path $DATAPATH \
                --output_dir $OUTPUT \
                --epochs $MAX_EPOCH \
                --lr_drop $LR_DROP \
                --backbone $BACKBONE \
                --resume auto \
                2>&1 | tee ${LOG}

else
  echo "$3"x" is unrecognized settings!"
fi