PYTHON="/opt/conda/bin/python"
$PYTHON -m pip install scipy
$PYTHON -m pip install yacs
$PYTHON -m pip install termcolor

DATAPATH=$1"/coco"
BACKBONE=$2

NODE_NUM=8
MAX_EPOCH=50
LR_DROP=40


if [ "$3"x == "1"x ]; then
    export encoder_high_resolution=0

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
    export encoder_high_resolution=0
    export sparse_transformer=1
    export fairseq_multi_head_attention=0

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride32x_sparse_fixbug"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride32x_sparse.txt"

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
    export encoder_high_resolution=0
    export sparse_transformer=1
    export fairseq_multi_head_attention=1

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride32x_fairseq_fixbug"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride32x_fairseq.txt"

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
    export encoder_high_resolution=1
    export sparse_transformer=0
    export cross_transformer=0

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride4x_upsample"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride4x_upsample.txt"

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

elif [ "$3"x == "5"x ]; then
    export encoder_high_resolution=1
    export sparse_transformer=0
    export cross_transformer=1

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_stride4x_crossfusion"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_stride4x_crossfusion.txt"

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

elif [ "$3"x == "6"x ]; then
    export encoder_high_resolution=1
    export linear_transformer=1
    export encoder_resolution=8

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_linear_stride8x"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_linear_stride8x.txt"

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

elif [ "$3"x == "7"x ]; then
    export encoder_high_resolution=1
    export linear_transformer=1
    export encoder_resolution=16

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_linear_stride16x"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_linear_stride16x.txt"

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

elif [ "$3"x == "8"x ]; then
    export encoder_high_resolution=1
    export linear_transformer=1
    export encoder_resolution=32

    OUTPUT="outputs/detr_${BACKBONE}_gpu${NODE_NUM}x_epoch${MAX_EPOCH}_linear_stride32x"
    LOG="logs/detr_${BACKBONE}_epoch${MAX_EPOCH}_linear_stride32x.txt"

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