# PYTHON="/root/miniconda3/envs/detr/bin/python"
PYTHON="/data/anaconda/envs/pytorch1.6.0/bin/python"

# $PYTHON -m pip install scipy
# $PYTHON -m pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

$PYTHON -m pip install yacs

DATAPATH="/home/yuhui/teamdrive/dataset/coco"

if [ "$1"x == "r50"x ]; then
    BACKBONE="resnet50"
    NODE_NUM=4
    OUTPUT="outputs/detr_${BACKBONE}"

    $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NODE_NUM \
    --use_env main.py \
    --coco_path $DATAPATH \
    --output_dir $OUTPUT
    # --coco_panoptic_path $DATAPATH \
    # --masks \

elif [ "$1"x == "r101"x ]; then
    BACKBONE="resnet101"
    NODE_NUM=4
    LOG="logs/detr_${BACKBONE}_bs_2x${NODE_NUM}.txt"

    $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NODE_NUM \
    --use_env main.py \
    --coco_path $DATAPATH \
    --masks \
    2>&1 | tee ${LOG}

elif [ "$1"x == "h18"x ]; then
    BACKBONE="hrnet18"
    NODE_NUM=1
    OUTPUT="outputs/detr_${BACKBONE}_singlegpu"

    export incre_memory_resolution=1
    export freeze_stem_stage1=0
    export sparse_transformer=1

    $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NODE_NUM \
    --use_env main_hrnet.py \
    --coco_path $DATAPATH \
    --output_dir $OUTPUT \
    --backbone $BACKBONE \
    --resume auto \
    --batch_size 2


elif [ "$1"x == "h32"x ]; then
    BACKBONE="hrnet32"
    NODE_NUM=1
    OUTPUT="outputs/detr_${BACKBONE}"

    $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NODE_NUM \
    --use_env main_hrnet.py \
    --coco_path $DATAPATH \
    --output_dir $OUTPUT \
    --backbone $BACKBONE

elif [ "$1"x == "h48"x ]; then
    BACKBONE="hrnet48"
    NODE_NUM=1
    OUTPUT="outputs/detr_${BACKBONE}"

    $PYTHON -m torch.distributed.launch \
    --nproc_per_node=$NODE_NUM \
    --use_env main_hrnet.py \
    --coco_path $DATAPATH \
    --output_dir $OUTPUT \
    --backbone $BACKBONE

else
  echo "$1"x" is unrecognized settings!"
fi

# python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --cfg experiments/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100_test_dist.yaml