#----------------------- ensemble three models (resnet50, resnet101, resnext101)-------------------------------
python tools/train.py --config_file='configs/visda.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet50_ibn_a')" \
MODEL.PRETRAIN_PATH "('.cache/torch/checkpoints/resnet50_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.FP16 False \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([128, 256])' \
INPUT.SIZE_TEST '([128, 256])' \
DATASETS.TRAIN "('personx_spgan',)" \
DATASETS.TEST "('personx_spgan',)" \
DATASETS.ROOT_DIR "('./data/challenge_datasets')" \
OUTPUT_DIR "('./output/visda/base-ensemble/r50-320-circle')"


python tools/train.py --config_file='configs/visda.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnet101_ibn_a')" \
MODEL.PRETRAIN_PATH "('.cache/torch/checkpoints/resnet101_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.FP16 False \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([128, 256])' \
INPUT.SIZE_TEST '([128, 256])' \
DATASETS.TRAIN "('personx_spgan', )" \
DATASETS.TEST "('personx_spgan',)" \
DATASETS.ROOT_DIR "('./data/challenge_datasets')" \
OUTPUT_DIR "('./output/visda/base-ensemble/r101-320-circle')"


python tools/train.py --config_file='configs/visda.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.MODEL_TYPE "baseline" \
MODEL.NAME "('resnext101_ibn_a')" \
MODEL.PRETRAIN_PATH "('.cache/torch/checkpoints/resnext101_ibn_a.pth.tar')" \
SOLVER.LR_SCHEDULER 'cosine_step' \
DATALOADER.NUM_INSTANCE 16 \
MODEL.ID_LOSS_TYPE 'circle' \
SOLVER.WARMUP_ITERS 0 \
SOLVER.MAX_EPOCHS 12 \
SOLVER.COSINE_MARGIN 0.35 \
SOLVER.COSINE_SCALE 64 \
SOLVER.FREEZE_BASE_EPOCHS 2 \
SOLVER.FP16 False \
MODEL.TRIPLET_LOSS_WEIGHT 1.0 \
INPUT.SIZE_TRAIN '([128, 256])' \
INPUT.SIZE_TEST '([128, 256])' \
DATASETS.TRAIN "('personx_spgan', )" \
DATASETS.TEST "('personx_spgan',)" \
DATASETS.ROOT_DIR "('./data/challenge_datasets')" \
OUTPUT_DIR "('./output/visda/base-ensemble/next101-320-circle')"
