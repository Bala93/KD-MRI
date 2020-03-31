BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='kd'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/MRI-KD/us_masks/'${DATASET_TYPE}'/'${ACC_TYPE}
TEACHER_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_teacher/best_model.pt'
STUDENT_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_feature/best_model.pt'
echo python train_sr_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --imitation_required
python train_sr_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT} --imitation_required 