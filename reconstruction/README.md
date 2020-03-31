# Train
## Teacher

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='teacher'
ACC_FACTOR='4x'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:1'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/KD-MRI/us_masks/'${DATASET_TYPE}

echo python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
```

## Student

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='student'
ACC_FACTOR='4x'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/KD-MRI/us_masks/'${DATASET_TYPE}

echo python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
python train_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --model_type ${MODEL_TYPE}
```

## Feature 

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='feature'
ACC_FACTOR='4x'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:1'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/KD-MRI/us_masks/'${DATASET_TYPE}
TEACHER_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_teacher/best_model.pt'

echo python train_feature.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT}
python train_feature.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT}
```

### KD

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='kd'
ACC_FACTOR='4x'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:1'

EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
USMASK_PATH=${BASE_PATH}'/KD-MRI/us_masks/'${DATASET_TYPE}
TEACHER_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_teacher/best_model.pt'
STUDENT_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_feature/best_model.pt'

echo python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT}
python train_kd.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT} --student_checkpoint ${STUDENT_CHECKPOINT}
```

# Valid

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='teacher' #student,kd

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:1'

DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/results'

echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE}
python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --model_type ${MODEL_TYPE}
```

# Evaluate

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='teacher'

ACC_FACTOR='4x'

TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/'

echo python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
```

# Report 

```bash
BASE_PATH='/media/htic/NewVolume1/murali/MR_reconstruction'
MODEL='attention_imitation'
DATASET_TYPE='mrbrain'
MODEL_TYPE='kd'
ACC_FACTOR='4x'

echo ${MODEL}
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'_'${MODEL_TYPE}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
```