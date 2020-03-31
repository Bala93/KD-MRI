# Train
## Teacher

```bash
BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='teacher'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x' # using this to get the fs data in the acc_4x path 
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
echo python train_sr_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --model_type ${MODEL_TYPE}
python train_sr_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --model_type ${MODEL_TYPE} 
```

## Student

```bash
BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='student'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x' # using this to get the fs data in the acc_4x path 
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
echo python train_sr_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --model_type ${MODEL_TYPE}
python train_sr_base_model.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --model_type ${MODEL_TYPE} 
```

## Feature 

```bash
BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='feature'

BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x' # using this to get the fs data in the acc_4x path 
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}
TRAIN_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/train/acc_'${ACC_FACTOR}
VALIDATION_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
TEACHER_CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_teacher/best_model.pt'
echo python train_sr_feature.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH}  --teacher_checkpoint ${TEACHER_CHECKPOINT}
python train_sr_feature.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --teacher_checkpoint ${TEACHER_CHECKPOINT}
```

### KD

```bash
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
```

# Valid

```bash
BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='kd'

ACC_FACTOR='4x'
BATCH_SIZE=1
DEVICE='cuda:0'
DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/results'
echo python valid_sr.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --model_type ${MODEL_TYPE}
python valid_sr.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --model_type ${MODEL_TYPE}
```

# Evaluate

```bash
BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='teacher'

ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
```

# Report 

```bash
BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='teacher'

echo ${MODEL}
ACC_FACTOR='4x'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/report.txt'
echo ${ACC_FACTOR}
cat ${REPORT_PATH}
echo "\n"
```