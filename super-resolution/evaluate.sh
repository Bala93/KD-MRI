BASE_PATH='/media/hticpose/drive2/Balamurali'
MODEL='attention_imitation'
DATASET_TYPE='calgary'
MODEL_TYPE='teacher'

ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/validation/acc_'${ACC_FACTOR}
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/sr/'${MODEL}'_'${MODEL_TYPE}'/'
python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 

