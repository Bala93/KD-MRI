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
