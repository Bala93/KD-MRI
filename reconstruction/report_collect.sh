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