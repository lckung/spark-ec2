#!/bin/sh
#SPARK_MASTER=spark://${MASTERS}:7077
SPARK_MASTER=yarn-client
SPARK_HOME=/root/spark
TRAIN_SET=s3a://spark.data/input/20151013/withpubs_pricer.gz
JAR_DIR=/vol0/lkung-work/pricer_depjars
DEP_JARS=$JAR_DIR/wstx-asl-3.1.2.jar,$JAR_DIR/stax2.jar,$JAR_DIR/wurfl-1.5.1.jar,$JAR_DIR/slf4j-api-1.7.7.jar,$JAR_DIR/servlet-api-2.5.jar,$JAR_DIR/commons-logging-1.1.3.jar,$JAR_DIR/commons-lang-2.6.jar,$JAR_DIR/commons-math3-3.5.jar,$JAR_DIR/trove4j-3.0.3.jar
HADOOP_OPTS="-Dmapreduce.map.java.opts=-Xmx5120m -Dmapreduce.map.memory.mb=6100"
HADOOP_OPTS2="-Dmapreduce.map.java.opts=-Xmx15000m -Dmapreduce.map.memory.mb=16384"
MODEL_FILE=spark_click_model.tsv
DONE_FILE=TRAINING_DONE
OUTPUT_PATH="s3a://spark.data/daily/$DATE/"
WORK_DIR=/vol0/lkung-work
HADOOP=/root/ephemeral-hdfs/bin/hadoop

run_cmd () {
  $@
  if [ $? -ne 0 ]; then exit 1; fi
}

run_spark_job () {
export HADOOP_CONF_DIR=/root/ephemeral-hdfs/conf
$SPARK_HOME/bin/spark-submit \
--master ${SPARK_MASTER} \
--conf "spark.shuffle.io.maxRetries=6" \
--conf "spark.driver.maxResultSize=7000M" \
--num-executors 50 \
--executor-memory 16000m \
--executor-cores 2 \
--driver-memory 16000m \
--driver-cores 2 \
"$@"
if [ $? -ne 0 ]; then 
  echo "spark cmd failed!"
  exit 1
fi
#--conf "spark.driver.extraLibraryPath=/root/persistent-hdfs/lib/native" \
#--conf "spark.executor.extraLibraryPath=/root/persistent-hdfs/lib/native" \
}

run_spark_trainer () {
  if [ $# -ne 4 ]; then
    echo "Wrong number of arguments to run_spark_trainer()!"
    exit 1
  fi
  echo "Current dir: `pwd`"
  FEATURE_MAP=$1
  INPUT=$2
  MODEL_OUT=$3
  TRAINLOG_OUT=$4
  echo "Featuremap: $FEATURE_MAP"
  echo "Input: $INPUT"
  echo "Model output: $MODEL_OUT"
  echo "Training log: $TRAINLOG_OUT"

  ALGORITHM=LBFGS
  REG_TYPE=MYL2
  ITER=35
  STEP_SIZE=1
  BATCH_FRAC=1.0
  REG_PARAM=5e-6
  TEST_FRAC=0.001

export HADOOP_CONF_DIR=/root/ephemeral-hdfs/conf
unset SPARK_WORKER_INSTANCES
  $HADOOP fs -rm -r ${MODEL_OUT}
  $HADOOP fs -rm -r ${MODEL_OUT}-raw
  ${SPARK_HOME}/bin/spark-submit \
--master ${SPARK_MASTER} \
--conf spark.local.dir=/vol0/scratch \
--conf spark.driver.extraJavaOptions=-Djava.io.tmpdir=/vol0/tmp \
--conf spark.executor.extraJavaOptions=-Djava.io.tmpdir=/vol0/tmp \
--conf spark.driver.maxResultSize=40000m \
--conf spark.rdd.compress=true \
--conf spark.network.timeout=240000 \
--conf spark.akka.frameSize=1024 \
--conf spark.shuffle.service.enabled=true \
--conf spark.dynamicAllocation.minExecutors=24 \
--conf spark.kryoserializer.buffer.max=2047m \
--num-executors 80 \
--executor-memory 30g \
--executor-cores 4 \
--driver-memory 30g \
--driver-cores 4 \
--class BinaryClassification \
fractional-trainer-1.5.jar \
--algorithm ${ALGORITHM} --regType ${REG_TYPE} --regParam ${REG_PARAM} \
--maxIter ${ITER} --stepSize ${STEP_SIZE} --miniBatchFrac ${BATCH_FRAC} --fracTest ${TEST_FRAC} \
--kryoSerializer --featureMap ${FEATURE_MAP} ${INPUT} ${MODEL_OUT} 2>&1 |tee ${TRAINLOG_OUT}
  if [ $? -ne 0 ]; then 
    echo "spark trainer failed!"
    exit 1
  fi
#--conf spark.dynamicAllocation.minExecutors=24 \
#--conf spark.dynamicAllocation.enabled=true \
}

start_newrun () {
  echo "Starting a new run. Deleting existing intermediate files on HDFS.."
  $HADOOP fs -rm -r selection/click_counts.tsv
  $HADOOP fs -rm -r selection/feature_pvalues.tsv
  $HADOOP fs -rm -r selection/featuremap.tsv
  $HADOOP fs -rm -r selection/libsvm.txt.bz2
}

if [ $# -ne 1 ]; then
  echo "Usage: run_aws_trainer.sh <date>"
  exit 1
fi

pushd $WORK_DIR >/dev/null

DATE=$1
TRAIN_SET="s3a://spark.data/daily/$DATE/withpubs_pricer.gz"
OUTPUT_PATH="s3a://spark.data/daily/$DATE"
echo "Setting DATE to $DATE.."
echo "Input path is $TRAIN_SET"
echo "Output path prefix is $OUTPUT_PATH"

if ! $HADOOP fs -test -d selection/click_counts.tsv ; then
  echo "Step 1: Running click counting map reduce.."
  $HADOOP fs -rm -r selection/click_counts.tsv
  run_cmd $HADOOP jar pricer-feature-selection.jar training.FeatureSelectionMapReduce ${HADOOP_OPTS} -libjars ${DEP_JARS} count_clicks ${TRAIN_SET} selection/click_counts.tsv
  run_cmd $HADOOP fs -getmerge selection/click_counts.tsv click_counts.tsv
fi

if ! $HADOOP fs -test -d selection/feature_pvalues.tsv ; then 
  rm -f click_counts.tsv
  run_cmd $HADOOP fs -getmerge selection/click_counts.tsv click_counts.tsv
  NUM_POS_CLICKS=`cat click_counts.tsv | cut -f2`
  NUM_NEG_CLICKS=`cat click_counts.tsv | cut -f3`
  echo "Step 2: Running feature selection map reduce with num_pos_click=${NUM_POS_CLICKS}, num_neg_clicks=${NUM_NEG_CLICKS}.."
  $HADOOP fs -rm -r selection/feature_pvalues.tsv
  run_cmd $HADOOP jar pricer-feature-selection.jar training.FeatureSelectionMapReduce ${HADOOP_OPTS} -libjars ${DEP_JARS} select_feature ${TRAIN_SET} selection/feature_pvalues.tsv ${NUM_POS_CLICKS} ${NUM_NEG_CLICKS} -chisquare -pvalue 1.01 
fi

if ! $HADOOP fs -test -f selection/featuremap.tsv ; then
  echo "Step 3: Running spark script to add indices to featuremap.."
  run_spark_job --class AssignIdToFeatureMap fractional-trainer-1.5.jar --numFeatures 50000000 selection/feature_pvalues.tsv selection/featuremap-out.tsv
  run_cmd $HADOOP fs -getmerge selection/featuremap-out.tsv featuremap.tsv
  run_cmd $HADOOP fs -put featuremap.tsv selection/featuremap.tsv
  $HADOOP fs -rm -r selection/featuremap-out.tsv
fi

if ! $HADOOP fs -test -d selection/libsvm.txt.bz2 ; then
  echo "Step 4: Running conversion map reduce.."
  run_cmd $HADOOP jar pricer-feature-selection.jar training.FeatureSelectionMapReduce ${HADOOP_OPTS2} -libjars ${DEP_JARS} convert ${TRAIN_SET} selection/libsvm.txt.bz2 selection/featuremap.tsv
fi

if ! $HADOOP fs -test -f ${OUTPUT_PATH}/${MODEL_FILE}.gz ; then
  echo "Step 5: Running spark trainer.."
  $HADOOP fs -rm -f ${OUTPUT_PATH}/${DONE_FILE}
  run_spark_trainer "selection/featuremap.tsv" "selection/libsvm.txt.bz2" "selection/one_click_model.tsv" "log-training"
  if ! $HADOOP fs -test -d selection/one_click_model.tsv ; then
    echo "Model output file 'selection/one_click_model.tsv' not found!"
    exit 1
  fi
  run_cmd $HADOOP fs -getmerge selection/one_click_model.tsv ${MODEL_FILE}
  run_cmd gzip -f ${MODEL_FILE}
  run_cmd $HADOOP fs -put ${MODEL_FILE}.gz ${OUTPUT_PATH}/
  run_cmd $HADOOP fs -touchz ${OUTPUT_PATH}/${DONE_FILE}
else
  echo "Found ${OUTPUT_PATH}/${MODEL_FILE}.gz . No more job to run."
fi

popd >/dev/null
exit 0
