#!/bin/sh

HADOOP=/root/ephemeral-hdfs/bin/hadoop
DIR=/vol0/lkung-work

if ! $HADOOP fs -test -d /user/$USER ; then
  echo "Creating /user/$USER on HDFS.."
  $HADOOP fs -mkdir -p /user/$USER
fi

pushd $DIR >/dev/null

if [ ! -d ./pricer_data ]; then
  s3cmd get s3://spark.data/pricer_data.tgz
  tar -zxf pricer_data.tgz
fi

cd $DIR/pricer_data
for f in ./*; do
  echo "Uploading $f to HDFS.."
  $HADOOP fs -put $f /user/$USER/
done
cd -

popd >/dev/null
