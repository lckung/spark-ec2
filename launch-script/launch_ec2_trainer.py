#!/usr/bin/env python
from __future__ import division, print_function, with_statement
from datetime import datetime
from optparse import OptionParser
from sys import stderr

import spark_ec2
import subprocess
import sys
import boto
import time

def get_opt_parser():
    parser = OptionParser(
        prog="launch_ec2_trainer",
        usage="%prog [options]\n")
    parser.add_option(
        "--input", default="trainout/withpubs_pricer.gz",
        help="Training input file on HDFS")
    parser.add_option(
        "--hadoop-cmd", default="/usr/bin/hadoop",
        help="Path to hadoop command")
    parser.add_option(
        "--skip-copy-input", action="store_true", default=False,
        help="Skip copy input file to S3")
    parser.add_option(
        "--cluster-name", default="spark-cluster-32n-spot",
        help="Name of the ec2 cluster to launch")
    parser.add_option(
        "--s3-folder", default="spark.data/daily",
        help="S3 path to store output model file and done file")
    parser.add_option(
        "--done-file", default="TRAINING_DONE",
        help="File on S3 path to signal that training is done")
    parser.add_option(
        "--model-file", default="spark_click_model.tsv.gz",
        help="Output model file")
    parser.add_option(
        "--stop-cluster", action="store_true", default=False,
        help="Whether to stop the aws cluster after training")
    parser.add_option(
        "--train-date", default="",
        help="Override the date of train set")
    return parser


def run_cmd(cmd):
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    return proc.returncode


def launch_aws_cluster(conn, our_opts, ec2_opts):
    (master_nodes, slave_nodes) = spark_ec2.launch_cluster(conn, ec2_opts, our_opts.cluster_name)
    spark_ec2.wait_for_cluster_state(
        conn=conn,
        opts=ec2_opts,
        cluster_instances=(master_nodes + slave_nodes),
        cluster_state='ssh-ready'
    )
    spark_ec2.setup_cluster(conn, master_nodes, slave_nodes, ec2_opts, True)
    return master_nodes


def ssh_args(opts):
    parts = ['-o', 'StrictHostKeyChecking=no']
    parts += ['-o', 'UserKnownHostsFile=/dev/null']
    if opts.identity_file is not None:
        parts += ['-i', opts.identity_file]
    return parts


def ssh_command(opts):
    return ['ssh'] + ssh_args(opts)


def stringify_command(parts):
    if isinstance(parts, str):
        return parts
    else:
        return ' '.join(map(pipes.quote, parts))


# Run a command on a host through ssh, retrying up to five times
# and then throwing an exception if ssh continues to fail.
def ssh(host, opts, command):
    tries = 0
    while True:
        try:
            return subprocess.check_call(
                ssh_command(opts) + ['-t', '-t', '%s@%s' % (opts.user, host),
                                     stringify_command(command)])
        except subprocess.CalledProcessError as e:
            if tries > 5:
                # If this was an ssh failure, provide the user with hints.
                if e.returncode == 255:
                    raise UsageError(
                        "Failed to SSH to remote host {0}.\n"
                        "Please check that you have provided the correct --identity-file and "
                        "--key-pair parameters and try again.".format(host))
                else:
                    raise e
            print("Error executing remote command, retrying after 30 seconds: {0}".format(e),
                  file=stderr)
            time.sleep(30)
            tries = tries + 1

def get_trainset_date(opts):
    try:
        print("Checking training input at {input} on HDFS..".format(input=opts.input))
        train_date_str = subprocess.check_output([opts.hadoop_cmd, "fs", "-stat", opts.input]).rstrip()
        train_date = datetime.strptime(train_date_str, "%Y-%m-%d %H:%M:%S")
    except subprocess.CalledProcessError:
        print("{input} doesn't exist on HDFS! Unable to continue.".format(input=opts.input))
        sys.exit(1)
    return train_date

def launch_copy_input(opts):
    trainset_date = get_trainset_date(opts)
    # run distcp to copy training data to S3
    s3_folder_path = "s3n://" + opts.s3_folder
    s3_input_prefix = s3_folder_path + "/" + trainset_date.strftime("%Y%m%d")
    if subprocess.call([opts.hadoop_cmd, "distcp", opts.input, s3_input_prefix + "/withpubs_pricer.gz"]) != 0:
        print("distcp failed!")
        sys.exit(1)
    print("Distcp finished!\n")

def launch_training_job(master_nodes, trainset_date, opts, ec2_opts):
    # TODO: check whether HDFS is running
    # TODO: check whether YARN is running
    """Launch a training job on spark cluster."""
    master = master_nodes[0].public_dns_name
    print("Setting up HDFS on the cluster..")
    ssh(host=master, opts=ec2_opts, command="chmod u+x /root/spark-ec2/setup_pricer_data.sh")
    ssh(host=master, opts=ec2_opts, command="/root/spark-ec2/setup_pricer_data.sh")
    print("Running trainer with train date={d}..".format(d=trainset_date))
    ssh(host=master, opts=ec2_opts, command="chmod u+x /root/spark-ec2/run_aws_trainer.sh")
    ssh(host=master, opts=ec2_opts, command="nohup /root/spark-ec2/run_aws_trainer.sh {d} 2>&1 </dev/null |tee log.aws_trainer".format(d=trainset_date))
    print("Trainer was launched successfully..")
    
def stop_aws_cluster(conn, opts, ec2_opts):
    (master_nodes, slave_nodes) = spark_ec2.get_existing_cluster(conn, ec2_opts, opts.cluster_name, die_on_error=False)
    print("Stopping master...")
    for inst in master_nodes:
        if inst.state not in ["shutting-down", "terminated"]:
            inst.stop()
    print("Stopping slaves...")
    for inst in slave_nodes:
        if inst.state not in ["shutting-down", "terminated"]:
            if inst.spot_instance_request_id:
                inst.terminate()
            else:
                inst.stop()
    print("All instances stopped...")

def get_boto_conn(ec2_opts):
    try:
        conn = boto.ec2.connect_to_region(ec2_opts.region)
    except Exception as e:
        print(repr(e), file=stderr)
        sys.exit(1)
    return conn

def get_ec2_opts():
    launch_cmd = ("-k spark -i spark.pem --region=us-east-1 --zone=us-east-1d "
                  "--instance-type=r3.4xlarge -s 18 --spot-price=0.60  --spark-version=v1.5.0 "
                  "--copy-aws-credentials --vpc-id=vpc-a39d60c7 --subnet-id=subnet-f5350dac "
                  "--hadoop-major-version=yarn --ebs-vol-size=250 --ebs-vol-type=gp2 "
                  "--spark-ec2-git-repo=https://github.com/lckung/spark-ec2 "
                  "--use-existing-master --ami=ami-49cc9b2c --resume").split()
    opt_parser = spark_ec2.get_parser()
    (ec2_opts, args) = opt_parser.parse_args(args=launch_cmd)
    return ec2_opts

def main():
    parser = get_opt_parser()
    (opts, args) = parser.parse_args()
    # check if input is there
    if not opts.skip_copy_input:
        launch_copy_input(opts)

    ec2_opts = get_ec2_opts()
    conn = get_boto_conn(ec2_opts) 
    # launch an AWS cluster
    existing_masters, existing_slaves = spark_ec2.get_existing_cluster(conn, ec2_opts, opts.cluster_name, die_on_error=False)
    if existing_slaves:
        print("Cluster {cluster} is already running".format(cluster=opts.cluster_name))
        master_nodes = existing_masters
    else:
        master_nodes = launch_aws_cluster(conn, opts, ec2_opts)
    
    # launch the training job
    if opts.train_date == "":
        trainset_date = get_trainset_date(opts).strftime("%Y%m%d")
    else:
        trainset_date = opts.train_date
    print("Train date is set to {d}".format(d=trainset_date))
    launch_training_job(master_nodes, trainset_date, opts, ec2_opts)
    
    # wait till training is done
    print("Waiting for training job..")
    while (True):
        if run_cmd("{hadoop} fs -stat s3n://{prefix}/{date}/{done_file}".format(
            hadoop=opts.hadoop_cmd, prefix=opts.s3_folder, date=trainset_date, done_file=opts.done_file)) == 0:
            print("Done file detected! Training job is done")
            break
        time.sleep(60)
        print(".")
    
    # run distcp to copy the model back to S3 and bidderpath
    if run_cmd("{hadoop} distcp s3n://{prefix}/{date}/{model_file} {model}".format(
        hadoop=opts.hadoop_cmd, prefix=opts.s3_folder, date=trainset_date, model_file=opts.model_file, model=opts.model_file)):
        print("distcp model back failed!")
        sys.exit(1)
    print("Model file distcp is done!")
    if opts.stop_cluster:
        stop_aws_cluster(conn, opts, ec2_opts)

if __name__ == '__main__':
    main()
