#!/usr/bin/env python
from __future__ import division, print_function, with_statement

from datetime import datetime
import subprocess
import sys
import spark_ec2
from optparse import OptionParser
import boto

TRAIN_INPUT = "trainout/withpubs_pricer.gz"
HADOOP = "/usr/bin/hadoop"
S3N_PREFIX = "s3n://spark.data/daily"
S3A_PREFIX = "s3a://spark.data/daily"
AWS_CLUSTER_NAME = "spark-cluster-32n-spot"
DONE_FILE = "TRAINING_DONE"
MODEL_FILE = "spark_click_model.tsv.gz"

def run_cmd(cmd):
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()
    return proc.returncode

def run_cmd_ignore(cmd):
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True)
    proc.wait()


def launch_aws_cluster():
    launch_cmd = ("-k spark -i spark.pem --region=us-east-1 --zone=us-east-1d "
                  "--instance-type=r3.4xlarge -s 18 --spot-price=0.60  --spark-version=v1.5.0 "
                  "--copy-aws-credentials --vpc-id=vpc-a39d60c7 --subnet-id=subnet-f5350dac "
                  "--hadoop-major-version=2 --ebs-vol-size=250 --ebs-vol-type=gp2 "
                  "--spark-ec2-git-repo=https://github.com/lckung/spark-ec2 "
                  "--use-existing-master --ami=ami-dd97d8b8 launch " + AWS_CLUSTER_NAME).split()
    opt_parser = spark_ec2.get_parser()
    (opts, args) = opt_parser.parse_args(args=launch_cmd)
    
    try:
        conn = boto.ec2.connect_to_region(opts.region)
    except Exception as e:
        print(repr(e), file=stderr)
        sys.exit(1)

    (master_nodes, slave_nodes) = spark_ec2.launch_cluster(conn, opts, cluster_name)
    wait_for_cluster_state(
        conn=conn,
        opts=opts,
        cluster_instances=(master_nodes + slave_nodes),
        cluster_state='ssh-ready'
    )
    setup_cluster(conn, master_nodes, slave_nodes, opts, True)
    return master_nodes


def ssh_args(opts):
    parts = ['-o', 'StrictHostKeyChecking=no']
    parts += ['-o', 'UserKnownHostsFile=/dev/null']
    if opts.identity_file is not None:
        parts += ['-i', opts.identity_file]
    return parts


def ssh_command(opts):
    return ['ssh'] + ssh_args(opts)

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

def launch_copy_input(opts):
    try:
        train_date_str = subprocess.check_output([HADOOP, "fs", "-stat", TRAIN_INPUT]).rstrip()
        train_date = datetime.strptime(train_date_str, "%Y-%m-%d %H:%M:%S")
    except subprocess.CalledProcessError:
        print("{input} doesn't exist on HDFS! Unable to continue.".format(input=opts.input))
        sys.exit(1)
    # run distcp to copy training data to S3
    s3_folder = S3N_PREFIX + "/" + train_date.strftime("%Y%m%d")
    s3_input_path = s3_folder + "/withpubs_pricer.gz"
    if subprocess.call([opts.hadoop_cmd, "distcp", opts.input, s3_input_path]) != 0:
        print("distcp failed!")
        sys.exit(1)

def launch_training_job(master_nodes, s3_folder):
    """Launch a training job on spark cluster."""
    master = master_nodes[0].public_dns_name
    print("Setting up HDFS on the cluster..")
    ssh(host=master, opts=opts, command="chmod u+x setup_pricer_data.sh")
    ssh(host=master,
        opts=opts,
        command="setup_pricer_data.sh")
    print("Running trainer..")
    ssh(host=master,opts=opts,
        command="run_aws_trainer.sh")
    
def get_opt_parser():
    parser = OptionParser(
        prog="launch_ec2_trainer",
        usage="%prog [options]\n")
    parser.add_option(
        "--input", default=TRAIN_INPUT,
        help="Training input file on HDFS")
    parser.add_option(
        "--hadoop-cmd", default="/usr/bin/hadoop",
        help="Path to hadoop command")
    parser.add_option(
        "--skip-copy-input", action="store_true", default=False,
        help="Skip copy input file to S3")
    return parser

def main():
    parser = get_opt_parser()
    (opts, args) = parser.parse_args()
    # check if input is there
    if not opts.skip_copy_input:
        launch_copy_input(opts)
    
    # launch an AWS cluster
    master_nodes = launch_aws_cluster()
    
    # launch the training job
    launch_training_job(master_nodes, s3_folder)
    
    # wait till training is done
    print("Waiting for training job..")
    while (True):
        if run_cmd("{hadoop} fs -stat {done_file}".format(hadoop=HADOOP, done_file=s3_folder + "/" + DONE_FILE)) == 0:
            print("Done file detected! Training job is done")
            break
        sys.sleep(60)
        print(".")
    
    # run distcp to copy the model back to S3 and bidderpath
    if run_cmd("{hadoop} distcp {s3_train_out} {model}".format(hadoop = HADOOP, s3_train_out = s3_folder + "/" + MODEL_FILE)):
        print("distcp model back failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
