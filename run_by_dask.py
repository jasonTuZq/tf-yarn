import tensorflow as tf
from dask_yarn import YarnCluster
from dask.distributed import Client
import os
import json
import sys
import time
import random
from collections import defaultdict

from mnist_keras_distributed import run as tf_run

def tf_job(cluster_spec, jon_name, task_index, mdl_args_str):
    os.environ['CLUSTER_SPEC'] = json.dumps(cluster_spec)
    os.environ["JOB_NAME"] = jon_name
    os.environ["TASK_INDEX"] = task_index
    tf_run(mdl_args_str)

def create_spec(yarn_cluster, nps=1, nworker=1):
    info = yarn_cluster._dask_client().scheduler_info()
    workers = sorted(list(info['workers'].keys()))
    n_tf_servers = nps + nworker
    assert len(workers) >= n_tf_servers, "only have %d, \
        but require %d"%(len(workers), n_tf_servers)
    dask_spec = {'ps': workers[0:nps], 
        'worker': workers[nps:n_tf_servers]}
    workers_ip = [v.split('://')[-1].rsplit(':')[0]
        for v in workers[:n_tf_servers]]
    ports = defaultdict(lambda: random.randint(11998,31233))
    tf_addrs = [None]*n_tf_servers
    for i in range(n_tf_servers):
        tf_ip = workers_ip[i]
        ports[tf_ip] += 1 
        tf_addrs[i] = "%s:%d" % (tf_ip, ports[tf_ip])
    tf_spec = {'ps': tf_addrs[0:nps], 
        'worker': tf_addrs[nps:n_tf_servers]}
    return tf_spec, dask_spec

if __name__ == "__main__":
    mdl_args_str = sys.argv[1]
    env_pack_path = sys.argv[2]
    cluster_mode = 'local'  # default cluster mode is local
    if len(sys.argv) >= 4 and sys.argv[3] == 'remote':
        cluster_mode = 'remote'
    
    # create dask cluster from yarn or local
    cluster = YarnCluster(environment=env_pack_path, n_workers=2, 
        worker_vcores=1, worker_memory='500MiB', deploy_mode=cluster_mode)
    time.sleep(3) # wait a while for cluster setup finish

    # create spec for tf servers
    nps, nworker = 1, 1
    tf_spec, dask_spec = create_spec(cluster, nps=nps, nworker=nworker)
    print(tf_spec, dask_spec)
    # create client for cluster to submit job
    client = Client(cluster)

    # submit job
    ps_jobs = [client.submit(tf_job, tf_spec, 'ps', i, mdl_args_str, 
        workers=worker, pure=False) for i,worker in enumerate(dask_spec[:nps])]
    worker_jobs = [client.submit(tf_job, tf_spec, 'worker', i, mdl_args_str, 
        workers=worker, pure=False) for i,worker in enumerate(dask_spec[nps:])]

    # wait for job to finish
    time.sleep(10000)        
