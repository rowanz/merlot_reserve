"""
This script, adapted very heavily from https://github.com/kingoflolz/mesh-transformer-jax
installs dependencies on a TPU you've made
"""
import functools
import os
import subprocess
import time

import glob
import requests
from fabric import Connection
from dataclasses import dataclass
import multiprocessing.pool
import regex as re
import pandas as pd
import random
multiprocessing.set_start_method("spawn")

from func_timeout import func_set_timeout
import time

@functools.lru_cache()
def get_bearer():
    return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
    return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
        "utf-8").strip()

@dataclass
class TPUCreator:
    """
    Utility for creating TPUs and stuff
    """
    name: str
    tpu_size: int
    zone: str = 'europe-west4-a'
    preemptible: bool = False
    network: str='main'
    subnetwork: str='europe-west4'

    @property
    def base_url(self):
        # https://cloud.google.com/tpu/docs/reference/rest/v2alpha1/projects.locations.nodes/create
        return f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{self.zone}/nodes'

    def check_tpu(self):
        response = requests.get(f'{self.base_url}/{self.name}',
                                headers={'Authorization': f'Bearer {get_bearer()}'})
        return response.json()

    def delete_tpu(self):
        response = requests.delete(f'{self.base_url}/{self.name}', headers={'Authorization': f'Bearer {get_bearer()}'})
        return response.json()

    def wait_until_tpu_ready(self):
        desired_state = {'state': 'READY', 'health': 'HEALTHY'}
        while True:
            ret = self.check_tpu()

            print(f"wait_until_tpu_ready check: {ret}", flush=True)

            if ("error" in ret) or (ret["state"] == "TERMINATED"):
                return False

            matches = True
            for k, expected_v in desired_state.items():
                if k not in ret:
                    matches = False
                    continue
                if ret[k] != expected_v:
                    matches = False

            if matches:
                return True
            time.sleep(30)

    def get_connections(self):
        """
        This was a pain to debug. basically sometimes things can work via `gcloud` and NOT on python
        For debugging see `gcloud alpha compute tpus tpu-vm ssh rz0-v3-32 --zone europe-west4-a --dry-run`

        Dry run often tries to connect via the external IP, but here we want to use the internal IP (because of Ray I think)

        Paramiko added more errors, but these are fixible if you just ssh into it using "ssh" and clear the cache.
        WHY do you need to do that? idk....
        :return:
        """
        # clear cache
        for fn in ['google_compute_known_hosts', 'known_hosts']:
            try:
                os.remove(os.path.expanduser(f'~/.ssh/{fn}'))
            except OSError:
                pass

        # I don't know why I need to do this but I do... the first time errors and then the second time works
        os.system('gcloud alpha compute tpus tpu-vm ssh {} --zone {} --command="echo hi"'.format(self.name, self.zone))

        info = self.check_tpu()
        outputs = []
        for x in info["networkEndpoints"]:
            key_path = os.path.expanduser('~/.ssh/google_compute_engine')
            # username = key_path.split('/')[2]
            # hostname = x['ipAddress']
            # user_known_hosts_file = os.path.expanduser('~/.ssh/google_compute_known_hosts')
            # os.system(f'ssh -t -i {key_path} -o CheckHostIP=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile={user_known_hosts_file} {username}@{hostname} "echo connected"')
            conn = Connection(x['ipAddress'], connect_kwargs={"key_filename": key_path})
            outputs.append(conn)
        if len(outputs) * 8 != self.tpu_size:
            raise ValueError(
                f"Something weird happened -- you wanted TPU {self.tpu_size} but only {len(outputs)} outputs")
        return outputs


def install_dependencies(conn):
    """
    Upload all the code
    :param conn:
    :param address:
    :return:
    """
    try:
        conn.sudo('killall python3')
    except Exception as e:
        print(e)
    try:
        conn.sudo('killall screen')
    except Exception as e:
        print(e)

    print(f"Starting on {conn}", flush=True)
    conn.sudo('rm -rf *.py')
    conn.sudo('rm -rf *.json')
    conn.run('rm -rf pretrain && rm -rf mreserve')

    local_code_path = os.path.expanduser('~/merlot_reserve/')

    # Copy python files
    for ok_folder in ['mreserve', 'pretrain', 'finetune']:
        conn.sudo(f'rm -rf {ok_folder}')
        conn.run(f"mkdir {ok_folder} -p")
        for i in glob.glob(os.path.join(local_code_path, ok_folder, '*.py')):
            conn.put(i, f'{ok_folder}/')

    # Copy finetune
    # conn.run(f"mkdir finetune/tvqa -p")
    # for i in glob.glob(os.path.join(local_code_path, 'finetune', 'tvqa', '*.py')):
    #     conn.put(i, 'finetune/tvqa/')

    for folder in ['configs', ]:
        conn.run(f"mkdir pretrain/{folder} -p")
        for i in glob.glob(os.path.join(local_code_path, 'pretrain', folder, '*.yaml')):
            conn.put(i, f'pretrain/{folder}/')

    # Copy BPE
    bpe_path = os.path.join(local_code_path, 'mreserve', 'lowercase_encoder.json')
    conn.put(bpe_path, f'mreserve/')

    conn.put(os.path.join(local_code_path, 'pretrain/tpu_startup_script.sh'), "/tmp/startup.sh")
    conn.sudo('chmod +x /tmp/startup.sh', hide=True)
    conn.run('/tmp/startup.sh', hide=True)

    # Comment this in if you want to add wandb
    # conn.run('python3 -m wandb login MYLOGIN')


if __name__ == '__main__':

    #######################################################
    ##### NOTE: You will need to change things here   #####
    ##### It runs the command on all the TPUs         #####
    #######################################################

    tpu_creator = TPUCreator(name='YOURTPUNAMEHERE', tpu_size=8, zone='YOURNAMEHERE')

    conns = tpu_creator.get_connections()

    with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
        p.map(install_dependencies, conns)
    time.sleep(30)

    def _run_things(conn):
        with conn.cd('pretrain'):
            conn.run('screen -d -m -L bash -c "python3 train.py configs/base.yaml"', pty=False)
            print('done')

    with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
        p.map(_run_things, conns)

    # # Or you can finetune
    # def _run_finetune(conn):
    #     with conn.cd('finetune/tvqa'):
    #         conn.run('screen -d -m -L bash -c "python3 tvqa_finetune.py ../../pretrain/configs/large.yaml ${MYCKPT} -lr=5e-6 -ne=3 -output_grid_h=18 -output_grid_w=32"', pty=False)
    #         print('done')
    #
    # with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
    #     p.map(_run_finetune, conns)
