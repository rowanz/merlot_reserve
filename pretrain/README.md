# Pretraining code

This folder contains code needed to pretrain MERLOT Reserve on a Cloud TPU VM. 

You can also use it to set up a Cloud TPU VM for other things (like finetuning)

## Setting up your Cloud TPU VM

_NOTE: I am documenting this to record the steps I took when doing experiments for this project. Cloud TPU VMs were in Alpha back then (circa Summer 2021), so this information might be outdated._

First, create a google cloud machine (which has access to the needed API to create your Cloud TPU VM)
* Create machine configuration with `Compute-optimized` and `c2-standard-4` in your desired region
* Boot disk `Debian GNU/Linux 10 Buster + TF 2-5-0`
* Add SSH key and static external IP address. Network=main. Then add to `~/.ssh/config` on the local machine
* Last under "Cloud API access scopes" you must Allow Full Access to All Cloud APIs

* You might need to do something weird about firewall allowing ssh [https://cloud.google.com/tpu/docs/users-guide-tpu-vm](see the users guide)
    * `gcloud compute firewall-rules create --network=default allow-ssh --allow=tcp:22` 
* Log in and do `gcloud config set compute/zone ${MYZONE}` to the zone you want
* Generate new ssh keys on the cloud machine -- just do it by sshing into a TPU
```
gcloud alpha compute tpus tpu-vm create TEST --zone europe-west4-a
gcloud alpha compute tpus tpu-vm ssh TEST --zone europe-west4-a --dry-run
```

Once you've created a TPU using that command, run the script in [pretrain/tpu_run.py](pretrain/tpu_run.py). Basically that SSH's into your TPU (all the workers too), installs dependencies, and then runs your command.
