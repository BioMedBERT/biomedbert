#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import webbrowser
from subprocess import call, Popen, CalledProcessError
from invoke import run, exceptions
from tqdm import trange


def set_gcp_project(project_id: str, zone: str):
    """Set GCP project ID"""
    try:
        run('gcloud config set project {}'.format(project_id))
        run('gcloud config set compute/zone {}'.format(zone))
    except exceptions.UnexpectedExit:
        print('Bad command')


def start_vm(vm_instance: str):
    """Start VM"""
    try:
        run('gcloud compute instances start {}'.format(vm_instance))
    except exceptions.UnexpectedExit:
        print('Bad command')


def stop_vm(vm_instance: str):
    """Stop VM"""
    try:
        run('gcloud compute instances stop {}'.format(vm_instance))
    except exceptions.UnexpectedExit:
        print('Bad command')


def connect_vm(project_id: str, zone: str, vm_instance: str):
    """SSH into VM"""
    try:
        call(['gcloud', 'compute', 'ssh', '--zone', zone, vm_instance, '--project', project_id])
    except CalledProcessError:
        print('Bad command')


def launch_notebook(project_id: str, zone: str, vm_instance: str):
    """Lauch notebook on VM"""
    try:
        process = Popen(['nohup', 'bash', './bash/connect_jupyterlab.sh', project_id, zone, vm_instance],
                        stdout=open('/dev/null', 'w'),
                        stderr=open('logfile.log', 'a'),
                        preexec_fn=os.setpgrp
                        )

        if process.returncode is None:
            print('Connecting to Jupyterlab at http://localhost:8080/lab?')
            # sleep for 25 seconds then launch http://localhost:8080/lab?
            for i in trange(25):
                time.sleep(1)
            webbrowser.open('http://localhost:8080/lab?')
    except CalledProcessError:
        print('Bad command')


def create_compute_vm(compute_instance: str, zone: str):
    """Create Compute VM"""
    try:
        # run works to print output to console
        run('bash ./bash/create_compute_vm.sh {} {}'.format(compute_instance, zone))
    except exceptions.UnexpectedExit:
        print('Bad command')


def create_tpu_vm(tpu_name: str, zone: str, preempt: str):
    """Create TPU VM"""
    if preempt is False:
        preempt = "no"
    else:
        preempt = "yes"
    try:
        # run works to print output to console
        run('bash ./bash/create_tpu_vm.sh {} {}'.format(tpu_name, zone, preempt))
    except exceptions.UnexpectedExit:
        print('Bad command')


def delete_tpu_vm(tpu_name: str, project_id: str, zone: str):
    """Create TPU VM"""
    try:
        # run works to print output to console
        run('gcloud compute tpus delete {} --project={} --zone={}'.format(tpu_name, project_id, zone))
    except exceptions.UnexpectedExit:
        print('Bad command')
