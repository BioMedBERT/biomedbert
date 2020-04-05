#!/usr/bin/env python
# -*- coding: utf-8 -*-


from invoke import run


class GCPHelpers():
    """
    Class for GCP helper functions
    """
    def __init__(self):
        """Constructor"""
        self.project_id = None
        self.zone = None
        self.vm_instance = None

    def set_gcp_project(self, project_id: str, zone: str):
        """Set GCP project ID"""
        self.project_id = project_id
        self.zone = zone
        run('gcloud config set project {}'.format(self.project_id))
        run('gcloud config set compute/zone {}'.format(self.zone))

    def set_vm_instance(self, vm_instance: str):
        """Set VM Instance"""
        self.vm_instance = vm_instance

    def start_vm(self):
        """Start VM"""
        if self.vm_instance is None:
            print('Insance Name {} not set'.format(self.vm_instance))
        else:
            run('gcloud compute instances start {}'.format(self.vm_instance))

    def stop_vm(self):
        """Stop VM"""
        if self.vm_instance is None:
            print('Insance Name {} not set'.format(self.vm_instance))
        else:
            run('gcloud compute instances stop {}'.format(self.vm_instance))

    def launch_notebook(self):
        """Lauch notebook on VM"""
        if self.project_id is None:
            print('GCP ProjectID not set')
        elif self.zone is None:
            print('GCP zone not set')
        elif self.vm_instance is None:
            print('GCP VM Instance not set')
        else:
            run('gcloud compute ssh --project {} --zone {} {} -- -L 8080:localhost:8080'.format(
                self.project_id, self.zone, self.vm_instance))

