import json
import os

from neko_sdk.environment.root import find_data_root
from neko_sdk.cfgtool.argsparse import neko_get_arg

class neko_platform_cfg:
    def set_up(this, data_root, save_root,log_root,devices,wandb_api_key=None):
        this.data_root = data_root;
        this.log_root=os.path.join(log_root,os.path.basename(os.getcwd()));
        this.devices=devices;
        this.save_root = os.path.join(save_root,os.path.basename(os.getcwd()));
        this.wandb_api_key=wandb_api_key
        os.makedirs(this.save_root, exist_ok=True);

    # if you have no wandb logger, don't call this function!
    def arm_wandb(this,project="moose",method_name=None):
        import wandb;
        if(method_name is None):
            method_name=os.path.basename(os.getcwd());
        this.project=project;
        this.entity=method_name;
        this.dir=os.path.join(this.log_root,this.project,this.entity);
        os.makedirs(this.dir,exist_ok=True);
        if(this.wandb_api_key is None):
            this.run=wandb.init(project=this.project, entity=this.entity, dir=this.dir,mode="offline");
        else:
            # initialize your key with some magic
            this.run=wandb.init(project=this.project, entity=this.entity, dir=this.dir);

    def __init__(this, cfg):
        if(cfg is None):
            this.set_up(find_data_root(),"/home/lasercat/hydra_saves/","/home/lasercat/hydra_logs/",["cuda:0"]);
        elif(type(cfg)==str):
            with open(cfg, "r") as fp:
                c = json.load(fp);
                this.set_up(c["data_root"], c["save_root"],c["log_root"], c["devices"],neko_get_arg("wandb_key",c,"NEP_skipped_NEP"));
        else:
            this.set_up(cfg["data_root"],cfg["save_root"],cfg["log_root"],cfg["devices"],neko_get_arg("wandb_key",cfg,"NEP_skipped_NEP"));


class platform_cfg:
    def set_up(this, data_root, save_root,log_root,devices):
        this.data_root = data_root;
        this.log_root=log_root;
        this.wandbdir=log_root; # backward compatition
        this.devices=devices;
        this.save_root = save_root;
        os.makedirs(this.save_root, exist_ok=True);
        os.makedirs(this.log_root, exist_ok=True);

    def __init__(this, cfg):
        if(cfg is None):
            this.set_up(find_data_root(),"/home/lasercat/hydra_saves/","/home/lasercat/hydra_logs/",["cuda:0"]);
        elif(type(cfg)==str):
            with open(cfg, "r") as fp:
                c = json.load(fp);
                this.set_up(c["data_root"], c["save_root"],c["log_root"], c["devices"]);
        else:
            this.set_up(cfg["data_root"],cfg["save_root"],cfg["log_root"],cfg["devices"]);