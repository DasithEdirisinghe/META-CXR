PATH_TO_MIMIC_CXR = "/home/cse/Documents/meta-cxr/mimic-cxr" #TODO set your own path to MIMIC-CXR-JPG dataset (should point to a folder containing "mimic-cxr-jpg" folder)
PATH_TO_MIMIC_NLE = "<PATH_TO_MIMIC_NLE>" #TODO set your own path to MIMIC-NLE dataset (should point to a folder containing "mimic-nle" folder)
VIS_ROOT = f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0"

JAVA_HOME = "/usr/lib/jvm/java-8-openjdk-amd64/jre" #TODO set your own path to java home, adapt version if necessary
JAVA_PATH = "/usr/lib/jvm/java-8-openjdk-amd64/jre/bin:"

CHEXBERT_ENV_PATH = '/home/cse/miniconda3/envs/chexbert/bin/python'  # Linux path to Python environment

CHEXBERT_PATH = '/home/cse/Documents/meta-cxr/RaDialog/chexbert/src'  # Linux path to chexbert project

WANDB_ENTITY = "dasith-dev-uom" #TODO set your own wandb entity