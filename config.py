import argparse

parser = argparse.ArgumentParser(description='MODEL NAME')
parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
parser.add_argument("--config_path", type=str, default="./config/burgers/fno.yaml")
