import os
import argparse
from simulators import (
    load_config, SpiritDynamicsPybullet
)

def main(config_file):
    config = load_config(config_file)
    config_env = config["environment"]
    config_agent = config["agent"]
    config_solver = config["solver"]

    # NOTE: This is supposed to be intialized in the following way: env --> agent --> dyn
    dynamics = SpiritDynamicsPybullet(config_agent)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str,
      default=os.path.join("simulators", "race_car", "race_car_env_v1.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)