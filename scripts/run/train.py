import argparse
import os

from .run_from_files import run

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train policy on given environment using specified algorithm.")
    parser.add_argument("--algo", default="hits", help="Which algorithm to use (hac or hits).")
    parser.add_argument("--env", default="Platforms", help="Which environment to run (Platforms, Drawbridge or Tennis2D).")

    args = parser.parse_args()

    assert args.algo in {"hits", "hac", "sac"}
    assert args.env in {"AntFourRooms", "Drawbridge", "Pendulum", "Platforms", 
            "Tennis2D", "UR5Reacher"}
    
    print(f"Training with {args.algo} on {args.env} environment.")
    
    path = os.path.join("./data", args.env, args.algo + "_trained")

    run(path)
