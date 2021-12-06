import argparse
import os


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description="Render episodes using policy read from provided directory.")
    parser.add_argument("--algo", default="hits", help="Which algorithm to use (hac or hits).")
    parser.add_argument("--env", default="Platforms", help="Which environment to run (Platforms, Drawbridge  or Tennis2D).")
    parser.add_argument("--newly_trained", default=False, action="store_true", help="Show newly trained policy and not pretrained one.")
    parser.add_argument("--stochastic", default=False, action="store_true", help="Show stochastic policy used during training.")

    args = parser.parse_args()

    assert args.algo in {"hits", "hac", "sac"}
    assert args.env in {"AntFourRooms", "Drawbridge", "Pendulum", "Platforms", 
            "Tennis2D", "UR5Reacher"}
    
    print(f"Running {args.algo} on {args.env} ({'newly trained' if args.newly_trained else 'pretrained'}).")
    
    path = os.path.join("./data", args.env, args.algo + "_" + ("trained" if args.newly_trained else "pretrained"))

    os.system(f"python -m scripts.run.render_from_files {path} {'--test ' if not args.stochastic else ' '}")
