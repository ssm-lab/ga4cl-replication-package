import json
import gymnasium as gym
import matplotlib.pyplot as plt
import env
import sys
import os


def show_env(desc, title):
    environment = gym.make(
        "BurningForest-v0",
        render_mode="rgb_array",
        desc=desc,
    )
    obs, info = environment.reset()
    frame = environment.render()
    plt.figure(figsize=(4, 4))
    plt.imshow(frame)
    plt.axis("off")
    plt.savefig(f"{title}.png", bbox_inches="tight", dpi=250)
    # plt.show()
    print(f"{title}")
    print(f"Map: {desc}")
    total_cells = sum(len(row) for row in desc)
    b_count = sum(row.count("B") for row in desc)
    print(f"B ratio: {b_count}/{total_cells} ({b_count / total_cells:.2%})")
    environment.close()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "../01-configurations/environment-config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Create output folder: ../04-results/map_visualization/
    output_dir = os.path.join("..", "04-results", "map_visualization")
    os.makedirs(output_dir, exist_ok=True)

    for env_config in config["environments"]:
        env_id = env_config["env_id"]
        desc = env_config["desc"]
        complexity = env_config["complexity"]
        save_path = os.path.join(output_dir, f"{env_id}-complexity_{complexity:.2f}")
        show_env(desc=desc, title=save_path)

    print(f"\nAll maps saved to: {output_dir}/")