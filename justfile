default:
    just --list

dev:
    rye run python ./src/main.py --target_dir demo