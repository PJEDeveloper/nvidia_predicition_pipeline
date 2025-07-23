# pipeline/train_meta_model.py

from models.ensemble import train_meta_model


def main():
    LOOKBACK_PERIODS = [365, 270, 180, 90, 60, 30, 14, 1]
    train_meta_model(LOOKBACK_PERIODS)


if __name__ == "__main__":
    main()