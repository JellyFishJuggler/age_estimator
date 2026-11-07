import pandas as pd
from pathlib import Path

def path_parser(img_dir: str | Path) -> pd.DataFrame:
    img_dir = Path(img_dir)
    data = []

    for i in img_dir.glob("*.jpg"):
        filename = i.name
        parts = filename.split("_")
        if len(parts) < 4:
            parts = filename.split("-")

        if len(parts) >= 3:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])

                data.append({
                    "file_path":str(i),
                    "filename":filename,
                    "age":age,
                    "gender":gender,
                    "race":race
                })

            except ValueError:
                print(f'Skipping corrupted img : {filename}')
                continue

    df = pd.DataFrame(data)
    return df


def main() -> None:
    img_dir = Path(__file__).resolve().parent / "data"
    df = path_parser(img_dir)
    df.to_csv(img_dir / "dataset.csv", index=False)
    print(f"Successfully processed {len(df)} images!")
    print(df.head())


if __name__ == "__main__":
    main()