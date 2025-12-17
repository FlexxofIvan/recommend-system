import fire

from rec_sys.dataset_modules.cols_data.parquet_data_utils import split_jsonl_by_user

def main():
    fire.Fire(split_jsonl_by_user)


if __name__ == "__main__":
    main()