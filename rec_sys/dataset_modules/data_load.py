import hydra
from omegaconf import DictConfig

from rec_sys.dataset_modules.cols_data.parquet_data_utils import split_jsonl_by_user
from rec_sys.dataset_modules.make_dataset import prepare_data


@hydra.main(
    version_base=None, config_path="../../config/data", config_name="data_config"
)
def main(cfg: DictConfig):
    split_jsonl_by_user(
        input_path=cfg.files.raw_mus_file,
        train_path=cfg.files.mus_file.train,
        test_path=cfg.files.mus_file.test,
        user_field=cfg.dataset.user_field,
        test_ratio=cfg.dataset.test_size,
        seed=cfg.dataset.seed,
    )
    prepare_data(
        raw_mus_file=cfg.files.mus_file.test,
        metadata_file=cfg.files.raw_metadata_file,
        unique_user_dir=cfg.dirs.unique_user_data.test,
        graphs_dir=cfg.dirs.graphs.test,
        max_name_len=cfg.dataset.max_name_len,
        max_desc_len=cfg.dataset.max_desc_len,
        model_name=cfg.dataset.model_name,
        batch_size=cfg.dataset.batch_size,
        words_fields=cfg.dataset.words_fields,
        user_field=cfg.dataset.user_field,
        eval_ratio=cfg.dataset.eval_ratio,
    )
    prepare_data(
        raw_mus_file=cfg.files.mus_file.train,
        metadata_file=cfg.files.raw_metadata_file,
        unique_user_dir=cfg.dirs.unique_user_data.train,
        graphs_dir=cfg.dirs.graphs.train,
        max_name_len=cfg.dataset.max_name_len,
        max_desc_len=cfg.dataset.max_desc_len,
        model_name=cfg.dataset.model_name,
        batch_size=cfg.dataset.batch_size,
        words_fields=cfg.dataset.words_fields,
        user_field=cfg.dataset.user_field,
        eval_ratio=cfg.dataset.eval_ratio,
    )


if __name__ == "__main__":
    main()
