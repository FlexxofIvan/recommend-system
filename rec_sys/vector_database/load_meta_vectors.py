import hydra
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer

from rec_sys.dataset_modules.cols_data.parquet_data_utils import load_metadata
from pathlib import Path

from rec_sys.dataset_modules.cols_data.vector_data_utils import vectorize_df

@hydra.main(version_base=None, config_path="../../config/data", config_name="metadata_config")
def main(cfg: DictConfig):
    sent_model = SentenceTransformer(cfg.model_name)
    sent_model_call = sent_model.encode
    df = load_metadata(mus_file=Path(cfg.mus_file),
                       max_title_length=cfg.max_title_length,
                       max_desc_length=cfg.max_desc_length,
                       meta_file=Path(cfg.meta_file))
    vect_meta = vectorize_df(df, sent_model_call, cfg.words_fields, cfg.batch_size)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vect_meta.write_parquet(output_path)
    vect_meta.write_parquet(cfg.output_path)

if __name__ == "__main__":
    main()