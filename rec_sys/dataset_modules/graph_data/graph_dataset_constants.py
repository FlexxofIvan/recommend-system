EVAL_PRODUCT_RATIO = 0.2

FEAT_PRODUCT_INFO = "x_product_info"
FEAT_COMMENT = "x_comment"
FEAT_PRODUCT_NAME = "x_product_name"
VER_FLAG = "ver_flag"
TARGET = "target"

USER_EMB = "x"

USER_NODE = "user"
PRODUCT_TRAIN_NODE = "product_train"
PRODUCT_TEST_NODE = "product_test"
USER_REL_PRODUCT = "comments"


template = {
    "user": {"x": None, "num_nodes": None},
    "product_train": {
        "x_product_name": None,
        "x_product_info": None,
        "target": None,
        "num_nodes": None,
    },
    "product_test": {
        "x_product_name": None,
        "x_product_info": None,
        "target": None,
        "num_nodes": None,
    },
    ("product_train", "comments", "user"): {"edge_index": None, "comments": None},
}
