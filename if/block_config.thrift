namespace py block_config
namespace py3 block_config


// ID Config

struct FeatSelectionConfig{
  // ID of the block (in DAG) to query input features from
  1: i32 block_id;
  // feature IDs to be queried from the target block output,
  2: list<i32> dense;
  3: list<i32> sparse;
}


// blocks
enum ExtendedBlockType{
  MLP_DENSE = 1;
  MLP_EMB = 2;
  CROSSNET = 3;
  FM_DENSE = 4;
  FM_EMB = 5;
  DOTPROCESSOR_DENSE = 6;
  DOTPROCESSOR_EMB = 7;
  CAT_DENSE = 8;
  CAT_EMB = 9;
  CIN = 10;
  ATTENTION = 11;
}

struct DenseBlockType{
}

struct EmbedBlockType{
  // embedding dimension, only used with "emb" type since features
  // may be with different dimensions and need to be concatenated column-wisely
  1: i32 comm_embed_dim;
  // treat dense feature as sparse feature, only for raw inputs, do not apply
  // for intermediate inputs
  2: bool dense_as_sparse=false;
}

union BlockType{
  1: DenseBlockType dense;
  2: EmbedBlockType emb;
}

struct MLPBlockConfig{
  // name of the block, might be removed later, but is convenient currently
  1: string name = "MLPBlock";
  // ID of the block DAG
  2: i32 block_id;
  // Input Feature IDs
  // (containing both block id and the corresponding dense & sparse feature ids)
  3: list<FeatSelectionConfig> input_feat_config;
  // type of the MLP module, either "dense" or "emb"
  // dense: dense MLP, which first row-wise concatenate all features
  // into one vector and then do MLP
  // emb: concatenate features column-wise concatenate all features
  // into a matrix/tensor and then do MLP
  4: BlockType type;
  // architecture of the MLP, e.g., [128, 100, 3]
  5: list<i32> arc;
  6: bool ly_act=true;
}

// TODO: to be tested
struct CrossNetBlockConfig{
  1: string name = "CrossNetBlock";
  2: i32 block_id;
  3: list<FeatSelectionConfig> input_feat_config;
  // number of crossnet layers
  4: i32 num_of_layers = 2;
  // Input Feature IDs of the cross features, will be automatically embedded
  // into the same dimension with the input feature and conduct cross operation
  5: list<FeatSelectionConfig> cross_feat_config;
  6: bool batchnorm=false;
}

struct FMBlockConfig{
  1: string name = "FMBlock";
  2: i32 block_id;
  3: list<FeatSelectionConfig> input_feat_config;
  4: BlockType type;
}


struct DotProcessorBlockConfig{
  1: string name = "DotProcessorBlock";
  2: i32 block_id;
  3: list<FeatSelectionConfig> input_feat_config;
  4: BlockType type;
}

// TODO: the cat block may need to be removed since it does not provide
// additional functionality?
struct CatBlockConfig{
  1: string name = "CatBlock";
  2: i32 block_id;
  3: list<FeatSelectionConfig> input_feat_config;
  4: BlockType type;
}

// TODO: to be tested
struct CINBlockConfig{
  1: string name = "CINBlock";
  2: i32 block_id;
  3: list<FeatSelectionConfig> input_feat_config;
  4: EmbedBlockType emb_config;
  // architecture of the CIN, e.g., [128, 100, 3]
  5: list<i32> arc;
  // decide half or full of the feature maps should be connected to output unit.
  6: bool split_half=true;
}

// TODO: to be tested
struct AttentionBlockConfig{
  1: string name = "AttentionBlock";
  2: i32 block_id;
  3: list<FeatSelectionConfig> input_feat_config;
  4: EmbedBlockType emb_config;
  5: i32 att_embed_dim = 10;
  // number of head of the attention block
  6: i32 num_of_heads = 2;
  // number of attention layers
  7: i32 num_of_layers = 1;
  8: float dropout_prob=0.0;
  // is use residual connection
  9: bool use_res=true;
  10:bool batchnorm=false;
}


union BlockConfig{
  1: MLPBlockConfig mlp_block;
  2: CrossNetBlockConfig crossnet_block;
  3: FMBlockConfig fm_block;
  4: DotProcessorBlockConfig dotprocessor_block;
  5: CatBlockConfig cat_block;
  6: CINBlockConfig cin_block;
  7: AttentionBlockConfig attention_block;
}
