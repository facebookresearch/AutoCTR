include "block_config.thrift"

namespace py config
namespace py3 config


struct DataFromFileConfig {
  1: string data_file;
  2: i32 batch_size = 100;
  3: i32 num_batches = -1;
  4: list<float> splits = [0.80, 0.10];
  5: i32 num_samples_meta = 100000;
}

union DataConfig {
  1: DataFromFileConfig from_file;
}

enum MacroSearchSpaceType{
  # Input feature independently differentiate
  INPUT_DIFF = 1;
  # Input feature group-wise
  INPUT_GROUP = 2;
  # Input feature independently differentiate with learnable group-wise prior
  INPUT_DIFF_PRIOR = 3;
  # Similar to INPUT_DIFF_PRIOR, but this is a smooth transit between 1 & 2
  INPUT_ELASTIC_PRIOR = 4;
}


struct MicroClose{
}

struct MicroMLPConfig {
  // micro architecture of MLP block
  1: list<i32> arc;
}

struct MicroCINConfig {
  // micro architecture of CIN block
  1: list<i32> arc;
  2: list<i32> num_of_layers = [1, 2, 3];
}

struct MicroAttentionConfig {
  // micro architecture of Attention block
  1: list<i32> num_of_layers = [1, 2, 3];
  2: list<i32> num_of_heads = [1, 2, 3];
  3: list<i32> att_embed_dim = [10];
  4: list<float> dropout_prob = [0.0, 0.2, 0.4];
}

union MicroSearchSpaceType{
  # Do not do micro search
  1: MicroClose close;
  # Do micro mlp units search (single layer)
  2: MicroMLPConfig micro_mlp;
  # Do micro CIN arc search (layer num + layer unit, every layer sample #units)
  3: MicroCINConfig micro_cin;
  # Do micro Attention arc search
  4: MicroAttentionConfig micro_attention;
}

struct InputDenseAsSparse{
}

union FeatureProcessingType{
  # treat input dense features as sparse features
  1: InputDenseAsSparse idasp;
}

struct NASRecNetConfig {
  1: list<block_config.BlockConfig> block_configs;
}


struct RandomSearcherConfig {
  1: i32 max_num_block = 3;
  2: list<block_config.ExtendedBlockType> block_types;
  3: MacroSearchSpaceType macro_space_type = INPUT_DIFF;
  5: list<MicroSearchSpaceType> micro_space_types;
  6: list<FeatureProcessingType> feature_processing_type = [];
}


struct EvolutionarySearcherConfig {
  1: i32 max_num_block = 3;
  2: list<block_config.ExtendedBlockType> block_types;
  3: i32 population_size = 10;
  4: i32 candidate_size = 5;
  5: MacroSearchSpaceType macro_space_type = INPUT_DIFF;
  7: list<MicroSearchSpaceType> micro_space_types;
  8: list<FeatureProcessingType> feature_processing_type = [];
}

union SearcherConfig {
  1: RandomSearcherConfig random_searcher;
  2: EvolutionarySearcherConfig evolutionary_searcher;
}

union ModelConfig {
  1: NASRecNetConfig nasrec_net;
}

// optimizers

struct SGDOptimConfig {
  1: float lr = 0.01;
  2: float momentum = 0.0;
  3: float dampening = 0.0;
  4: bool nesterov = false;
  5: float weight_decay = 0.0;
}

struct AdagradOptimConfig {
  1: float lr = 0.01;
  2: float lr_decay = 0.0;
  3: float weight_decay = 0.0;
  4: float initial_accumulator_value = 0.0;
}

struct SparseAdamOptimConfig {
  1: float lr =0.001;
  2: float betas0 = 0.9;
  3: float betas1 = 0.999;
  4: float eps = 1e-08;
}

struct AdamOptimConfig {
  1: float lr = 0.001;
  2: bool amsgrad = false;
  3: float weight_decay = 0.0;
  4: float betas0 = 0.9;
  5: float betas1 = 0.999;
  6: float eps = 1e-08;
}

struct RMSpropOptimConfig {
  1: float lr = 0.01;
  2: float alpha = 0.99;
  3: float weight_decay = 0.0;
  4: float momentum = 0.0;
  5: bool centered = false;
  6: float eps = 1e-08;
}

union OptimConfig {
  1: SGDOptimConfig sgd;
  2: AdagradOptimConfig adagrad = {};
  3: SparseAdamOptimConfig sparse_adam;
  4: AdamOptimConfig adam;
  5: RMSpropOptimConfig rmsprop;
}

// features

struct SumPooling {}

struct AvgPooling {}

union PoolingConfig {
  1: SumPooling sum = {};
  2: AvgPooling avg;
}

struct SparseFeatureItem {
  1: string name;
  2: i32 hash_size = 10000;
  3: i32 embed_dim = -1;
  4: optional OptimConfig optim;
  5: PoolingConfig pooling = {"sum": {}};
}

struct SparseFeatureConfig {
  1: list<SparseFeatureItem> features = [];
  2: i32 embed_dim = -1;
  3: OptimConfig optim;
}

struct DenseFeatureConfig{
  1: list<string> features;
  2: OptimConfig optim;
}

struct FeatureConfig {
  1: DenseFeatureConfig dense;
  2: SparseFeatureConfig sparse;
}


// loss

struct BCEWithLogitsLoss {
}

struct BCELoss {
}

struct MSELoss {
}

union LossConfig {
  1: BCEWithLogitsLoss bcewithlogits;
  2: BCELoss bce;
  3: MSELoss mse;
}

struct LoggingConfig {
  1: i32 log_freq = 10000;
  2: i32 tb_log_freq = -1;
  3: bool tb_log_model_weight_hist = false;
  4: bool tb_log_pr_curve_batch = true;
  5: list<string> tb_log_model_weight_filter_regex = ["sparse"];
}

struct TrainConfig {
  1: LoggingConfig logging_config;
  3: i32 nepochs = 1;
  // if true, the training will be terminated when the validation loss start
  // to increase, if avialable
  5: bool early_stop_on_val_loss = true;
  6: LossConfig loss = {
    "bcewithlogits": {}
  };
}

struct EvalConfig {
  1: LoggingConfig logging_config;
  2: LossConfig loss = {
    "bcewithlogits": {}
  };
  3: bool compute_ne = true;
}

struct CheckpointConfig {
  1: i32 ckp_interval = 10;
  2: string ckp_path = "";
}

struct KoskiReaderConfig {
  1: i64 prefetch_capacity = 128;
  2: bool pin_memory = true;
  3: i32 num_workers = 4;
}

struct PerformanceConfig {
  1: bool use_gpu = false;
  2: i32 num_readers = 4;
  3: i32 num_trainers = 1;
  4: CheckpointConfig ckp_config = {
    "ckp_interval": 10
  };
  5: i32 data_queue_maxsize = 100;
  6: i32 reader_threads = 8;
  7: i32 num_gpu = 1;
  8: optional bool enable_profiling = false;
  9: optional KoskiReaderConfig koski;
  10: optional i32 omp_num_threads = 0;
}
