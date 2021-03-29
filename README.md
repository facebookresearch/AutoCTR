### AutoCTR

This repo provide the experimental code for the KDD paper: [Towards Automated Neural Interaction Discovery for Click-Through Rate Prediction](https://dl.acm.org/doi/10.1145/3394486.3403137). The paper introduces an automated interaction architecture discovering framework for CTR prediction named AutoCTR with the help of neural architecture search techniques.


### Environment

The repo has been tested with the **python3.6** environment. Currently, only GPU running is supported.


### Install FBthrift
Our general setting of the search space and model training are written with thrift files. If you made any changes, please install fb thrift for python first, codes are migrated from: https://github.com/facebook/fbthrift/tree/master/thrift/lib/py

 ```shell
cd py
python3.6 setup.py install
```
Add "--user" if required during installation


### Convert thrift files to python codes
Running the following codes can help you generate the corresponding python files for the thrift files. If you modify the thrift files in `/if` folder, please rerun the following codes to regenerate the python codes:

```shell
thrift -r --gen py if/config.thrift
```


### Install required packages

Install all the python packages required in the repo.

```shell
python3.6 -m pip install -r requirements.txt
```

### Data preprocessing
You should download and unzip the dataset ([criteo](www.kaggle.com/c/criteo-display-ad-challenge)/[avazu](https://www.kaggle.com/c/avazu-ctr-prediction/data)/[kdd2012](https://www.kaggle.com/c/kddcup2012-track2/data)) first, and preprocess the data with script `scripts/preprocess.py`. For example, if you want to preprocess criteo dataset, you can use the `shell-scripts/preprocess_criteo.sh` file.  Please make sure the raw data file path and save data file path, and dataset name is correct before doing the preprocessing. Also, checkout the `scripts/preprocess.py` file to see more arguments.

```shell
sh shell-scripts/preprocess_criteo.sh
```


### Data subsampling
You can follow, modify, and run the `shell-scripts/subsample_criteo.sh` file to subsampe criteo data. (Avazu, and KDD2012 dataset are similar.)

```shell
sh shell-scripts/subsample_criteo.sh
```


### Run random search
You may need to change some arguments such as "--total-gpus" to the number of total gpus on your machine. The description of all the arguments can be found in `utils/search_utils.py`.
```shell
sh shell-scripts/random_search_criteo.sh
```

### Run proposed evolutionary search
You may need to change some arguments such as "--total-gpus" to the number of total gpus on your machine. The description of all the arguments can be found in `utils/search_utils.py`.

```shell
sh shell-scripts/evo_search_criteo.sh
```

You can also run our proposed evolutionary search with first 20 arch warm-started by existing random search results if existed to save running time and keep fair comparison. The warm start file could be changed to other folders containing some search results. Also, you may need to change some arguments such as "--total-gpus" to the number of total gpus on your machine. The description of all the arguments can be found in `utils/search_utils.py`.

```shell
sh shell-scripts/evo_search_criteo_warm_start.sh
```


### Check out the results

You can use the jupyter notebooks in the `notebook` folder to display the search results (`plot.ipynb`), and the check the name and structure of the best architecutre (`graph.ipynb`). We provide two search results as example in the `results` folder.



### Final fit on larger datasets
To do the final fit of the best model on the **full** dataset, you can use the `notebook/graph.ipynb` to find out the name and json file of the best architecture. Then change the `--model-file` in the `random_final_fit_criteo.sh`. Please also change the `--data-file` to the dataset you wanna use. Also, change the gpu configurations such as "--total-gpus" to the number of total gpus on your machine.

**Note**: For other arguments of the `search.py` and `final_fit.py` scripts, please checkout `utils/search_utils.py` for more information.





## Cite this work

Qingquan Song, Dehua Cheng, Hanning Zhou, Jiyan Yang, Yuandong Tian, and Xia Hu. "Towards Automated Neural Interaction Discovery for Click-Through Rate Prediction." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2020. ([Download](https://dl.acm.org/doi/10.1145/3394486.3403137))

Biblatex entry:

```bibtex
@inproceedings{song2020towards,
  title={Towards automated neural interaction discovery for click-through rate prediction},
  author={Song, Qingquan and Cheng, Dehua and Zhou, Hanning and Yang, Jiyan and Tian, Yuandong and Hu, Xia},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={945--955},
  year={2020}
}
```

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
AutoCTR is Creative Commons Attribution-NonCommercial 4.0 International licensed, as found in the LICENSE file.
