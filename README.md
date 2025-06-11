# Code for thesis "Sample Mining for Contrastive Fine-Tuning of Sentence Transformers"

Basically a small wrapper around the `setfit` Trainer class, that changes the sentence transformer body training + sample mining methods.

## Dependencies
Download and install the required libraries by running:

```bash
pip install -r requirements.txt
```

Make sure to install torch with cuda first if you want the program to use your GPU.

## Training

The main file for training is `train.py`. For example, to train SetFit on SST-5 using the original sampling strategy using 4 and 64 samples per class run:
```
python train.py --num_iterations=20 --datasets sst5 --sample_sizes 4 64
```

By default training is done for all sample sizes `[2, 4, 8, 16, 32, 64]`.

If you want to train the model on all datasets use `--all_datasets`.

Here is and example using some of the more important input arguments to modify the training process, such as changing the mining/sampling strategy:

```
python train.py \
    --model=paraphrase-mpnet-base-v2 \
    --datasets sst2 sst5 emotion amazon_counterfactual_en \
    --sample_sizes 4 16 64 \
    --sampling_strategy=random-semi \
    --sampling_steps=10
    --k=2 \
    --sampling_margin=0.2 \
```

If you want to change the loss function to triplet loss, use `--loss=TripletLoss`. This should also change the sampling strategies behavior to return triplets etc.

We sometimes experienced memory issues, due to creating multiple `Trainer` objects in a loop. If you experience this, consider using the `--spawn_subprocess` option, which spawns a subprocess for every training process (everything still happens sequentially though).

## Mining Strategies

Mining/Sampling strategies can be changed through the `--sampling_strategy` argument. Either input a combination, such as `topk-semi` for Topk positive mining and semi-hard negative mining. Alternatively, if you provide only one method, it uses the same for both positive and negative mining (if possible).

`k` is the number of positive and negative pairs we draw for each training sample (can set `k_n` to generate a different number of negative pairs).

`sampling_steps` is the number of mining/sampling steps.

If you want to allow duplicates you can use the `--duplicates` input argument.

(For more input arguments look at `train.py` and `train_args.py`)

Below are the different mining strategies:

* Baselines
   * N-Iterations sampling  
      Use --num_iterations to set a value for N. Takes precedent over the --sampling_strategy option.
   * oversampling
* Positive Mining Strategies
   * random
   * all
   * easyk
   * topk
   * semi
   * hard

* Negative Mining Strategies
   * random
   * topk
   * semi
   * hard
   * distance