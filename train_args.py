import logging
import math
from typing import Literal, Optional, Union 

from dataclasses import dataclass, field

from setfit import TrainingArguments
from sentence_transformers.losses import BatchHardTripletLossDistanceFunction, TripletLoss

import util

logger = logging.getLogger(__name__)

LEGACY_SAMPLING_STRATEGIES = ["oversampling", "undersampling", "unique"]
GENERAL_SAMPLING_STRATEGIES = ["random", "topk", "hard", "semi"] 
POSITIVE_SAMPLING_STRATEGIES =  GENERAL_SAMPLING_STRATEGIES + ["all", "easyk", "topkmix"]
NEGATIVE_SAMPLING_STRATEGIES = GENERAL_SAMPLING_STRATEGIES + ["topkclass", "distance"]

@dataclass
class MiningArguments(TrainingArguments):

    positive_sampling_strategy: Optional[str] = None
    negative_sampling_strategy: Optional[str] = None   
    num_sampling_epochs: int = 1

    # topk
    k: Union[int, tuple[int, int]] = 1  # NOTE: k[1] is basically a multiplier for the number of negatives vs. positives, unless we have independent negatives
    skip_k: int = 0 
    sampling_margin: Optional[float] = None
    max_margin: Optional[float] = None # TODO: scaling margin until max

    # semi/hard sampling flags
    remove_pos_pairs: bool = False
    use_pair_dist: bool = False

    # whether to pass pairs and use the provides anchors or sample negatives for every anchor
    # NOTE: doesn't work with triplet loss (TODO)
    independent_negatives: bool = False

    no_duplicates: bool = True
    min_pairs: Union[int, tuple[int, int]] = field(default=(0, 0))
    max_pairs: Union[int, tuple[int, int]] = field(default=(0, 0))

    # maybe something like 
    # Seems like "hard" selection methods struggle with outliers, especially the "hardest"/topk methods..
    # this might fix that. e.g. disregard positive pairs that are > max dist 
    pos_lim: tuple[Union[float, None], Union[float, None]] = field(default=(None, None))
    neg_lim: tuple[Union[float, None], Union[float, None]] = field(default=(None, None))
    quantile_limits: bool = False

    # TODO fill / reduce / maybe something like fill hards/remove easy or something..
    # balance_method: str = "fill" 
    
    # old schedule = create new optimizer/scheduler for each sampling/training epoch => behaves kind of like "stochastic gradient descent with warm resets"
    old_lr_schedule: bool = False
    no_loop: bool = False
    min_steps_for_warmup: int = 10  # NOTE: having a warmup step when there is only very few total steps not very beneficial..
    normalize_embeddings: bool = False

    # something for saving (like save only last distances)
    save_distances: bool = False
    only_last_distance: bool = False 
    save_pairs: bool = False
    evaluate_test_dataset: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.sampling_strategy not in LEGACY_SAMPLING_STRATEGIES + ["mining"]:
            # try to split
            strats = self.sampling_strategy.split("-")
            self.positive_sampling_strategy = strats[0]
            if len(strats) > 1:
                self.negative_sampling_strategy = strats[1]

        if self.positive_sampling_strategy is not None:
            if self.positive_sampling_strategy not in POSITIVE_SAMPLING_STRATEGIES:
                raise ValueError(f"Positive mining strategy must be one of: {POSITIVE_SAMPLING_STRATEGIES}.")
            self.sampling_strategy = "mining"

            if self.negative_sampling_strategy is None:
                if self.positive_sampling_strategy in GENERAL_SAMPLING_STRATEGIES:
                    self.negative_sampling_strategy = self.positive_sampling_strategy
                else:
                    logger.warning("Positive mining strategy incompatible with negative sample selection. Defaulting to random.")
                    self.negative_sampling_strategy = "random"

        if self.negative_sampling_strategy == "distance":
            # distance metric 'cutoffs' only work with euclidean distance (for now.. need to figure out good values for cosine sim..)
            if self.distance_metric == BatchHardTripletLossDistanceFunction.cosine_distance:
                logger.warning("'Distance' Sampling only works with euclidean distance as the distance metric! (for now..)")
                self.distance_metric = BatchHardTripletLossDistanceFunction.eucledian_distance
                self.normalize_embeddings = True

        if self.num_iterations:
            # if num_iterations set ignore all this and go old-school setfit sampling
            self.sampling_strategy = "oversampling"

        self.sampling_margin = self.margin if self.sampling_margin is None else self.sampling_margin

        if isinstance(self.k, int):
            # self.k = (self.k, self.k)
            self.k = (self.k, self.k) if self.independent_negatives else (self.k, 1) 
                    
        if isinstance(self.min_pairs, int):
            self.min_pairs = (self.min_pairs, self.min_pairs)

        if isinstance(self.max_pairs, int):
            self.max_pairs = (self.max_pairs, self.max_pairs)

    def update(self, arguments, ignore_extra = False):
        return MiningArguments.from_dict({**self.to_dict(), **arguments}, ignore_extra=ignore_extra)

    def get_setfit_args(self):
        return TrainingArguments.from_dict(self.to_dict(), ignore_extra=True)

    def get_modified(self):
        _default = MiningArguments().to_dict()
        changes = {}
        for key, value in self.to_dict().items():
            if value != _default[key]:
                changes[key] = value
        
        return changes

    def to_json_dict(self):
        d = self.to_dict()
        d["distance_metric"] = util.DISTANCE_FUNCTION_MAP[d["distance_metric"]]
        d["loss"] = util.LOSS_MAP[d["loss"]]
        return {k: v for k,v in d.items()}
        
    @classmethod
    def from_json_dict(cls, arguments):
        d = {}
        for k,v in arguments.items():
            if k == "distance_metric":
                d[k] = util.DISTANCE_FUNCTION_MAP[v]
            elif k == "loss":
                d[k] = util.LOSS_MAP[v]
            else: 
                d[k] = v

        return cls(**d)

    def get_run_name(self):
        if self.run_name:
            return self.run_name
        else:
            # sampling_strategy
            if self.sampling_strategy != "mining":
                out = f"{self.num_iterations}-iterations" if self.num_iterations else self.sampling_strategy
            else:
                out = self.positive_sampling_strategy + "-" + self.negative_sampling_strategy
                out += f"-{self.num_sampling_epochs}"
                # if "k" in out or "random" in out:
                # all methods take k now soo..
                out += f"-{self.k[0]}p{self.k[1]}nK"
                if ("semi" in out) or ("hard" in out):
                    out += f"-{self.sampling_margin}m"

                if any(self.min_pairs):
                    out += "-min" + (f"{self.min_pairs[0]}p" if self.min_pairs[0] else "")
                    out += (f"{self.min_pairs[1]}n" if self.min_pairs[1] else "")

                if any(self.max_pairs):
                    out += "-max" + (f"{self.max_pairs[0]}p" if self.max_pairs[0] else "")
                    out += (f"{self.max_pairs[1]}n" if self.max_pairs[1] else "")

                if any(self.pos_lim):
                    out += "-" + (f"{self.pos_lim[0]}p" if self.pos_lim[0] else "")
                    out += (f"{self.pos_lim[1]}P" if self.pos_lim[1] else "")

                if any(self.neg_lim):
                    out += "-" + (f"{self.neg_lim[0]}n" if self.neg_lim[0] else "")
                    out += (f"{self.neg_lim[1]}N" if self.neg_lim[1] else "")

                # last bits...
                flags = ""
                # flags += ("" if self.sample_selection == "anchor" else "F")
                flags += ("T" if self.loss == TripletLoss else "")
                flags += ("" if self.no_duplicates else "D")
                flags += ("N" if self.normalize_embeddings else "")
                flags += ("O" if self.old_lr_schedule else "")
                flags += ("L" if self.no_loop else "")
                flags += ("Q" if self.quantile_limits else "")
                flags += ("I" if self.independent_negatives else "")
                # FIXME: if we start deleting in other methods we need to remove this as well!
                if self.negative_sampling_strategy in ["semi", "hard"]:
                    flags += ("R" if self.remove_pos_pairs else "")
                    flags += ("P" if self.use_pair_dist else "")
                if "topk" in out:
                    flags += (f"S{self.skip_k}" if self.skip_k else "")
                    
                flags += ("" if self.distance_metric == BatchHardTripletLossDistanceFunction.cosine_distance else "E")

                out += ("-"+flags if flags else "")

            return out

