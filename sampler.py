import logging
import numpy as np
import torch
from transformers.utils import ExplicitEnum
from collections import defaultdict

from sentence_transformers.losses import BatchHardTripletLoss

from train_args import (
    MiningArguments,
    POSITIVE_SAMPLING_STRATEGIES,
    NEGATIVE_SAMPLING_STRATEGIES,
)
from util import _convert_to_numpy, _convert_to_tensor, get_mask

logger = logging.getLogger(__name__)


def get_unique_indices(indices: list[list[int, int] | tuple[int, int, int]]):
    # NOTE: returns list of list for triplets instead of list of tuples..
    # only sort when it's a list of pair indices.
    return np.unique(
        [(sorted(p)) if len(p) == 2 else p for p in indices], axis=0
    ).tolist()


def distance_sampling(
    distances: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
    no_duplicates: bool = True,
    positive_pairs: list | None = None,
    return_triplets: bool = False,
    rng: np.random.Generator | None = None,
    emb_dim: int = 768,
    cutoff: float = 0.5,
    nonzero_loss_cutoff: float = 1.4,
    no_loop: bool = False,
):
    # modified from https://github.com/suruoxi/DistanceWeightedSampling/blob/master/model.py
    # NOTE: need to think about other values for cosine dist? or just stick with euclidean dist only..
    if return_triplets and positive_pairs is None:
        raise ValueError(
            "When sampling negative samples for triplets topk-sampling expects a list of anchor and positive samples."
        )

    if rng is None:
        rng = np.random.default_rng()

    if mask is None:
        mask = get_mask(labels=labels, positive=False)
    else:
        mask = mask.clone()

    if positive_pairs is None:
        positive_pairs = [[i, 0] for i in range(len(mask))]

    num_samples = len(labels)
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    samples_per_label = num_samples // num_labels
    indices = []

    num_expected_idx = len(positive_pairs) * k
    logger.debug(f"Distance-Sampling: Expected num. indices: {num_expected_idx}")
    # check if we can just return all unique indices
    if no_duplicates and not return_triplets:
        present_anchors = torch.zeros_like(mask)
        for pair in positive_pairs:
            present_anchors[pair[0]] = True
        mask = mask * present_anchors

        num_unique = torch.count_nonzero(torch.triu(mask)).item()
        # expected_num_idx = num_samples * k
        if num_unique < num_expected_idx:
            logger.debug(
                f"Distance-Sampling: Expected indices {num_expected_idx} > num_unique {num_unique}. Returning all unique indices."
            )
            return torch.nonzero(torch.triu(mask)).tolist()

    # num_unique = torch.count_nonzero(torch.triu(mask)).item()
    distances = _convert_to_tensor(distances)

    # we don't care about positive pairs so set them to max distance
    max_dist, _ = distances.max(1, keepdim=True)
    dist = torch.where(mask == 0, max_dist + 1e-8, distances)

    # log weights
    dist = dist.clamp(min=cutoff)
    log_weights = (2.0 - float(emb_dim)) * dist.log() - (
        float(emb_dim - 3) / 2
    ) * torch.log(torch.clamp(1.0 - 0.25 * (dist * dist), min=1e-8))

    # normalize log weights
    log_weights = (log_weights - log_weights.min()) / (
        log_weights.max() - log_weights.min() + 1e-8
    )

    weights = torch.exp(log_weights - torch.max(log_weights))
    
    weights = weights * mask * (dist < nonzero_loss_cutoff).float()  # + 1e-8 # NOTE: this is what they do in the source.. but why?.. doesn't this just allow (although very unlikely) drawing invalid pairs?

    weights_sum = torch.sum(weights, dim=1, keepdim=True)
    weights = weights / weights_sum

    np_weights = weights.cpu().numpy()

    num_uniform = 0
    # NOTE: we need to loop here if we want to get as many pairs as expected with no duplicates
    # otherwise masking picked samples might result in fewer pairs than expected..
    prev_num_pairs = len(indices)
    while len(indices) < num_expected_idx:
        for pair in positive_pairs:
            anchor = pair[0]
            # NOTE: Probably a way better way and more efficient way to do all of this with recalculating weight etc. when we don't want duplicates
            # but hey it works.. hopefully..
            num_nan = np.count_nonzero(np.isnan(np_weights[anchor]))
            neg_indices = None
            if weights_sum[anchor] != 0 and num_nan == 0:
                # we got weights so use them
                num_nonzero = np.count_nonzero(np_weights[anchor])
                n = min(num_nonzero, k) if no_duplicates else k
                if num_nonzero > 0:
                    neg_indices = rng.choice(
                        num_samples,
                        size=n,
                        p=np_weights[anchor],
                        replace=not no_duplicates,
                    ).tolist()

            else:
                if weights_sum[anchor] != 0:
                    # means we had weights before but now exhausted all possible pairs
                    # don't want to start drawing from this "pool" as well
                    continue
                # uniform sampling
                candidates = torch.squeeze(torch.nonzero(mask[anchor]), dim=-1).tolist()
                num_candidates = len(candidates)
                if num_candidates > 0:
                    n = min(k, num_candidates) if no_duplicates else k
                    if n == -1:
                        n = num_candidates
                    neg_indices = rng.choice(
                        candidates, size=n, replace=n > num_candidates
                    ).tolist()

            if neg_indices is not None:
                for i in neg_indices:
                    pair_or_triplet = (
                        (anchor, pair[1], i) if return_triplets else [anchor, i]
                    )
                    if no_duplicates and not return_triplets:
                        # set probability to 0 for this pair
                        np_weights[anchor][i] = 0
                        np_weights[i][anchor] = 0
                        # normalize np_weights[i]
                        sum_i = np_weights[i].sum()
                        if sum_i > 0:
                            np_weights[i] /= sum_i
                        # set mask to 0 for this pairs in case we sampled uniformly
                        mask[anchor][i] = False
                        mask[i][anchor] = False

                    indices.append(pair_or_triplet)

                    if len(indices) >= num_expected_idx:
                        break

                # normalize weights for this row
                sum_anchor = np_weights[anchor].sum()
                if sum_anchor > 0:
                    np_weights[anchor] /= sum_anchor

            if len(indices) >= num_expected_idx:
                break

        if no_loop:
            break

        # sometimes had the issue of this infinitely looping, when samples get cut by the "non_zero_loss" cutoff
        # this seems like the easiest solution. When we don't find new pairs in an iteration we stop.
        if prev_num_pairs == len(indices):
            logger.warning(
                f"Distance-Sampling: No new samples found this loop. Stopping sampling."
            )
            logger.debug(f"Distance-Sampling: Num. unique pairs: {num_unique}")
            logger.debug(f"Distance-Sampling: Num. indices: {len(indices)}")
            logger.debug(f"Distance-Sampling: Num. uniform: {num_uniform}")
            logger.debug(f"Distance-Sampling: np_weights: {np_weights}")
            # logger.debug(f"Distance-Sampling: uniform_probs: {mask_uniform_probs}")
            break
        prev_num_pairs = len(indices)

    return indices

def topk_sampling(
    distances: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    k: int,
    skip_k: int = 0,
    mask: torch.Tensor | None = None,
    positive: bool = True,
    easy_positives: bool = False,
    no_duplicates: bool = True,
    positive_pairs: list | None = None,
    return_triplets: bool = False,
    no_loop: bool = False,
):
    if easy_positives and not positive:
        raise ValueError("Can't find easy positives when sampling negatives")
    if return_triplets and positive_pairs is None:
        raise ValueError(
            "When sampling negative samples for triplets TopK sampling expects a list of anchor and positive samples."
        )
    if return_triplets and positive:
        raise ValueError(
            "Trying to sample positive Triplets via TopK sampling. 'return_triplets' is exclusively for sampling negative pairs."
        )

    distances = _convert_to_tensor(distances)
    if mask is None:
        mask = get_mask(labels=labels, positive=positive)
    else:
        mask = mask.clone()

    if positive_pairs is None:
        positive_pairs = [[i, 0] for i in range(len(labels))]

    # check if we can just return all unique indices
    num_samples = len(positive_pairs)
    num_expected_idx = num_samples * k
    logger.debug(f"TopK-Sampling: Expected num. indices: {num_expected_idx}")
    if no_duplicates and not return_triplets:
        # mask rows that are not in positive pairs
        present_anchors = torch.zeros_like(mask)
        for pair in positive_pairs:
            present_anchors[pair[0]] = True
        mask_ = mask * present_anchors

        num_unique = torch.count_nonzero(torch.triu(mask_)).item()
        logger.debug(f"TopK-Sampling: Num. Unique: {num_unique}")
        if num_expected_idx >= num_unique:
            logger.debug(
                f"TopK-Sampling: Expected indices {num_expected_idx} > num_unique {num_unique}. Returning all unique indices."
            )
            return torch.nonzero(torch.triu(mask_)).tolist()

    masked_dist = distances * mask
    if not positive:
        masked_dist[~mask] = torch.inf

    indices = []
    if easy_positives:
        if not positive:
            raise ValueError("Can't find easy positives when sampling negatives")
        masked_dist[~mask] = torch.inf
        positive = not positive

    INVALID = 0 if positive else torch.inf

    while len(indices) < num_expected_idx:
        for pair in positive_pairs:
            anchor, posi = pair[0], pair[1]
            row = masked_dist[anchor]
            # print(row)
            top_vals, top_indices = torch.topk(row, k+skip_k, largest=positive, sorted=True)
            if skip_k:
                # we don't want to loop. 
                # checking how many samples are left with duplicates and stuff too much of a pain
                # -> at least no infinite loop this way..
                no_loop = True
                top_indices = top_indices[skip_k:]

            # print(top_indices)
            for i in top_indices.tolist():
                # print(f"d[{anchor}][{i}] = {masked_dist[anchor][i]}")
                if (
                    masked_dist[anchor][i] != INVALID
                    and masked_dist[i][anchor] != INVALID
                ):
                    pair_or_triplet = (
                        (anchor, posi, i) if return_triplets else [anchor, i]
                    )
                    indices.append(pair_or_triplet)
                    if no_duplicates and not return_triplets:
                        masked_dist[anchor][i] = masked_dist[i][anchor] = INVALID
                else:
                    break # topk values are sorted so when we encounter an invalid sample we can just stop searching..

                if len(indices) == num_expected_idx:
                    break
            if len(indices) == num_expected_idx:
                break
        if no_loop:
            break

    return indices


def random_sampling(
    labels: list | np.ndarray | torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
    positive: bool = True,
    no_duplicates: bool = True,
    positive_pairs: list | None = None,
    return_triplets: bool = False,
    rng: np.random.Generator | None = None,
    no_loop: bool = False,
):
    if return_triplets and positive_pairs is None:
        logger.warning(
            "Random-Sampling: When sampling negative samples for triplets Random sampling expects a list of anchor and positive samples."
        )
        logger.warning("Random-Sampling: Returning pairs instead (hopefully we are sampling positve pairs)")
        # assume we are sampling positive pairs and don't return triplets
        return_triplets = False

    if return_triplets and positive:
        logger.warning(
            "Random-Sampling: Trying to sample positive Triplets via Random sampling. 'return_triplets' is exclusively for sampling negative pairs."
        )
        logger.warning("Random-Sampling: Returning pairs instead (hopefully we are sampling positve pairs)")
        # assume we are sampling positive pairs and don't return triplets
        return_triplets = False

    if rng is None:
        rng = np.random.default_rng()

    if mask is None:
        mask = get_mask(labels=labels, positive=positive)
    else:
        mask = mask.clone()

    if positive_pairs is None:
        positive_pairs = [[i, 0] for i in range(len(labels))]

    # check if we can just return all unique indices
    num_samples = len(positive_pairs)
    num_expected_idx = num_samples * k
    logger.debug(f"Random-Sampling: Expected num. indices: {num_expected_idx}")
    if no_duplicates and not return_triplets:
        present_anchors = torch.zeros_like(mask)
        for pair in positive_pairs:
            present_anchors[pair[0]] = True
        mask = mask * present_anchors

        num_unique = torch.count_nonzero(torch.triu(mask)).item()
        logger.debug(f"Random-Sampling: Num. Unique: {num_unique}")
        if num_expected_idx >= num_unique:
            logger.debug(
                f"Random-Sampling: Expected indices {num_expected_idx} > num_unique {num_unique}. Returning all unique indices."
            )
            return torch.nonzero(torch.triu(mask)).tolist()

    # num_unique = torch.count_nonzero(torch.triu(mask))
    indices = []
    while len(indices) < num_expected_idx:
        for pair in positive_pairs:
            anchor, posi = pair[0], pair[1]
            candidates = torch.squeeze(torch.nonzero(mask[anchor]), dim=-1).tolist()
            num_candidates = len(candidates)
            # print(candidates)
            n = min(k, num_candidates) if no_duplicates else k
            if n == -1:
                n = num_candidates
            choices = rng.choice(
                candidates, size=n, replace=n > num_candidates
            ).tolist()
            # print(choices)
            for i in choices:
                pair_or_triplet = (anchor, posi, i) if return_triplets else [anchor, i]
                indices.append(pair_or_triplet)
                if no_duplicates and not return_triplets:
                    mask[anchor][i] = mask[i][anchor] = False

                if len(indices) == num_expected_idx:
                    break

            if len(indices) == num_expected_idx:
                break

        if no_loop:
            break

    return indices


def hard_sampling(
    distances: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None,
    positive: bool = True,
    margin: float | None = None,
    max_margin: float | None = None,
    break_max_margin:bool = True,
    no_duplicates: bool = True,
    semi: bool = False,
    use_pair_dist: bool = False,
    positive_pairs: list | None = None,
    return_triplets: bool = False,
    delete_positives: bool = True,  # NOTE: hmmm..
    rng: np.random.Generator | None = None,
    no_loop: bool = False,
):
    if return_triplets and positive_pairs is None:
        raise ValueError(
            "When sampling negative samples for triplets (Semi-)Hard sampling expects a list of anchor and positive samples."
        )
    if return_triplets and positive:
        raise ValueError(
            "Trying to sample positive Triplets via (Semi-)Hard sampling. 'return_triplets' is exclusively for sampling negative pairs."
        )
    if rng is None:
        rng = np.random.default_rng()

    distances = _convert_to_tensor(distances)
    if mask is None:
        mask = get_mask(labels=labels, positive=positive)
    else:
        mask = mask.clone()

    masked_dist = distances * mask
    if not positive:
        masked_dist[~mask] = torch.inf

    num_unique_pairs = torch.count_nonzero(torch.triu(mask)).tolist()
    logger.debug(f"(Semi-)Hard - Num. Unique Pairs = {num_unique_pairs}")
    indices = []

    if positive_pairs is None or positive:
        positive_pairs = [[i, 0] for i in range(len(distances))]
        if use_pair_dist:
            logger.warning("(Semi-)Hard - Can't use positive pair distance if no positive pairs are passed")
        use_pair_dist = False

    mask_opposite = get_mask(labels, positive=not positive)
    opposite_dist = distances * mask_opposite
    opposite_fn = torch.lt
    if positive:
        opposite_dist[~mask_opposite] = torch.inf
        opposite_fn = torch.gt

    hardest_val, _ = (
        opposite_dist.min(1, keepdim=True)
        if positive
        else opposite_dist.max(1, keepdim=True)
    )
    
    # candidates = opposite_fn(masked_dist, hardest_val)
    num_expected_idx = len(positive_pairs) * k
    logger.debug(f"(Semi-)Hard - Expected num. indices: {num_expected_idx}")
    triplets = defaultdict(list)
    no_candidates = set()
    cur_margin = margin if margin else 0.0
    max_margin = 0.0 if max_margin is None else max_margin
    num_found = 0
    num_iter = 0
    # TODO: repeat until expected number of indices is reached or no more candidates are left..
    while num_found < num_expected_idx and len(no_candidates) < len(positive_pairs):
        for pos_pair_id, pair in enumerate(positive_pairs):
            if pos_pair_id in no_candidates:  # skip if already no candidates
                continue
            anchor = pair[0]
            cur_dist_row = masked_dist[anchor].clone()
            # cur_dist_row = masked_dist[anchor]
            
            dist_to_beat = (
                distances[anchor][pair[1]] if (return_triplets or use_pair_dist) else hardest_val[anchor]
            )
            # NOTE: exclude pairs that are 'too hard' (e.g. we don't want D_ap > D_an)
            if semi:
                # determine easiest opposite value (max negative pair distance, min pos)
                opposite_row_dist = distances[anchor][mask_opposite[anchor]]
                easiest_opposite_val = opposite_row_dist.max() if positive else opposite_row_dist.min()
                # if we are not using actual pair distances, just use the easiest to determine whats "too hard"
                # this should gives us more options. Not totally correct probably, but we can always just mine triplets if we want 
                # actual valid semi-hard triplets..
                semi_bound = easiest_opposite_val if not use_pair_dist else distances[anchor][pair[1]]
                    
                valid_semi = (opposite_fn(semi_bound, cur_dist_row))
                cur_dist_row[~valid_semi] = 0 if positive else torch.inf
            
            if cur_margin:
                if positive:
                    dist_to_beat = torch.clamp(dist_to_beat - cur_margin, min=1e-16)
                else:
                    dist_to_beat = dist_to_beat + cur_margin

            candidates = torch.squeeze(
                torch.nonzero(opposite_fn(cur_dist_row, dist_to_beat)), dim=-1
            ).tolist()
            num_candidates = len(candidates)
            if num_candidates > 0:
                n = min(k, num_candidates) if no_duplicates else k
                if n == -1:
                    n = num_candidates

                chosen_negi = rng.choice(
                    candidates, size=n, replace=n > num_candidates
                ).tolist()

                for i in chosen_negi:
                    if mask[anchor][i] == 0:
                        logger.warning(
                            f"Mask for pair {anchor}-{i} is 0! Dist: {masked_dist[anchor][i]:.4f}, To-beat: {dist_to_beat.item():.4f}. Skipping this one..."
                        )
                        continue

                    pair_or_triplet = (
                        (anchor, pair[1], i) if return_triplets else [anchor, i]
                    )
                    indices.append(pair_or_triplet)
                    num_found += 1
                    triplets[pos_pair_id].append(i)

                    if no_duplicates and not return_triplets:
                        masked_dist[anchor][i] = masked_dist[i][anchor] = (
                            0 if positive else torch.inf
                        )

                    if num_found == num_expected_idx:
                        break
            else:
                if cur_margin >= max_margin:
                    no_candidates.add(pos_pair_id)

            if num_found == num_expected_idx:
                break

        num_iter += 1
        # grow margin if desired
        if cur_margin < max_margin:
            logger.debug(f"(Semi-)Hard: Found {num_found} so far. Increasing margin - {cur_margin} -> {cur_margin + 0.1}")
            cur_margin += 0.1
        else:
            if max_margin and break_max_margin:
                logger.debug(f"Reached max margin. Breaking.")
                break
            # cur_margin += something 
        if no_loop:
            break

    # maybe rename to replace positives..
    # should end up pretty much the same as old implementation
    # but with additional duplicates at the end to balance number of positive and negative pairs
    # if we don't do duplicates, they get removed after this anyways..
    if delete_positives:
        # print(triplets)
        # loop through all positive pair indices that we found negative pairs for
        valid_pos_pairs = list(triplets)
        num_valid_pos = len(valid_pos_pairs)
        _ppairs_copy = [pair for pair in positive_pairs]
        x = 0
        if valid_pos_pairs:
            for i in range(len(positive_pairs)):
                positive_pairs[i] = _ppairs_copy[valid_pos_pairs[x % num_valid_pos]]
                x += 1    

        # delete if we didn't find enough negative pairs.
        # this is to keep the number of positive and negative pairs balanced..
        if num_found < len(positive_pairs):
            missing = len(positive_pairs) - num_found
            for i in range(missing):
                del positive_pairs[len(_ppairs_copy)-(i+1)]

    return indices
    

def _get_pairs(
    distances: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    args: MiningArguments,
    mask: torch.Tensor | None = None,
    positive: bool = True,
    rng: np.random.Generator | None = None,
    return_triplets: bool = False,
    emb_dim: int = 768,
    positive_pairs: list | None = None,
):
    pairs = []
    if positive:
        strategy = args.positive_sampling_strategy
        k = args.k[0]
    else:
        strategy = args.negative_sampling_strategy
        k = args.k[1]

    no_duplicates = args.no_duplicates
    no_loop = args.no_loop

    if strategy == "random":
        pairs = random_sampling(
            labels=labels,
            k=k,
            mask=mask,
            positive=positive,
            no_duplicates=no_duplicates,
            positive_pairs=positive_pairs,
            return_triplets=return_triplets,
            rng=rng,
            no_loop=no_loop,
        )
    elif strategy == "topk":
        pairs = topk_sampling(
            distances=distances,
            labels=labels,
            k=k,
            skip_k=args.skip_k,
            mask=mask,
            positive=positive,
            easy_positives=False,
            no_duplicates=no_duplicates,
            positive_pairs=positive_pairs,
            return_triplets=return_triplets,
            no_loop=no_loop,
        )
    elif strategy == "easyk":
        pairs = topk_sampling(
            distances=distances,
            labels=labels,
            k=k,
            mask=mask,
            positive=positive,
            easy_positives=True,
            no_duplicates=no_duplicates,
            positive_pairs=positive_pairs,
            return_triplets=return_triplets,
            no_loop=no_loop,
        )

    elif strategy == "topkmix":
        pairs += topk_sampling(
            distances=distances,
            labels=labels,
            k=k,
            mask=mask,
            positive=positive,
            easy_positives=False,
            no_duplicates=no_duplicates,
            positive_pairs=positive_pairs,
            return_triplets=return_triplets,
            no_loop=no_loop,
        )

        pairs += topk_sampling(
            distances=distances,
            labels=labels,
            k=k,
            mask=mask,
            positive=positive,
            easy_positives=True,
            no_duplicates=no_duplicates,
            positive_pairs=positive_pairs,
            return_triplets=return_triplets,
            no_loop=no_loop,
        )

    elif strategy == "hard":
        pairs = hard_sampling(
            distances=distances,
            labels=labels,
            k=k,
            mask=mask,
            margin=args.sampling_margin,
            positive=positive,
            positive_pairs=positive_pairs,
            no_duplicates=no_duplicates,
            delete_positives=args.remove_pos_pairs,
            use_pair_dist=args.use_pair_dist,
            return_triplets=return_triplets,
            rng=rng,
            no_loop=no_loop,
        )
    elif strategy == "semi":
        pairs = hard_sampling(
            distances=distances,
            labels=labels,
            k=k,
            mask=mask,
            margin=args.sampling_margin,
            semi=True,
            positive=positive,
            positive_pairs=positive_pairs,
            no_duplicates=no_duplicates,
            delete_positives=args.remove_pos_pairs,
            use_pair_dist=args.use_pair_dist,
            return_triplets=return_triplets,
            rng=rng,
            no_loop=no_loop,
        )
    elif strategy == "distance":
        pairs = distance_sampling(
            distances=distances,
            labels=labels,
            k=k,
            mask=mask,
            no_duplicates=no_duplicates,
            positive_pairs=positive_pairs,
            # positive_pairs=positive_indices if return_triplets else None,
            return_triplets=return_triplets,
            emb_dim=emb_dim,
            rng=rng,
            no_loop=no_loop,
        )
    elif args.negative_sampling_strategy == "topkclass":
        mask_clone = mask.clone()
        num_uni_labels = len(np.unique(labels))
        for cur_label in range(num_uni_labels):
            label_mask = get_mask(labels=labels, positive=False, target_label=cur_label)
            combined_mask = mask_clone * label_mask
            cur_indices = topk_sampling(
                distances=distances,
                labels=labels,
                k=k,
                mask=combined_mask,
                positive=False,
                no_duplicates=no_duplicates,
                # positive_pairs=positive_indices if return_triplets else None,
                positive_pairs=positive_pairs,
                return_triplets=return_triplets,
                no_loop=no_loop,
            )
            pairs += cur_indices
            if no_duplicates:
                for pair in cur_indices:
                    mask_clone[pair[0]][pair[1]] = mask_clone[pair[1]][pair[0]] = False

    return pairs

def _count_non_zero(mat: torch.Tensor) -> int:
    if mat.dim() > 1:
        mat = torch.triu(mat)

    return torch.count_nonzero(mat).tolist()

def exclude_pairs(
    distances: torch.Tensor,
    mask: torch.Tensor,
    low:float | None,
    high:float | None,
    as_quantiles: bool = False,
    per_anchor: bool = True,
) -> tuple[torch.Tensor]:
    if per_anchor and as_quantiles:
        logging.disable(logging.DEBUG)
        mask_c = mask.clone()
        for i in range(len(mask)):
            row_mask = mask_c[i]
            row_distances = distances[i]
            exc_row_mask = exclude_pairs(row_distances, row_mask, low, high, as_quantiles, False)
            mask_c[i] = exc_row_mask

        logging.disable(logging.NOTSET)
        logger.debug(f"Left after excluding: {_count_non_zero(mask)}")
        return mask_c
    # find actual values if we treat input 'low' 'high' as quantiles
    num_pre_exclude = _count_non_zero(mask)
    if as_quantiles:
        # NOTE: apparently 'too hard' to do from the start. Gets stuck in local minima (a lot..)
        logger.debug(f"Treating limits as quantiles.")
        _low = low
        _high = high
        if low is not None:
            low = torch.quantile(distances[mask], low)
            # low = np.quantile(m_distances.numpy(), q=low)
            logger.debug(f"Low Limit ({_low}Q) = {low}.")
        if high is not None:
            high = torch.quantile(distances[mask], high)
            # high = np.quantile(m_distances.numpy(), q=high)
            logger.debug(f"High Limit ({_high}Q) = {high}")

    # exclude based on limits
    num_low_exclude = num_pre_exclude
    num_high_exclude = num_pre_exclude
    if low is not None:
        mask = mask * (distances >= low)
        num_low_exclude = _count_non_zero(mask)
    if high is not None:
        mask = mask * (distances <= high)
        num_high_exclude = _count_non_zero(mask)
    
    logger.debug(f"Num. Unique Samples = {num_pre_exclude}")
    if low is not None:
        logger.debug(f"Num. excluded low limit = {num_pre_exclude - num_low_exclude}")
    if high is not None:
        logger.debug(f"Num. excluded high limit = {num_low_exclude - num_high_exclude}")

    logger.debug(f"Left after excluding: {_count_non_zero(mask)}")
    return mask


def get_samples(
    distances: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    args: MiningArguments,
    return_triplets=False,
    emb_dim: int = 768,
    rng: np.random.Generator | None = None,
    n_rng: np.random.Generator | None = None,
    positive_indices=[],
    negative_indices=[],
) -> tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """_summary_

    Args:
        distances (list | np.ndarray | torch.Tensor): pairwise distance matrix
        labels (list | np.ndarray | torch.Tensor): list of labels
        args (MiningArguments): Arguments needed for sampling/mining.
        emb_dim (int, optional): Dimension of embeddings returned by the sentence transformer. Only required by distance sampling. Defaults to 768.
        return_triplets (bool, optional): If True returns (anchor,positive,negative) triplets instead of negative pair indices. Defaults to False.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_rng is None:
        n_rng = rng

    num_samples = len(labels)
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    samples_per_label = num_samples // num_labels

    # treat min/max pairs as multipliers of num_samples if negative
    min_pos = (
        num_samples * -args.min_pairs[0] if args.min_pairs[0] < 0 else args.min_pairs[0]
    )
    min_neg = (
        num_samples * -args.min_pairs[1] if args.min_pairs[1] < 0 else args.min_pairs[1]
    )
    max_pos = (
        num_samples * -args.max_pairs[0] if args.max_pairs[0] < 0 else args.max_pairs[0]
    )
    max_neg = (
        num_samples * -args.max_pairs[1] if args.max_pairs[1] < 0 else args.max_pairs[1]
    )

    mask_pos = get_mask(labels, positive=True)
    mask_neg = get_mask(labels, positive=False)

    # exclude pairs based on limits
    num_unique_pos = torch.count_nonzero(torch.triu(mask_pos)).item()
    num_unique_neg = torch.count_nonzero(torch.triu(mask_neg)).item()
    logger.debug(f"Number of unique positive pairs = {num_unique_pos}.")
    logger.debug(f"Number of unique negative pairs = {num_unique_neg}.")

    if any(args.pos_lim):
        mask_pos = exclude_pairs(distances, mask_pos, low=args.pos_lim[0], high=args.pos_lim[1], as_quantiles=args.quantile_limits)

    if any(args.neg_lim):
        mask_neg = exclude_pairs(distances, mask_neg, low=args.neg_lim[0], high=args.neg_lim[1], as_quantiles=args.quantile_limits)

    # update in case we removed pairs based on limits
    num_unique_pos = torch.count_nonzero(torch.triu(mask_pos)).item()
    num_unique_neg = torch.count_nonzero(torch.triu(mask_neg)).item()

    random_pos_indices = []
    random_neg_indices = []
    # sample positive pairs
    if not positive_indices:
        if (
            args.no_duplicates
            and min_pos
            >= num_unique_pos  # and args.positive_sampling_strategy != "hard"
        ) or args.positive_sampling_strategy == "all":
            # too few unique pairs so pos indices = all unique pairs
            positive_indices = torch.nonzero(torch.triu(mask_pos)).tolist()
        else:
            positive_indices = _get_pairs(
                distances=distances,
                labels=labels,
                args=args,
                mask=mask_pos,
                positive=True,
                rng=rng,
                return_triplets=False,
                emb_dim=emb_dim,
                positive_pairs=None,
            )

        logger.debug(f"Found {len(positive_indices)} positive pairs")
        # check if actually no duplicates (if that is what we want..)
        uni_pos_indices = get_unique_indices(positive_indices)
        if len(positive_indices) > len(uni_pos_indices):
            if args.no_duplicates:
                logger.warning(
                    f"Found {len(positive_indices)} positive pairs ({len(uni_pos_indices)} of them unique)"
                )
                positive_indices = uni_pos_indices
            logger.debug(
                f"Found {len(positive_indices)} positive pairs ({len(uni_pos_indices)} of them unique)"
            )

    # add random samples if less than min
    if len(positive_indices) < min_pos:
        logger.debug(
            f"Found fewer positive pairs than minimum: {len(positive_indices)} vs. {min_pos}"
        )
        diff = min_pos - len(positive_indices)
        mask_clone = mask_pos.clone()
        # do we care about no duplicates if we have too few pairs?
        if args.no_duplicates:
            # mask already picked samples
            for pair in positive_indices:
                mask_clone[pair[0]][pair[1]] = mask_clone[pair[1]][pair[0]] = False

        avail_indices = torch.nonzero(torch.triu(mask_clone)).tolist()
        n = min(diff, len(avail_indices)) if args.no_duplicates else diff
        random_pos_indices = rng.choice(
            avail_indices, size=n, replace=n > len(avail_indices)
        ).tolist()
        positive_indices += random_pos_indices

    # sample negative pairs
    # TODO: maybe we just always return triplets when mining negatives?
    # would make oversampling -> drawing up to max easier..
    # we then split it back into pairs..
    # only problem would be non unique positive pairs..
    
    if not negative_indices:
        negative_indices = _get_pairs(
            distances=distances,
            labels=labels,
            args=args,
            mask=mask_neg,
            positive=False,
            rng=n_rng,
            return_triplets=return_triplets,
            emb_dim=emb_dim,
            positive_pairs=None if args.independent_negatives else positive_indices,
        )

    logger.debug(f"Found {len(negative_indices)} negative pairs")

    # check if actually not duplicates
    uni_neg_indices = get_unique_indices(negative_indices)
    if len(negative_indices) > len(uni_neg_indices):
        if args.no_duplicates:
            logger.warning(
                f"Found {len(negative_indices)} negative pairs ({len(uni_neg_indices)} of them unique)"
            )
            negative_indices = uni_neg_indices
        logger.debug(
            f"Found {len(negative_indices)} negative pairs ({len(uni_neg_indices)} of them unique)"
        )

    # add random positive pairs again (in case semi/hard sampling removed some..)
    if len(positive_indices) < min_pos:
        logger.debug(
            f"Found fewer positive pairs than minimum: {len(positive_indices)} vs. {min_pos}"
        )
        diff = min_pos - len(positive_indices)
        mask_clone = mask_pos.clone()
        # do we care about no duplicates if we have too few pairs?
        if args.no_duplicates:
            # mask already picked samples
            for pair in positive_indices:
                mask_clone[pair[0]][pair[1]] = mask_clone[pair[1]][pair[0]] = False

        avail_indices = torch.nonzero(torch.triu(mask_clone)).tolist()
        n = min(diff, len(avail_indices)) if args.no_duplicates else diff
        random_pos_indices = rng.choice(
            avail_indices, size=n, replace=n > len(avail_indices)
        ).tolist()
        positive_indices += random_pos_indices

    # randomly pick some if we have more than max
    if max_pos > 0 and len(positive_indices) > max_pos:
        logger.debug(
            f"Found more positive pairs than maximum: {len(positive_indices)} vs. {max_pos}"
        )
        choices = rng.choice(positive_indices, size=max_pos, replace=False).tolist()
        positive_indices = choices

    # add random samples if less than min
    if len(negative_indices) < min_neg:
        logger.debug(
            f"Found fewer negative pairs than minimum: {len(negative_indices)} vs. {min_neg}"
        )
        # FIXME: fix for triplet selection..
        diff = min_neg - len(negative_indices)
        mask_clone = mask_neg.clone()

        if not return_triplets:
            if args.no_duplicates:
                # mask already picked samples
                for pair in negative_indices:
                    mask_clone[pair[0]][pair[1]] = mask_clone[pair[1]][pair[0]] = False

            avail_indices = torch.nonzero(torch.triu(mask_clone)).tolist()
            n = min(diff, len(avail_indices)) if args.no_duplicates else diff
            random_neg_indices = rng.choice(
                avail_indices, size=n, replace=n > len(avail_indices)
            ).tolist()
        else:
            # just fully random triplets for now..
            for _ in range(diff):
                random_anchor = int(rng.integers(num_samples))
                rand_pos_available = torch.squeeze(
                    torch.nonzero(mask_pos[random_anchor]), dim=-1
                ).tolist()
                rand_neg_available = torch.squeeze(
                    torch.nonzero(mask_neg[random_anchor]), dim=-1
                ).tolist()
                random_positive = int(rng.choice(rand_pos_available))
                random_negative = int(rng.choice(rand_neg_available))
                random_neg_indices.append(
                    (random_anchor, random_positive, random_negative)
                )

        negative_indices += random_neg_indices

    # randomly pick some if we have more than max
    if max_neg > 0 and len(negative_indices) > max_neg:
        logger.debug(
            f"Found more negative pairs than maximum: {len(negative_indices)} vs. {max_neg}"
        )
        choices = rng.choice(negative_indices, size=max_neg, replace=False).tolist()
        negative_indices = choices

    return positive_indices, negative_indices, random_pos_indices, random_neg_indices
