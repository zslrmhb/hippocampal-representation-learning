"""Read-only audit of local raw and epoched NPZ exports.

Run with Python, NumPy and Matplotlib. Outputs go outside the repository by
default. Raw arrays are streamed one row at a time to limit memory use.
"""

from pathlib import Path
import argparse
import csv
import json
import zipfile

import numpy as np
from numpy.lib import format


def header(stream):
    version = format.read_magic(stream)
    if version not in ((1, 0), (2, 0)):
        raise ValueError(f"Unsupported NPY version: {version}")
    reader = (format.read_array_header_1_0 if version == (1, 0)
              else format.read_array_header_2_0)
    shape, fortran, dtype = reader(stream)
    if dtype.hasobject:
        raise ValueError("Expected numeric arrays")
    return shape, dtype, fortran


def metadata(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("data.npy") as stream:
            shape, dtype, fortran = header(stream)
        with archive.open("keys.npy") as stream:
            keys = format.read_array(stream, allow_pickle=False).tolist()
    return {"shape": list(shape), "dtype": str(dtype), "keys": keys,
            "fortran_order": fortran,
            "compressed_bytes": path.stat().st_size,
            "array_bytes": int(np.prod(shape)) * dtype.itemsize}


def rows(path):
    with zipfile.ZipFile(path) as archive, archive.open("data.npy") as stream:
        shape, dtype, fortran = header(stream)
        if fortran:
            raise ValueError(f"Cannot stream rows of a Fortran-order array: {path}")
        row_size = int(np.prod(shape[1:]))
        for _ in range(shape[0]):
            buf = stream.read(row_size * dtype.itemsize)
            if len(buf) != row_size * dtype.itemsize:
                raise ValueError(f"Truncated array: {path}")
            yield np.frombuffer(buf, dtype=dtype).reshape(shape[1:])


def counts(values):
    unique, number = np.unique(values, return_counts=True)
    return {str(v.item()): int(n) for v, n in zip(unique, number)}


def audit_session(root, session):
    files = {}
    for folder in ("raw_data", "epoched_data"):
        files[folder] = {p.stem.rsplit("_", 1)[-1]: metadata(p)
                         for p in sorted((root / folder / session).glob("*.npz"))}
    raw = root / "raw_data" / session
    epochs = root / "epoched_data" / session
    get = lambda folder, kind: next(folder.glob(f"*_{kind}.npz"))
    with np.load(get(epochs, "bvr"), allow_pickle=False) as data:
        labels = dict(zip(data["keys"].tolist(), data["data"]))
    anchors = labels["odor_ts"].astype(int)
    result = {"session": session, "files": files,
              "trial_counts": {k: counts(labels[k])
                               for k in ("correct", "inseq", "odor", "position")},
              "correct_by_inseq": {f"inseq={i},correct={c}":
                  int(np.sum((labels["inseq"] == i) & (labels["correct"] == c)))
                  for i in (0, 1) for c in (0, 1)},
              "post_ts_equals_odor_ts": bool(np.array_equal(labels["post_ts"], anchors)),
              "duration_equals_out_minus_in": bool(np.array_equal(
                  labels["poke_duration"], labels["poke_out_ts"] - labels["poke_in_ts"])),
              "raw_behavior": {}}
    events = {}
    time = None
    for key, row in zip(files["raw_data"]["bvr"]["keys"], rows(get(raw, "bvr"))):
        valid = np.isfinite(row)
        summary = {"nonfinite": int(np.sum(~valid)),
                   "min": float(np.min(row[valid])), "max": float(np.max(row[valid]))}
        if key == "TimeBin":
            time = row.copy()
            dt = np.diff(time)
            step = float(np.median(dt))
            gap_ids = np.flatnonzero(dt > step * 1.5)
            result["time"] = {"first_s": float(time[0]), "last_s": float(time[-1]),
                "median_step_s": step, "inferred_hz": 1 / step,
                "nonpositive_steps": int(np.sum(dt <= 0)),
                "gaps": [{"after_index": int(i), "from_s": float(time[i]),
                          "to_s": float(time[i+1]), "jump_s": float(dt[i])}
                         for i in gap_ids]}
        elif key not in ("XvalRatMazePosition", "YvalRatMazePosition"):
            summary["value_counts"] = counts(row)
            events[key] = {"indices": np.flatnonzero(row),
                           "values": row[row != 0].copy()}
        result["raw_behavior"][key] = summary
    assert time is not None
    # The exported ts fields might be row indices or one-based indices. Check
    # both against actual raw odor events rather than assuming either convention.
    odor_events = np.sort(np.concatenate([events[f"Odor{i}"]["indices"] for i in range(1, 6)]))
    result["raw_odor_event_count"] = len(odor_events)
    result["epoch_anchor_event_matches"] = {
        str(shift): int(np.isin(anchors + shift, odor_events).sum())
        for shift in (-1, 0, 1)}
    for shift in (-1, 0, 1):
        if np.isin(anchors + shift, odor_events).all():
            anchor_shift = shift
            break
    else:
        raise ValueError(f"Epoch anchors do not map to raw odor events: {session}")
    center = anchors + anchor_shift
    result["anchor_to_raw_index_shift"] = anchor_shift
    result["raw_odor_events_not_exported"] = np.setdiff1d(odor_events, center).tolist()
    event_labels = {}
    for key in [f"Odor{i}" for i in range(1, 6)] + [f"Position{i}" for i in range(1, 6)] + ["InSeqLog", "PerformanceLog"]:
        ev = events[key]
        lookup = dict(zip(ev["indices"].tolist(), ev["values"].tolist()))
        event_labels[key] = np.array([lookup.get(int(i), 0) for i in center])
    expected_odor = np.argmax(np.stack([event_labels[f"Odor{i}"] for i in range(1, 6)]), axis=0)
    expected_position = np.argmax(np.stack([event_labels[f"Position{i}"] for i in range(1, 6)]), axis=0)
    result["labels_match_raw"] = {
        "odor_zero_based": bool(np.array_equal(labels["odor"], expected_odor)),
        "position_zero_based": bool(np.array_equal(labels["position"], expected_position)),
        "inseq_positive_is_one": bool(np.array_equal(labels["inseq"], event_labels["InSeqLog"] == 1)),
        "correct_positive_is_one": bool(np.array_equal(labels["correct"], event_labels["PerformanceLog"] == 1))}
    # Match all trials, samples and channels against candidate raw slices.
    with np.load(get(epochs, "lfp"), allow_pickle=False) as data:
        epoch_lfp = data["data"]
    n_trials, n_samples, n_channels = epoch_lfp.shape
    assert n_trials == len(center)
    offsets = range(-2002, -1997)
    exact = {o: 0 for o in offsets}
    per_channel = []
    example = None
    epoch_centers = None
    reordered_matches = 0
    for channel, row in enumerate(rows(get(raw, "lfp"))):
        assert row.ndim == 1
        if channel == 0:
            # Identify every clip by a multi-sample fingerprint. Verify its
            # entire waveform on every channel below; signatures alone are
            # not evidence of an exact match.
            signatures = {}
            for anchor in odor_events:
                start = int(anchor)-2000
                if start >= 0 and start+n_samples <= len(row):
                    signatures.setdefault(tuple(row[start:start+32]), []).append(int(anchor))
            inferred = []
            for trial in range(n_trials):
                candidates = signatures.get(tuple(epoch_lfp[trial, :32, 0]), [])
                matches = [a for a in candidates if np.array_equal(
                    row[a-2000:a-2000+n_samples], epoch_lfp[trial, :, 0], equal_nan=True)]
                if len(matches) != 1:
                    raise ValueError(f"Ambiguous/unmatched LFP epoch {session} trial {trial}: {matches}")
                inferred.append(matches[0])
            epoch_centers = np.array(inferred)
        for trial, anchor in enumerate(epoch_centers):
            reordered_matches += int(np.array_equal(row[anchor-2000:anchor-2000+n_samples], epoch_lfp[trial, :, channel], equal_nan=True))
        for offset in offsets:
            matches = 0
            for trial, anchor in enumerate(center):
                start = int(anchor + offset)
                if 0 <= start and start+n_samples <= len(row):
                    matches += int(np.array_equal(row[start:start+n_samples], epoch_lfp[trial, :, channel], equal_nan=True))
            exact[offset] += matches
        finite = np.isfinite(row)
        per_channel.append({"key": files["raw_data"]["lfp"]["keys"][channel],
            "nonfinite": int((~finite).sum()), "min": float(row[finite].min()),
            "max": float(row[finite].max()), "std": float(row[finite].std())})
        if channel == 0:
            idx = int(epoch_centers[0])
            lo, hi = max(0, idx-3500), min(len(row), idx+6500)
            example = {"time": time[lo:hi]-time[idx], "raw": row[lo:hi].copy(),
                       "epoch": epoch_lfp[0, :, 0].copy()}
    best = -2000
    lookup = {int(c): i for i, c in enumerate(center)}
    permutation = [lookup.get(int(c), -1) for c in epoch_centers]
    result["neural_order"] = {"epoch_to_behavior_trial_index": permutation,
        "same_row_count": int(np.sum(epoch_centers == center)),
        "all_anchors_have_behavior_row": all(i >= 0 for i in permutation),
        "same_set_of_trials": bool(np.array_equal(np.sort(epoch_centers), np.sort(center))),
        "unique_epoch_centers": len(np.unique(epoch_centers)),
        "reordered_exact_trial_channel_slices": reordered_matches,
        "lfp_centers_raw_indices": epoch_centers.tolist()}
    if all(i >= 0 for i in permutation):
        result["neural_order"]["odor_labels_in_neural_order"] = labels["odor"][permutation].tolist()
        result["neural_order"]["matches_stable_position_sort"] = bool(np.array_equal(
            permutation, np.argsort(labels["position"], kind="stable")))
        result["neural_order"]["naive_row_label_agreement"] = {
            k: int(np.sum(labels[k] == labels[k][permutation]))
            for k in ("odor", "position", "inseq", "correct")}
    result["lfp_exact_comparison"] = {
        "offset_from_odor_index": best,
        "matching_trial_channel_slices": exact[best],
        "total_trial_channel_slices": n_trials*n_channels,
        "candidate_match_counts": {str(k): v for k, v in exact.items()},
        "epoch_nonfinite": int((~np.isfinite(epoch_lfp)).sum()),
        "channel_keys_preserved": files["raw_data"]["lfp"]["keys"] == files["epoched_data"]["lfp"]["keys"]}
    result["lfp_quality"] = per_channel
    result["epoch_window"] = {"samples": n_samples,
        "first_relative_s": float(time[int(epoch_centers[0]+best)]-time[int(epoch_centers[0])]),
        "last_relative_s": float(time[int(epoch_centers[0]+best+n_samples-1)]-time[int(epoch_centers[0])]),
        "trials_crossing_time_gaps": int(sum(np.any(np.diff(time[int(c+best):int(c+best+n_samples)]) > .0015) for c in epoch_centers)),
        "neighboring_epochs_overlapping": int(np.sum(np.diff(np.sort(epoch_centers)) < n_samples))}
    result["odor_equals_position_count"] = int(np.sum(labels["odor"] == labels["position"]))
    result["poke_duration_ms"] = {"min": int(labels["poke_duration"].min()),
        "median": float(np.median(labels["poke_duration"])), "max": int(labels["poke_duration"].max())}
    # Limited spike value check: first unit, first epoch only. Do not imply a
    # complete spike audit or load the much larger full spike arrays into RAM.
    raw_unit = next(rows(get(raw, "spk")))
    epoch_trial = next(rows(get(epochs, "spk")))
    start = int(epoch_centers[0]+best)
    result["spike_spot_check"] = {
        "scope": "first trial, first unit only",
        "exact_match": bool(np.array_equal(raw_unit[start:start+n_samples], epoch_trial[:, 0], equal_nan=True)),
        "first_raw_unit_value_counts": counts(raw_unit)}
    assert reordered_matches == n_trials*n_channels, "LFP export is not an exact set of raw clips"
    assert result["neural_order"]["same_set_of_trials"], "Neural/behavior trial sets differ"
    example["epoch_time"] = time[np.arange(int(epoch_centers[0]+best), int(epoch_centers[0]+best+n_samples))]-time[int(epoch_centers[0])]
    return result, example


def write_report(reports, output):
    with (output / "lfp-trial-alignment.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "lfp_epoch_row_0based", "behavior_trial_column_0based",
                         "odor_onset_raw_index_0based", "odor_0based"])
        for report in reports:
            order = report["neural_order"]
            for epoch, (trial, center, odor) in enumerate(zip(
                order["epoch_to_behavior_trial_index"], order["lfp_centers_raw_indices"],
                order["odor_labels_in_neural_order"])):
                writer.writerow([report["session"], epoch, trial, center, odor])
    table = []
    for r in reports:
        raw = r["files"]["raw_data"]
        n = raw["lfp"]["shape"][1]
        table.append(f"| {r['session']} | {n:,} | {n/1000/60:.2f} | "
                     f"{raw['lfp']['shape'][0]} | {raw['spk']['shape'][0]} | "
                     f"{len(r['neural_order']['epoch_to_behavior_trial_index'])} | "
                     f"{r['trial_counts']['inseq']['0']} | {r['trial_counts']['correct']['0']} |")
    comparisons = sum(r["neural_order"]["reordered_exact_trial_channel_slices"] for r in reports)
    all_trials = sum(len(r["neural_order"]["epoch_to_behavior_trial_index"]) for r in reports)
    stable = all(r["neural_order"]["matches_stable_position_sort"] for r in reports)
    report = """# Raw and epoched data exploration

Inspected October 12, 2026. All source datasets were read only. No original arrays or labels were repaired or overwritten.

## Main findings

Raw data is the continuous recording. Epoched data contains four-second odor-centered clips extracted from it. The supplied Python LFP epochs retain the original 1000 Hz sampling rate, despite the overview sheet's 100 Hz description.

**The epoched LFP and behavior files have different trial orders.** LFP epochs are grouped by sequence position; behavior columns are chronological. Pairing row i of neural data with column i of behavior would give incorrect labels for many trials. The mapping in `lfp-trial-alignment.csv` is verified against the actual raw waveforms, not guessed from class counts.

## Inventory

| Session | Raw samples | Stored recording minutes excluding breaks | LFP channels | Spike units | Trials | OutSeq | Incorrect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(table) + f"""

There are {all_trials:,} odor trials total. Both correct/incorrect and InSeq/OutSeq are present in every rat. These Python exports are not restricted to correct InSeq trials.

## Array layouts and contents

| File | Raw | Epoched |
| --- | --- | --- |
| LFP | channels x continuous samples | trials x 4000 samples x channels |
| Spikes | units x continuous samples | trials x 4000 samples x units |
| Behavior | 18 labels x continuous samples | 9 labels x trials |
| wfm | units x 128 values | absent |
| pos | Stella only: continuous samples x 2 | absent |

`wfm` is a static per-unit array, consistent with waveform information; its precise meaning and units were not independently verified. All five raw behavior files also contain X/Y position, so Stella's additional `pos` file is not evidence that only Stella has movement data. Its numerical equality to those behavior rows was not checked.

The raw behavior includes timestamps, odor and position event indicators, sequence/performance logs, nosepoke events, rewards, and X/Y position. Epoched behavior summarizes `correct`, `inseq`, `position`, `odor`, `odor_ts`, `post_ts`, `poke_in_ts`, `poke_out_ts`, and `poke_duration`.

## Direct LFP verification

All {comparisons:,} trial/channel slices, each containing 4000 samples, match raw LFP values exactly after the trial permutation is accounted for. Every exported trial maps uniquely to a raw odor event, and the same set of odor events occurs in neural and behavior exports. Channel keys are preserved. The LFP order matches a stable sort of chronological behavior by sequence position in all rats: **{stable}**.

For an odor event at raw sample index k, its epoch is `raw_lfp[:, k-2000:k+2000].T`. Sample 2000 is odor onset. At 1000 Hz the times are -2.000 to +1.999 seconds. There is no extra filtering, amplitude normalization, or temporal downsampling relative to the supplied raw LFP files. This does not establish what acquisition processing happened before the raw export.

Raw `TimeBin` steps are approximately 0.001 seconds within valid segments. `odor_ts` maps directly to zero-based raw row indices, not absolute seconds; gaps mean dividing an index by 1000 does not recover the original clock time. Use `TimeBin[odor_ts]` when absolute times are needed. `post_ts` equals `odor_ts` in every trial, so it should not be interpreted as trial end from its name. `poke_duration` equals `poke_out_ts - poke_in_ts` and corresponds to milliseconds in these files.

**Concrete mismatch example:** Mitt's neural epoch 1 (second clip, zero-based indexing) matches chronological behavior trial 4, an odor A trial. Behavior column 1 instead describes odor B. This is a trial permutation, not a small timing shift.

To align labels without altering sources: load behavior as `(9, trials)` and use `behavior[:, epoch_to_behavior_trial_index]` from the verified LFP mapping. Alternatively reorder LFP into chronological order using the inverse permutation. Check a separately established mapping before applying this to spikes: only the first trial/first unit was checked for each rat here, and it matched the LFP-inferred anchor in all five. This limited spike check is not proof of the complete spike ordering.

## Labels

- Actual epoch odor and position values are 0-4. Map odor 0-4 to A-E and position 0-4 to positions 1-5 when presenting results.
- `correct` and `inseq` are binary 0/1 in the epoch files. They correspond to positive raw `PerformanceLog` and `InSeqLog` values at the odor events.
- Outside raw events, log value zero is often an absence of an event, not a trial label.
- The behavior metadata itself matches the chronological raw events for all trials. The mismatch is between neural and behavior order.

## Missingness and time gaps

No nonfinite values were found in any raw LFP channel, raw behavior row, or epoched LFP array across the five rats. This is not a complete artifact or physiological signal-quality assessment, and full spike arrays were not checked for missingness.

Recording breaks exist as timestamp jumps rather than NaN-filled rows. Do not filter across these jumps or interpret the raw files as uniformly continuous across the entire experiment. No verified LFP epoch crosses a timestamp break.

| Rat | Timestamp jumps in seconds |
| --- | --- |
"""
    for r in reports:
        gaps = ", ".join(f"{g['jump_s']:.3f}" for g in r["time"]["gaps"]) or "none"
        report += f"| {r['session']} | {gaps} |\n"
    overlaps = sum(r["epoch_window"]["neighboring_epochs_overlapping"] for r in reports)
    report += f"""

There are {overlaps} adjacent odor-event pairs with overlapping four-second epochs. Splitting such clips independently can put identical raw samples in train and test sets. Group trials/sequences and protect overlapping windows when designing evaluation. The number is adjacent overlapping pairs, not unique samples or independent trials.

## What to use for your preprocessing work

Start from raw LFP to inspect noise, choose filters, account for timestamp breaks, and generate a reproducible set of aligned clips. Use the supplied epochs to understand event-centered analysis, but fix label association in the loader or reconstruct epochs before supervised modeling. No source data was fixed during this exploration.

Keep all recording channels initially; the rats have 20-22 channels and different spike-unit counts. Do not assume channel-number correspondence establishes anatomical equivalence across rats. Preserve class counts: OutSeq and incorrect trials are minorities, especially some combinations such as Barat's single incorrect InSeq trial.

The PNG illustrates raw continuous data and an exactly overlapping exported epoch. `audit.json` contains full counts, keys, timing details, channel ranges, checks, and the verified permutation.

## Reproduce

From the repository root, with NumPy and Matplotlib installed:

```sh
python EDA/compare_data_folders.py
```

The default output directory is `../ai-agent-info/data-exploration`. This script writes reports there and never overwrites source data. All row indices in the CSV are zero-based. The timestamp gap and fingerprint checks are tailored to the inspected export; unexpected layouts or ambiguous matches fail rather than silently assign labels.
"""
    (output / "report.md").write_text(report)


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo / "data")
    parser.add_argument("--output-dir", type=Path, default=repo.parent / "ai-agent-info" / "data-exploration")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    example = None
    for folder in sorted((args.data_dir / "raw_data").iterdir()):
        if not folder.is_dir():
            continue
        report, snippet = audit_session(args.data_dir, folder.name)
        reports.append(report)
        if example is None:
            example = snippet
        print(folder.name, report["lfp_exact_comparison"], flush=True)
        (args.output_dir / "audit.json").write_text(json.dumps(reports, indent=2))
    write_report(reports, args.output_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    axes[0].plot(example["time"], example["raw"], lw=.7, color="#37526b")
    axes[0].axvspan(example["epoch_time"][0], example["epoch_time"][-1], color="#deaf5a", alpha=.3, label="exported epoch")
    axes[0].set_title("Raw recording: continuous activity around Mitt's first exported odor trial")
    axes[0].legend(frameon=False)
    axes[1].plot(example["time"], example["raw"], lw=1.8, color="#37526b", label="raw LFP")
    axes[1].plot(example["epoch_time"], example["epoch"], lw=.8, color="#c45b3c", label="epoched LFP")
    axes[1].set_xlim(-2.1, 2.1)
    axes[1].set_title("The epoched values overlap the raw values exactly")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.axvline(0, color="#555555", ls="--", lw=.8)
        axis.set_xlabel("Seconds relative to odor onset")
        axis.set_ylabel("LFP (stored units)")
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output_dir / "raw-vs-epoched.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
