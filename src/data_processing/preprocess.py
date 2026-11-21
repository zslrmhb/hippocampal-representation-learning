"""Matched-window preprocessing on raw project NPZ recordings (samples at 1 kHz).

Public functions discover session pairs, prepare an atomic disk cache, and load
trial × time × feature arrays. Stored data and decoder labels both use 1..5.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve, sosfiltfilt

from .vendor.events import load_behavior
from .vendor.filtering import artifact_mask, notch_design, runs
from .vendor.io import metadata, rows
from .vendor.quality import Config, partition_trials, robust_stats


FREQUENCIES = np.geomspace(4, 150, 12)
WINDOWS = tuple((s, d) for d in (100, 200, 500) for s in range(0, 501 - d, 100))
CONFIG = Config(profile="inseq-decoding", include_outseq=True)
SPLITS = ("train", "val", "test")
META = [
    "trial_id",
    "onset",
    "odor",
    "position",
    "inseq",
    "correct",
    "split",
    "segment",
    "poke_out",
    "sampling_duration_ms",
    "filter_block_id",
]
WAVELET_CHUNK = 8000


def discover(root: Path) -> dict[str, tuple[Path, Path]]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Raw-data directory does not exist: {root}")
    pairs = {}
    missing = []
    for file in sorted(root.rglob("*_lfp.npz")):
        session = file.name.removesuffix("_lfp.npz")
        behavior = file.with_name(f"{session}_bvr.npz")
        if not behavior.is_file():
            missing.append(str(behavior))
        elif session in pairs:
            raise ValueError(
                f"Duplicate raw session {session}: {pairs[session][0]} and {file}"
            )
        else:
            pairs[session] = (file, behavior)
    if missing:
        raise FileNotFoundError("Missing matching behavior NPZ: " + ", ".join(missing))
    if not pairs:
        raise FileNotFoundError(
            f"No paired *_lfp.npz and *_bvr.npz recordings under {root}"
        )
    return pairs


def _json(path, record):
    path.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(paths):
    package = Path(__file__).resolve().parent
    return {
        "version": 1,
        "sources": {Path(p).name: _digest(p) for p in paths},
        "implementation": {
            p.name: _digest(p)
            for p in [
                Path(__file__).resolve(),
                *sorted((package / "vendor").glob("*.py")),
            ]
        },
        "config": dataclasses.asdict(CONFIG),
        "centers_hz": FREQUENCIES.tolist(),
    }


def _wavelets():
    kernels = []
    for hz in FREQUENCIES:
        # A finite, zero-mean complex Morlet with four cycles and unit energy.
        sigma = 4 / (2 * np.pi * hz)
        radius = int(np.ceil(3 * sigma * CONFIG.sample_rate_hz))
        t = np.arange(-radius, radius + 1) / CONFIG.sample_rate_hz
        envelope = np.exp(-0.5 * (t / sigma) ** 2)
        carrier = np.exp(2j * np.pi * hz * t)
        kernel = envelope * (carrier - np.sum(envelope * carrier) / np.sum(envelope))
        kernels.append((kernel / np.linalg.norm(kernel)).astype(np.complex64))
    return kernels


def _target_trials(trials, target):
    return [
        t
        for t in trials
        if not t["reasons"] and t["correct"] and (target == "position" or t["inseq"])
    ]


def _usable_raw(t, intervals):
    # Behavior reconstruction validates the poke. Include all samples up to
    # 500 ms, limited by exit and source ownership when fitting raw artifact QC.
    return (
        min(t["onset"] + 500, t["poke_out"], intervals[(t["split"], t["segment"])])
        - t["onset"]
    )


def _training_thresholds(lfp, candidates, ends, channels):
    chunks = [
        np.arange(t["onset"], t["onset"] + n)
        for t in candidates
        if t["split"] == "train" and (n := _usable_raw(t, ends)) > 0
    ]
    if not chunks:
        return None
    support = np.unique(np.concatenate(chunks))
    adjacent = np.diff(support) == 1
    stats = {
        k: np.zeros(channels)
        for k in ("robust_center", "amplitude_limit", "jump_center", "jump_limit")
    }
    stats["channel_mask"] = np.zeros(channels, dtype=bool)
    for ch, row in enumerate(rows(lfp)):
        sample = row[support].astype(np.float64)
        center, scale = robust_stats(sample)
        jcenter, jscale = robust_stats(np.diff(sample)[adjacent])
        stats["channel_mask"][ch] = scale > 0
        stats["robust_center"][ch] = center
        stats["amplitude_limit"][ch] = scale * CONFIG.artifact_z
        stats["jump_center"][ch] = jcenter
        stats["jump_limit"][ch] = jscale * CONFIG.artifact_z
    stats["unique_training_samples"] = len(support)
    return stats


def _clean_blocks(lfp, intervals, stats, guard):
    invalid, totals = artifact_mask(lfp, intervals, stats, CONFIG)
    blocks = []
    for interval in intervals:
        lo, hi = interval["start"], interval["end"]
        for a, b in runs(~invalid[lo:hi]):
            if b - a >= 2 * guard + 100:
                blocks.append(
                    dict(
                        start=int(lo + a),
                        end=int(lo + b),
                        split=interval["split"],
                        segment=interval["segment"],
                        block_id=len(blocks),
                    )
                )
    return blocks, totals


def _membership(trial, start, duration, ends, lookup, guard):
    a = trial["onset"] + start
    b = a + duration
    owner = (trial["split"], trial["segment"])
    if b > min(trial["poke_out"], ends[owner]):
        return None
    group = lookup.get(owner)
    if group is None:
        return None
    starts, blocks = group
    i = np.searchsorted(starts, a, side="right") - 1
    if i >= 0 and a >= blocks[i]["start"] + guard and b <= blocks[i]["end"] - guard:
        return blocks[i]["block_id"]
    return None


def _matched_and_coverage(candidates, blocks, ends, guard, target, session):
    by_owner = {}
    for block in blocks:
        by_owner.setdefault((block["split"], block["segment"]), []).append(block)
    lookup = {k: (np.array([b["start"] for b in v]), v) for k, v in by_owner.items()}
    matched = []
    eligible = {}
    for s, d in WINDOWS:
        members = []
        for trial in candidates:
            block_id = _membership(trial, s, d, ends, lookup, guard)
            if block_id is not None:
                members.append(trial)
                if (s, d) == (0, 500):
                    matched.append(dict(trial, filter_block_id=block_id))
        eligible[s, d] = members
    counts = []
    for (s, d), members in eligible.items():

        def tally(cohort):
            result = {}
            for t in cohort:
                key = (t["split"], t[target], t["inseq"])
                result[key] = result.get(key, 0) + 1
            return result

        all_counts, matched_counts = tally(members), tally(matched)
        for split in SPLITS:
            for label in range(1, 6):
                for status in (0, 1):
                    key = (split, label, status)
                    counts.append(
                        dict(
                            session=session,
                            target=target,
                            start=s,
                            duration=d,
                            split=split,
                            class_label=label,
                            inseq=status,
                            n_eligible=all_counts.get(key, 0),
                            n_matched=matched_counts.get(key, 0),
                        )
                    )
    return matched, counts


def _arrays(folder, matched, n_ch):
    base = (len(matched), 500)
    result = {}
    for name, features in (
        ("raw", n_ch),
        ("amp", n_ch * len(FREQUENCIES)),
        ("phase", n_ch * len(FREQUENCIES) * 2),
    ):
        path = folder / f"{name}.npy"
        if len(matched) == 0:
            np.save(path, np.empty((*base, features), dtype=np.float32))
        else:
            result[name] = np.lib.format.open_memmap(
                path, mode="w+", dtype="float32", shape=(*base, features)
            )
    return result


def _chunk_groups(indices, matched):
    group = []
    for i in indices:
        if (
            group
            and matched[i]["onset"] + 500 - matched[group[0]]["onset"] > WAVELET_CHUNK
        ):
            yield group
            group = []
        group.append(i)
    if group:
        yield group


def _extract_channel(
    arrays, channel, row, grouped, blocks, matched, kernels, sos, notch_guard
):
    freq_count = len(kernels)
    for block_id, indices in grouped.items():
        block = blocks[block_id]
        lo, hi = block["start"], block["end"]
        source = row[lo:hi]
        voltage = (
            source if sos is None
            else sosfiltfilt(sos, source, padtype="odd", padlen=notch_guard)
        )
        if "raw" in arrays:
            for i in indices:
                rel = matched[i]["onset"] - lo
                arrays["raw"][i, :, channel] = voltage[rel : rel + 500]
        for group in _chunk_groups(indices, matched):
            first = matched[group[0]]["onset"] - lo
            last = matched[group[-1]]["onset"] + 500 - lo
            for b, kernel in enumerate(kernels):
                radius = len(kernel) // 2
                source = voltage[first - radius : last + radius].astype(np.float32)
                wave = fftconvolve(source, kernel, mode="valid")
                feature = channel * freq_count + b
                for i in group:
                    rel = matched[i]["onset"] - lo - first
                    z = wave[rel : rel + 500]
                    amplitude = np.abs(z)
                    arrays["amp"][i, :, feature] = np.log(
                        np.maximum(amplitude, np.finfo(np.float32).tiny)
                    )
                    if "phase" in arrays:
                        angle = np.angle(z)
                        arrays["phase"][i, :, 2 * feature] = np.sin(angle)
                        arrays["phase"][i, :, 2 * feature + 1] = np.cos(angle)


def _prepare_target(folder, session, target, lfp, lfp_meta, trials, intervals):
    folder.mkdir()
    ends = {(i["split"], i["segment"]): i["end"] for i in intervals}
    candidates = _target_trials(trials, target)
    stats = _training_thresholds(lfp, candidates, ends, lfp_meta["shape"][0])
    kernels = _wavelets()
    sos, info = notch_design(CONFIG)
    notch_guard = info["guard_samples"]
    guard = notch_guard + max(len(k) // 2 for k in kernels)
    blocks, totals = [], {}
    channels = []
    if stats is not None:
        channels = np.flatnonzero(stats["channel_mask"]).tolist()
        np.savez_compressed(folder / "raw_qc.npz", **stats)
        if channels:
            blocks, totals = _clean_blocks(lfp, intervals, stats, guard)
    matched, coverage = _matched_and_coverage(
        candidates, blocks, ends, guard, target, session
    )
    frame = pd.DataFrame([{k: t[k] for k in META} for t in matched], columns=META)
    frame.to_csv(folder / "matched_trials.csv", index=False)
    arrays = _arrays(folder, matched, len(channels))
    if matched:
        grouped = {}
        for i, t in enumerate(matched):
            grouped.setdefault(t["filter_block_id"], []).append(i)
        for c, row in enumerate(rows(lfp)):
            if c in channels:
                _extract_channel(
                    arrays,
                    channels.index(c),
                    row,
                    grouped,
                    blocks,
                    matched,
                    kernels,
                    sos,
                    notch_guard,
                )
        for array in arrays.values():
            array.flush()
    _json(
        folder / "qa.json",
        dict(
            target=target,
            frequencies_hz=FREQUENCIES.tolist(),
            notch_guard_samples=notch_guard,
            morlet_guard_samples=guard - notch_guard,
            matching="all targets use correct trials; within target all windows use retained 0-500 ms trial IDs",
            signal_policy="Offline symmetric notch + 4-cycle Morlet with finite +/-3-sigma support. Filtering can see outside odor windows within its guarded split/segment/clean block.",
            channel_keys=[lfp_meta["keys"][c] for c in channels],
            raw_artifact_counts=totals,
            clean_blocks=blocks,
            n_matched=len(matched),
            n_behavior_eligible=len(candidates),
            amp_feature_order="channel-major, then ascending frequency",
            phase_feature_order="channel-major, frequency, then sin and cos",
            standardization="each target/window, unique matched training samples, population mean/SD; constants set to zero",
        ),
    )
    return coverage


def _inventory(folder):
    """Record sizes of every published data artifact and the small provenance files."""
    return {
        item.relative_to(folder).as_posix(): item.stat().st_size
        for item in folder.rglob("*")
        if item.is_file() and item != folder / "artifacts.json"
    }


def _recorded_identity(folder, expected):
    # The second copy allows recovery if either identity file was interrupted
    # or truncated on disk. An intact mismatch always needs a new namespace.
    observed = []
    for name in ("signature.json", "artifacts.json"):
        try:
            record = json.loads((folder / name).read_text(encoding="utf-8"))
            value = record if name == "signature.json" else record["signature"]
            if (
                isinstance(value, dict)
                and "sources" in value
                and "implementation" in value
            ):
                observed.append(value)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if not observed:
        raise ValueError(
            f"Cannot verify the identity of {folder}. Choose a fresh output directory and preserve the old cache."
        )
    if any(item != expected for item in observed):
        raise ValueError(
            f"Input or preprocessing changed for {folder.name}. Start a new output directory."
        )


def _verify_cache(base):
    record = json.loads((base / "artifacts.json").read_text(encoding="utf-8"))
    expected = record["sizes"]
    if not isinstance(expected, dict) or expected != _inventory(base):
        raise ValueError("a cache artifact is missing, truncated, or unexpected")
    required = {"coverage.csv", "session.json", "event_manifest.csv", "signature.json"}
    if not required.issubset(expected):
        raise ValueError("incomplete cache inventory")
    for target in ("odor", "position"):
        directory = base / target
        qa = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(directory / "matched_trials.csv")
        n_trials = len(frame)
        if (
            list(frame.columns) != META
            or not frame.trial_id.is_unique
            or n_trials != qa["n_matched"]
        ):
            raise ValueError(f"{target} trial metadata disagrees with preprocessing QA")
        channels = len(qa["channel_keys"])
        for name, n_features in (
            ("raw", channels),
            ("amp", channels * len(FREQUENCIES)),
            ("phase", channels * len(FREQUENCIES) * 2),
        ):
            # Read headers and compare the declared payload size without paging
            # multi-gigabyte feature arrays through RAM on every resume.
            path = directory / f"{name}.npy"
            with path.open("rb") as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported array file version: {version}")
                header_end = stream.tell()
            if (
                shape != (n_trials, 500, n_features)
                or fortran
                or dtype != np.float32
                or path.stat().st_size
                != header_end + n_trials * 500 * n_features * dtype.itemsize
            ):
                raise ValueError(f"Inconsistent feature array: {path}")


def _publish_coverage(base, session, output):
    public = Path(output) / "coverage"
    public.mkdir(parents=True, exist_ok=True)
    temp = public / f".{session}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copyfile(base / "coverage.csv", temp)
        os.replace(temp, public / f"{session}.csv")
    finally:
        temp.unlink(missing_ok=True)


def prepare(session, paths, output: Path):
    """Build one session. Commit by atomic directory rename; recover a partial write by rerunning."""
    lfp, behavior = map(Path, paths)
    parent = Path(output) / "preprocessing"
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / session
    signature = _fingerprint((lfp, behavior))
    # The study supervisor holds the exclusive output lock before prepare().
    # Dedicated namespaces ensure cleanup can never select a user recording.
    temp = parent / ".work" / session
    rejected = parent / ".quarantine" / session
    if temp.exists():
        shutil.rmtree(temp)
    if dest.exists():
        _recorded_identity(dest, signature)
        try:
            _verify_cache(dest)
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            print(f"[preprocess] Rebuilding damaged {session} cache: {exc}", flush=True)
            rejected.parent.mkdir(exist_ok=True)
            if rejected.exists():
                shutil.rmtree(rejected)
            os.replace(dest, rejected)
        else:
            _publish_coverage(dest, session, output)
            if rejected.exists():
                shutil.rmtree(rejected)
            return
    elif rejected.exists():
        # Prior recovery may have stopped after quarantine but before commit.
        _recorded_identity(rejected, signature)
    temp.mkdir(parents=True)
    try:
        time, trials, segments, anomalies, _, _ = load_behavior(
            behavior, CONFIG.sample_rate_hz
        )
        info = metadata(lfp)
        if info["shape"][1] != len(time):
            raise ValueError(
                f"{session}: LFP and behavior have different sample counts"
            )
        # Ask the original package to partition by RAW timeline. Do not impose
        # eligibility of its 500-ms example on candidate windows or coverage.
        cloned = [dict(t, reasons=list(t["reasons"])) for t in trials]
        intervals = partition_trials(cloned, segments, len(time), CONFIG)
        for iv in intervals:
            if iv["split"] == "validation":
                iv["split"] = "val"
        for t, owned in zip(trials, cloned):
            t["split"] = "val" if owned["split"] == "validation" else owned["split"]
            for key in ("odor", "position"):
                t[key] = t[key] + 1 if t[key] >= 0 else None
        pd.DataFrame(trials).to_csv(temp / "event_manifest.csv", index=False)
        _json(
            temp / "session.json",
            dict(
                session=session,
                sample_rate=CONFIG.sample_rate_hz,
                original_channels=info["keys"],
                intervals=intervals,
                anomalies=anomalies,
                source_lfp=str(lfp),
                source_behavior=str(behavior),
                matched_window_ms=[0, 500],
            ),
        )
        all_counts = []
        for target in ("odor", "position"):
            print(
                f"[preprocess] {session} {target}: raw quality, notch and narrowband features",
                flush=True,
            )
            all_counts.extend(
                _prepare_target(
                    temp / target, session, target, lfp, info, trials, intervals
                )
            )
        pd.DataFrame(all_counts).to_csv(temp / "coverage.csv", index=False)
        _json(temp / "signature.json", signature)
        _json(
            temp / "artifacts.json", {"signature": signature, "sizes": _inventory(temp)}
        )
        _verify_cache(temp)
        os.replace(temp, dest)
        _publish_coverage(dest, session, output)
        if rejected.exists():
            shutil.rmtree(rejected)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def _window_scale(x, frame, start, duration, *, parameters=None, fitted=None):
    width = x.shape[-1]
    if parameters is None:
        train = np.flatnonzero(frame["split"].to_numpy() == "train")
        if not len(train) or not width:
            # Never use held-out data to normalize a cohort without training.
            return np.zeros(x.shape, dtype=np.float32)
        train_indices = (
            frame.iloc[train]["onset"].to_numpy()[:, None]
            + start
            + np.arange(duration)[None, :]
        )
        _, first = np.unique(train_indices.ravel(), return_index=True)
        trial_indices = train[first // duration]
        sample_indices = first % duration
        if fitted is not None:
            fitted["mean"], fitted["sd"] = np.empty(width), np.empty(width)
    elif (any(parameters[k].shape != (width,) or not np.isfinite(parameters[k]).all()
              for k in ("mean", "sd")) or np.any(parameters["sd"] < 0)):
        raise ValueError("Invalid saved window scaling parameters")
    result = np.empty(x.shape, dtype=np.float32)
    for lo in range(0, x.shape[-1], 32):
        hi = min(lo + 32, x.shape[-1])
        if parameters is None:
            fit = np.asarray(x[trial_indices, sample_indices, lo:hi], dtype=np.float64)
            mean, sd = fit.mean(axis=0), fit.std(axis=0)
            if fitted is not None:
                fitted["mean"][lo:hi], fitted["sd"][lo:hi] = mean, sd
        else:
            mean, sd = (parameters[k][lo:hi] for k in ("mean", "sd"))
        active = sd > 0
        result[:, :, lo:hi] = (
            (x[:, :, lo:hi] - mean) / np.where(active, sd, 1)
        ) * active
    return result


def load_data(
    output: Path, session: str, target: str, start: int, duration: int, rep: str,
    *, splits=None, amplitude_parameters=None, fitted_amplitude=None,
):
    """Return trial × time × feature data; optional splits must include train for scaling."""
    if target not in ("odor", "position") or (start, duration) not in WINDOWS:
        raise ValueError(f"Unknown target/window: {target}, {start}, {duration}")
    if rep not in ("raw", "amp", "phase", "combined"):
        raise ValueError(f"Unknown input representation: {rep}")
    base = Path(output) / "preprocessing" / session / target
    frame = pd.read_csv(
        base / "matched_trials.csv",
        dtype={"trial_id": int, "odor": int, "position": int},
    )
    n_trials = len(frame)
    subset = slice(None)
    if splits is not None:
        if "train" not in splits or not set(splits).issubset(SPLITS):
            raise ValueError("Requested splits must include train and be known")
        indices = np.flatnonzero(frame.split.isin(splits))
        if len(indices) and indices[-1] - indices[0] + 1 == len(indices):
            subset = slice(indices[0], indices[-1] + 1)
        else:
            subset = indices
        frame = frame.iloc[subset].reset_index(drop=True)
    if rep == "combined":
        inputs = ["amp", "phase"]
    else:
        inputs = [rep]
    features = []
    for name in inputs:
        filename = base / f"{name}.npy"
        a = np.load(
            filename, mmap_mode=("r" if n_trials else None), allow_pickle=False
        )
        x = a[subset, start : start + duration]
        features.append(
            x if name == "phase" else _window_scale(
                x, frame, start, duration,
                parameters=amplitude_parameters if name == "amp" else None,
                fitted=fitted_amplitude if name == "amp" else None,
            )
        )
    if len(features) == 1:
        return features[0], frame
    return np.concatenate(features, axis=-1), frame
