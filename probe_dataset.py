#!/usr/bin/env python
"""Report a MOABB dataset's true geometry before adding it to the registry.

Why this exists
---------------
``fbcsp_snn.pipeline._sfreq`` reads the sampling rate from
``DATASET_REGISTRY``, not from the data, and falls back to 250 Hz for an
unregistered dataset. A wrong entry therefore designs every Butterworth
band for the wrong rate -- an 8-14 Hz band would pass roughly 16-29 Hz at
512 Hz data -- and nothing raises. The registry values must be measured.

Usage
-----
::

    python probe_dataset.py --dataset BNCI2014_002 --subject 1

Prints the sampling rate, channel count, class labels, session structure and
trial counts, formatted as a registry entry ready to paste.
"""

from __future__ import annotations

import argparse
import logging
import warnings


def main() -> None:
    """Load one subject and report what the registry entry should say."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True,
                    help="MOABB dataset class name, e.g. BNCI2014_002")
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    for n in ("moabb", "mne"):
        logging.getLogger(n).setLevel(logging.ERROR)

    import moabb.datasets as mds
    from moabb.paradigms import MotorImagery

    cls = getattr(mds, args.dataset, None)
    if cls is None:
        raise SystemExit(f"moabb.datasets has no {args.dataset}")
    ds = cls()

    print(f"=== {args.dataset} ===")
    print(f"  subjects listed : {len(ds.subject_list)}  {ds.subject_list}")
    print(f"  event_id        : {getattr(ds, 'event_id', None)}")
    print(f"  interval        : {getattr(ds, 'interval', None)}")
    print(f"  paradigm        : {getattr(ds, 'paradigm', None)}")

    # Raw first: this is where the true sampling rate lives.
    raw_dict = ds.get_data(subjects=[args.subject])
    sess = raw_dict[args.subject]
    print(f"  sessions        : {list(sess)}")
    for s_name, runs in sess.items():
        print(f"    {s_name}: runs {list(runs)}")
    first_run = next(iter(next(iter(sess.values())).values()))
    sfreq = float(first_run.info["sfreq"])
    eeg = [c for c, t in zip(first_run.ch_names,
                             first_run.get_channel_types()) if t == "eeg"]
    print(f"  RAW sfreq       : {sfreq}")
    print(f"  RAW EEG channels: {len(eeg)}")

    # Then epoched, which is what the pipeline actually consumes.
    paradigm = MotorImagery(n_classes=len(ds.event_id))
    X, y, meta = paradigm.get_data(dataset=ds, subjects=[args.subject])
    classes = sorted(set(y))
    print(f"  EPOCHS X        : {X.shape}   (trials, channels, samples)")
    print(f"  classes         : {classes}")
    print(f"  trials/class    : {{{', '.join(f'{c}: {(y == c).sum()}' for c in classes)}}}")
    print(f"  window          : {X.shape[2] / sfreq:.3f} s at {sfreq} Hz")
    if "session" in meta:
        print(f"  meta sessions   : {sorted(set(meta['session']))}")

    print()
    print("Registry entry (verify before pasting):")
    print(f'    "{args.dataset}": {{')
    print(f'        "n_classes": {len(classes)},')
    print(f'        "sfreq": {sfreq:g},')
    print(f'        "n_channels": {X.shape[1]},')
    print(f'        "description": "{args.dataset} ({len(classes)}-class MI, '
          f'{len(ds.subject_list)} subjects)",')
    print(f'        "moabb_cls": "{args.dataset}",')
    print("    },")
    print()
    n_sessions = len(sess)
    if n_sessions >= 2:
        print(f"  -> {n_sessions} sessions: load_moabb will train on the first "
              "and test on the second (cross-session).")
    else:
        print("  -> single session: load_moabb applies a stratified 80/20 split "
              "(within-session).")


if __name__ == "__main__":
    main()
