# Data handling: why the feed files are encrypted

**Summary.** On 2026-09-23 we stopped publishing phishing and malware URLs from
live threat feeds. Every file that contains them is now committed encrypted, and
this repository's git history was rewritten so that no earlier commit contains
them in plain text either. This page records what was done, when, and why, so
the change isn't mistaken for something hidden or accidental.

## Why

The live half of this study collects URLs from two feeds:

- **OpenPhish community feed** (phishing URLs). Its [terms of use](https://openphish.com/terms.html)
  allow research use, but state: *"you agree not to license, sell, rent, lease,
  transfer, assign, distribute, display, disclose, create derivative works or
  otherwise make all or any portion of the information obtained through the
  Services available to any third party."*
- **URLhaus** (malware-distribution URLs, used as a cross-check). Its
  fair-use terms don't clearly say whether redistribution is allowed.

Until 2026-09-23 the daily snapshots in `data/live/` (and, on the
`testing-new` branch, `testing_new/live/` and the retrospective corpus) were
committed as plain gzipped CSV in a public repository. That makes OpenPhish's
data available to anyone, which its terms don't allow. It also meant a public
repo was serving live phishing and malware links. We found this during our own
review of the project and fixed it the same day.

## What changed

1. **Encrypted storage.** Any file containing feed URLs is written as
   `<name>.csv.gz.enc`: the same gzipped CSV, encrypted with
   [Fernet](https://cryptography.io/en/latest/fernet/) (AES-128-CBC with an
   HMAC). The key is a GitHub Actions secret (`FEED_KEY`) and a local file the
   team keeps; it is never committed. `phishdrift/sealed.py` does the reading and
   writing, and every loader decrypts in memory.

2. **History rewritten.** Using `git filter-repo`, every past version of those
   files was replaced by its encrypted version, in the same commit, under the
   same path plus `.enc`. Author and committer dates were kept. Afterwards we
   scanned every object in the history: no blob contains a feed row, every
   `.enc` blob decrypts to the original schema, and a random sample of 300 real
   phishing URLs from the data appears in no blob at all. Both `main` and
   `testing-new` were force-pushed on 2026-09-23, so every commit hash from
   before that date has changed.

3. **Nothing else changed.** Feature code, models, scores, results and the
   dashboard are untouched. None of the published results files contain feed
   URLs.

## What it means for the prospective record

The live record's value is that each day's data is committed *before* it is
scored, so it can't be adjusted afterward. That still holds: the ciphertext is
committed first, exactly as the plain file used to be, and changing a single
byte of it makes decryption fail.

What the rewrite costs is outside evidence for the dates of snapshots taken
before 2026-09-23. Commit dates were preserved, but the original commits'
hashes no longer exist in the repository. For those days the record rests on
the preserved commit dates and on the timestamps inside the rows themselves
(`first_seen_utc`, and for the retrospective corpus the dates recorded by
OpenPhish's own public git history).

## How to check it

`SEALED_SHA256.txt` lists, for every encrypted file, the SHA-256 of its
decrypted CSV. Anyone we give the key to (a judge, a reviewer) can decrypt a
file and confirm it matches. Without the key the URLs can't be recovered, which
is the point.

The feeds' own data remains public at its source: OpenPhish's feed is a public
git repository, and `phishdrift/retro.py` rebuilds the retrospective phishing
sample from it. Anyone who wants the URLs can get them from the publisher under
the publisher's terms.

## What this doesn't cover

GitHub can keep commits that are no longer on any branch reachable by their
hash for a while after a force-push. Removing those cached copies requires a
request to GitHub Support, which the repository owner has to make. Forks and
existing local clones made before 2026-09-23 also still hold the old history.
