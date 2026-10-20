# Qwen3.8-27B B550 transfer verification

Date: 2026-10-18

## Scope

This rechecks the reported Qwen3.8-27B download failure against the live B550
artifact, its full HFD verification receipt, and active transfer processes. It
does not change the model or any download job.

## Result

The 27B artifact did not fail. The live B550 copy is complete and verified:

- repository: `Qwen/Qwen3.8-27B`;
- immutable revision:
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- path: `/home/kearm/training/models/Qwen3.8-27B-1d4bf0f`;
- files: 32/32;
- bytes: 55,586,114,863;
- `.aria2` control files: 0;
- manifest SHA-256:
  `3f7f666dfa59f7b7bbc761a2dc3bbf122ebffc112a872ff11983c3c95f76d390`;
- verification receipt:
  `/home/kearm/training/logs/hfd-qwen38-27b-verify.json`;
- verification status: `verified` in full mode at 2026-10-18T20:13:57Z.

The failed-looking `/home/kearm/training/logs/hfd-qwen38-27b.json` is the
earlier slow WAN HFD attempt. Its final lines still show two shards in progress
and approximately 326-344 KiB/s throughput. That attempt was intentionally
stopped because unrelated HFD jobs were consuming the wide-area link, then the
verified Mac copy was transferred over the LAN and HFD full verification was
rerun on the final destination.

At the live check, unrelated HFD jobs for DeepSeek-V4.1-Flash-1017 and
Qwen3.8-Flash-Next were still active. No Qwen3.8-27B HFD process remained,
which is consistent with completion rather than failure. B550 had about
2.0 TB free on the destination filesystem.

## Evidence

Successful live SSH probe over the cataloged `B550` alias:

```text
ssh -o BatchMode=yes -o ConnectTimeout=8 B550 \
  'test -d /home/kearm/training/models/Qwen3.8-27B-1d4bf0f &&
   find /home/kearm/training/models/Qwen3.8-27B-1d4bf0f -type f | wc -l &&
   find /home/kearm/training/models/Qwen3.8-27B-1d4bf0f -name "*.aria2" -type f | wc -l &&
   du -sb /home/kearm/training/models/Qwen3.8-27B-1d4bf0f &&
   jq . /home/kearm/training/logs/hfd-qwen38-27b-verify.json &&
   pgrep -af "hfd|Qwen3.8-27B" || true'
```

The equivalent combined probe exited 0 and returned the counts, byte total,
verified receipt, and active-process list recorded above.

## Non-claims

This receipt does not claim that the pinned BF16 27B teacher has been loaded
for DFlash2 capture, that a teacher precision decision has been made, or that
any unrelated download job was modified. The earlier WAN HFD log remains as
historical evidence and must not be read as final transfer status.
