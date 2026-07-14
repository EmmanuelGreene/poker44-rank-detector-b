# poker44-rank-detector-b

Poker44 SN126 bot-risk detector miner package.

This repository contains the serving code, model/runtime files, and manifest-facing implementation used for a Bittensor Poker44 miner. It serves per-chunk bot-risk scores through the standard Poker44 DetectionSynapse miner interface.

## Runtime

- Subnet: SN126 Poker44
- Inference: local Python model/runtime
- Data attestation: no validator-private labels are used in this repository.

## Run

Use an existing registered wallet/hotkey and axon port:

```bash
python neurons/miner.py --netuid 126 \
  --wallet.name <wallet> --wallet.hotkey <hotkey> \
  --subtensor.network finney --axon.port <port> \
  --blacklist.force_validator_permit
```
