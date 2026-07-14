#!/bin/bash
cd /opt/poker44-model-miner-2
exec /opt/poker44-model-miner-2/.venv/bin/python model/poker44_miner.py \
  --netuid 126 \
  --wallet.name mywallet \
  --wallet.hotkey sn126_2 \
  --subtensor.network finney \
  --neuron.name poker44-miner-2 \
  --axon.port 8193 \
  --blacklist.force_validator_permit
