module.exports = {
  apps: [{
    name: 'poker44-model-2',
    script: '/opt/poker44-model-miner-2/start_miner.sh',
    cwd: '/opt/poker44-model-miner-2',
    env: {
      PATH: '/opt/poker44-model-miner-2/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
      VIRTUAL_ENV: '/opt/poker44-model-miner-2/.venv',
      PYTHONPATH: '/opt/poker44-model-miner-2:/opt/poker44-model-miner-2/model',
      POKER44_MODEL_REPO_URL: 'https://github.com/EmmanuelGreene/poker44-rank-detector-b',
      POKER44_ARTIFACT: 'rank_detector_b_v32.pkl',
      POKER44_MAX_POS_FRAC: '0.16',
      LOGGING_DEBUG: 'false'
    },
    autorestart: true,
    max_restarts: 5,
    restart_delay: 30000,
    max_memory_restart: '2G',
    kill_timeout: 10000,
    log_date_format: 'YYYY-MM-DDTHH:mm:ss'
  }],
};
