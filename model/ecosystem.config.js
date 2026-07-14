// pm2 process definition for poker44-rank-detector-b (HG2Blend).
// SECURITY: no wallet/hotkey/port/repo hardcoded — everything comes from .env.
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const REPO = path.resolve(__dirname, "..");
const MODEL = path.join(REPO, "model");
const CANDIDATES = [
  path.join(REPO, "miner_env", "bin", "python"),
  "/root/miner_env/bin/python",
];
const PY = CANDIDATES.find((p) => fs.existsSync(p)) || "python3";

function loadEnv(p) {
  const out = {};
  try {
    for (const raw of fs.readFileSync(p, "utf8").split("\n")) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const i = line.indexOf("=");
      if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).split("#")[0].trim().replace(/^["']|["']$/g, "");
    }
  } catch (e) {}
  return out;
}
const E = { ...process.env, ...loadEnv(path.join(REPO, ".env")) };
for (const key of [
  "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "SOCKS_PROXY", "SOCKS5_PROXY",
  "http_proxy", "https_proxy", "all_proxy", "socks_proxy", "socks5_proxy",
  "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY",
]) {
  E[key] = "";
}

const WALLET = (E.POKER44_WALLET_NAME || "").trim();
const HOTKEY = (E.POKER44_WALLET_HOTKEY || "").trim();
const NETUID = E.POKER44_NETUID || "126";
const PORT = E.POKER44_AXON_PORT || "9002";
const NETWORK = E.POKER44_SUBTENSOR_NETWORK || "finney";
const NAME = E.POKER44_PM2_NAME || "poker44_high_2";
if (!WALLET || !HOTKEY) {
  throw new Error("poker44-rank-detector-b: missing POKER44_WALLET_NAME / POKER44_WALLET_HOTKEY in .env");
}

let REPO_COMMIT = (E.POKER44_MODEL_REPO_COMMIT || "").trim();
if (!REPO_COMMIT) try { REPO_COMMIT = execSync(`git -C ${REPO} rev-parse HEAD`).toString().trim(); } catch (e) {}

module.exports = {
  apps: [
    {
      name: NAME,
      script: path.join(MODEL, "poker44_miner.py"),
      interpreter: PY,
      interpreter_args: "-u",
      cwd: REPO,
      args: [
        "--netuid", NETUID,
        "--wallet.name", WALLET,
        "--wallet.hotkey", HOTKEY,
        "--subtensor.network", NETWORK,
        "--axon.port", PORT,
        "--logging.debug",
        "--blacklist.force_validator_permit",
      ].join(" "),
      env: {
        ...E,
        POKER44_REPO: REPO,
        POKER44_MODEL_REPO_COMMIT: REPO_COMMIT,
      },
      autorestart: true, max_restarts: 20, min_uptime: "30s",
      restart_delay: 5000, kill_timeout: 10000,
    },
  ],
};
