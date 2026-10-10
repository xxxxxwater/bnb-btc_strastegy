# Extend the official Freqtrade user environment; no secrets are baked in.
# For regulated deployments, replace stable with a validated immutable digest.
FROM freqtradeorg/freqtrade:stable

# The upstream image installs Freqtrade's dependencies as ftuser in ~/.local.
# Do NOT install MLSD as root: that creates a second Python dependency set.
USER root
COPY --chown=ftuser:ftuser pyproject.toml README.md /opt/bnb-btc-strategy/
COPY --chown=ftuser:ftuser bnb_btc_strategy /opt/bnb-btc-strategy/bnb_btc_strategy
COPY --chown=ftuser:ftuser OptimizedCTA_v2_Strategy.py OptimizedETHStrategy.py /freqtrade/user_data/strategies/

USER ftuser
RUN python -m pip install --user --no-cache-dir '/opt/bnb-btc-strategy[mlsd]' \
    && python -m pip check
# Build must prove Freqtrade AND MLSD can import under the actual runtime user.
RUN python -c "import MLSD; from bnb_btc_strategy.freqtrade_strategy import BaseBTCCointegrationStrategy; import bnb_btc_strategy.mlsd_layer as m; assert MLSD.__version__ == '0.2.1'; assert len(m.FEATURE_SERIES) == 5; assert BaseBTCCointegrationStrategy.INTERFACE_VERSION == 3"
# Research-only artifact: operator mounts any trained model as trusted read-only JSON.
