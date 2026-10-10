# Research Freqtrade adapter. No credentials are baked into the image.
# Pin the base image by digest in regulated/production environments.
FROM freqtradeorg/freqtrade:stable

USER root
COPY pyproject.toml README.md /opt/bnb-btc-strategy/
COPY bnb_btc_strategy /opt/bnb-btc-strategy/bnb_btc_strategy
RUN python -m pip install --no-cache-dir '/opt/bnb-btc-strategy[mlsd]'
RUN python -c "import MLSD; from bnb_btc_strategy.freqtrade_strategy import BaseBTCCointegrationStrategy; import bnb_btc_strategy.mlsd_layer as m; assert MLSD.__version__ == '0.2.1'; assert len(m.FEATURE_SERIES) == 5; assert BaseBTCCointegrationStrategy.INTERFACE_VERSION == 3"
COPY OptimizedCTA_v2_Strategy.py OptimizedETHStrategy.py /freqtrade/user_data/strategies/
# Research MLSD model file is NEVER bundled; mount a trusted JSON file explicitly.
USER ftuser
