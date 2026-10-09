# Research Freqtrade adapter. No credentials are baked into the image.
# Pin the base image by digest in regulated/production environments.
FROM freqtradeorg/freqtrade:stable

USER root
COPY pyproject.toml README.md /opt/bnb-btc-strategy/
COPY bnb_btc_strategy /opt/bnb-btc-strategy/bnb_btc_strategy
RUN python -m pip install --no-cache-dir '/opt/bnb-btc-strategy[mlsd]'
COPY OptimizedCTA_v2_Strategy.py OptimizedETHStrategy.py /freqtrade/user_data/strategies/
# Research MLSD model file is NEVER bundled; mount a trusted JSON file explicitly.
USER ftuser
