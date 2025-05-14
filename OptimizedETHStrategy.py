

    # ============ 追踪止损参数 ============
    trailing_stop = True
    trailing_stop_positive = 0.03  # 盈利3%后激活
    trailing_stop_positive_offset = 0.08  # 从8%利润开始追踪
    trailing_only_offset_is_reached = True

    # ============ 订单类型 ============
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # 训练过的yperopt参数 （不要修改）
    buy_rsi = IntParameter(low=1, high=50, default=31.685, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=72.581, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    