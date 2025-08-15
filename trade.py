# trade.py
from datetime import datetime

class Trade:
    def __init__(self, symbol, direction, entry_price, sl, tps, confidence=None):
        self.symbol = symbol
        self.direction = direction  # "BUY" or "SELL"
        self.entry_price = entry_price
        self.sl = sl
        self.tps = tps  # list of target prices (e.g., [TP1, TP2, TP3, TP4])
        # tp_hits will track which individual TP levels have been achieved
        # Initialize based on the number of TPs provided, e.g., if tps has 3 elements, it will be [False, False, False]
        self.tp_hits = [False] * len(tps) 
        self.active = True # True if the trade is currently open/active
        self.telegram_message_id = None # Optional: To link with a Telegram message for updates
        self.entry_time = datetime.now() # Timestamp when the trade was initiated
        
        # Trade Status: Tracks the current state of the trade
        # Possible values: 'OPEN', 'CLOSED_SL', 'CLOSED_TP_PARTIAL', 'CLOSED_TP_FULL', 'CLOSED_REVERSAL_TO_ENTRY'
        self.status = 'OPEN' 
        self.closed_time = None # Timestamp when the trade was closed (if applicable)
        
        self.confidence = confidence # The confidence score of the signal that generated this trade

    def __repr__(self):
        # A helpful representation for debugging and logging
        tps_str = ", ".join([f"{tp:.4f}" for tp in self.tps])
        tp_hits_str = ", ".join([f"TP{i+1}: {'Hit' if hit else 'Open'}" for i, hit in enumerate(self.tp_hits)])
        return (
            f"Trade(\n"
            f"  Symbol: {self.symbol}, Direction: {self.direction},\n"
            f"  Entry: {self.entry_price:.4f}, SL: {self.sl:.4f},\n"
            f"  TPs: [{tps_str}], TP Hits: [{tp_hits_str}]\n"
            f"  Status: {self.status}, Active: {self.active},\n"
            f"  Entry Time: {self.entry_time.strftime('%Y-%m-%d %H:%M:%S')},\n"
            f"  Closed Time: {self.closed_time.strftime('%Y-%m-%d %H:%M:%S') if self.closed_time else 'N/A'},\n"
            f"  Confidence: {self.confidence:.2f}% (if available)\n"
            f")"
        )