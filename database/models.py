# database/models.py
import datetime
# Import necessary types from SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, DateTime, BigInteger
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func # For default timestamp

# Define a Base class using SQLAlchemy's Declarative system
class Base(DeclarativeBase):
    pass

# Define the model for our trades log table
class TradeLog(Base):
    __tablename__ = 'trades_log' # The actual table name in the database

    # Columns Definition
    order_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True) # MT5 Order Ticket ID
    position_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=True) # <<< COLUMN ADDED HERE (MT5 Position ID)
    timestamp_open: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=False), default=func.now()) # Time trade was logged/opened
    symbol: Mapped[str] = mapped_column(String(30), index=True) # e.g., 'EURUSD'
    trade_type: Mapped[int] = mapped_column(Integer) # 0 for Buy, 1 for Sell (MT5 constants)
    lot_size: Mapped[float] = mapped_column(Float) # Volume traded
    entry_price: Mapped[float] = mapped_column(Float) # Execution price
    stop_loss_price: Mapped[float] = mapped_column(Float) # Requested SL
    take_profit_price: Mapped[float] = mapped_column(Float) # Requested TP
    prediction: Mapped[int] = mapped_column(Integer, nullable=True) # Model prediction (0 or 1)
    confidence: Mapped[float] = mapped_column(Float, nullable=True) # Model confidence
    magic_number: Mapped[int] = mapped_column(Integer, index=True) # Magic number used
    comment: Mapped[str] = mapped_column(String(100), nullable=True) # Order comment
    status: Mapped[str] = mapped_column(String(20), default='OPEN', index=True) # e.g., OPEN, CLOSED, ERROR
    timestamp_close: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=False), nullable=True) # Time trade was closed
    close_price: Mapped[float] = mapped_column(Float, nullable=True) # Price at closure
    profit: Mapped[float] = mapped_column(Float, nullable=True) # Profit/Loss amount
    commission: Mapped[float] = mapped_column(Float, nullable=True) # Commission charged
    swap: Mapped[float] = mapped_column(Float, nullable=True) # Swap charged/paid

    def __repr__(self):
        # Helpful representation when printing the object
        return (f"<TradeLog(order_id={self.order_id}, position_id={self.position_id}, symbol='{self.symbol}', "
                f"type={self.trade_type}, lots={self.lot_size}, status='{self.status}')>")

# You can define other tables/models here later if needed