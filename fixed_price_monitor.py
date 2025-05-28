"""
Fixed Price Monitor for GFX Trading Bot
Uses proper Polygon.io API endpoints for reliable price monitoring
"""

import os
import logging
import asyncio
import requests
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from gfx_trading_bot import Base, User, Signal
import telegram

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedPriceMonitor:
    """Fixed price monitoring with proper API usage"""
    
    def __init__(self):
        """Initialize the price monitor"""
        self.api_key = os.getenv('POLYGON_API_KEY')
        self.session_maker = None
        self.is_monitoring = False
        
        # Initialize Telegram bot for notifications
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.bot = None
        if self.telegram_token:
            self.bot = telegram.Bot(token=self.telegram_token)
            logger.info("Telegram bot initialized for notifications")
        
        if self.api_key:
            logger.info("Polygon API key found")
        else:
            logger.error("POLYGON_API_KEY not found")
        
        # Initialize database
        try:
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                engine = create_engine(database_url)
                Session = sessionmaker(bind=engine)
                self.session_maker = Session
                logger.info("Database connection initialized")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price using HTTP requests to Polygon API"""
        if not self.api_key:
            return None
            
        try:
            base_url = "https://api.polygon.io"
            
            # Handle different instrument types
            if symbol == 'XAUUSD':
                # Gold price using currency conversion
                url = f"{base_url}/v1/conversion/XAU/USD?amount=1&precision=4&apikey={self.api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'converted' in data:
                        return float(data['converted'])
            
            elif symbol == 'XAGUSD':
                # Silver price
                url = f"{base_url}/v1/conversion/XAG/USD?amount=1&precision=4&apikey={self.api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'converted' in data:
                        return float(data['converted'])
            
            elif 'BTC' in symbol or 'ETH' in symbol:
                # Crypto using multiple approaches
                if symbol == 'BTCUSD':
                    # Try crypto last trade
                    url = f"{base_url}/v1/last/crypto/BTC/USD?apikey={self.api_key}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'last' in data and 'price' in data['last']:
                            return float(data['last']['price'])
                    
                    # Fallback to conversion
                    url = f"{base_url}/v1/conversion/BTC/USD?amount=1&precision=2&apikey={self.api_key}"
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if 'converted' in data:
                            return float(data['converted'])
            
            elif len(symbol) == 6:
                # Standard forex pairs
                base_currency = symbol[:3]
                quote_currency = symbol[3:]
                url = f"{base_url}/v1/conversion/{base_currency}/{quote_currency}?amount=1&precision=6&apikey={self.api_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if 'converted' in data:
                        return float(data['converted'])
            
        except Exception as e:
            logger.debug(f"Could not get price for {symbol}: {e}")
            
        return None
    
    async def check_all_signals(self):
        """Check all active signals for TP/SL hits"""
        if not self.session_maker:
            return
            
        session = self.session_maker()
        try:
            active_signals = session.query(Signal).filter(Signal.is_active == True).all()
            
            if not active_signals:
                return
                
            logger.info(f"Monitoring {len(active_signals)} active signals")
            
            for signal in active_signals:
                current_price = self.get_current_price(signal.instrument)
                
                if current_price:
                    await self.check_signal_levels(signal, current_price, session)
                    logger.info(f"{signal.instrument}: ${current_price:.4f}")
                else:
                    logger.warning(f"No price data for {signal.instrument}")
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
            session.rollback()
        finally:
            session.close()
    
    async def check_signal_levels(self, signal: Signal, current_price: float, session):
        """Check if price hits TP or SL levels"""
        is_buy = signal.signal_type.upper() == 'BUY'
        updated = False
        
        # Check Take Profit levels
        if not signal.tp1_hit and signal.take_profit1:
            tp1_hit = (is_buy and current_price >= signal.take_profit1) or \
                     (not is_buy and current_price <= signal.take_profit1)
            if tp1_hit:
                signal.tp1_hit = True
                updated = True
                logger.info(f"TP1 HIT: {signal.instrument} at {current_price}")
                await self.send_tp_notification(signal, 1, current_price)
                
                # Close trade immediately after TP1 hit (winning trade)
                signal.is_active = False
                signal.closed_at = datetime.now()
                logger.info(f"TRADE CLOSED (WIN): {signal.instrument} - TP1 reached")
        
        if not signal.tp2_hit and signal.take_profit2 and signal.is_active:
            tp2_hit = (is_buy and current_price >= signal.take_profit2) or \
                     (not is_buy and current_price <= signal.take_profit2)
            if tp2_hit:
                signal.tp2_hit = True
                updated = True
                logger.info(f"TP2 HIT: {signal.instrument} at {current_price}")
                await self.send_tp_notification(signal, 2, current_price)
                
                # Close trade immediately after TP2 hit (winning trade)
                signal.is_active = False
                signal.closed_at = datetime.now()
                logger.info(f"TRADE CLOSED (WIN): {signal.instrument} - TP2 reached")
        
        if not signal.tp3_hit and signal.take_profit3 and signal.is_active:
            tp3_hit = (is_buy and current_price >= signal.take_profit3) or \
                     (not is_buy and current_price <= signal.take_profit3)
            if tp3_hit:
                signal.tp3_hit = True
                updated = True
                logger.info(f"TP3 HIT: {signal.instrument} at {current_price}")
                await self.send_tp_notification(signal, 3, current_price)
                
                # Close trade immediately after TP3 hit (winning trade)
                signal.is_active = False
                signal.closed_at = datetime.now()
                logger.info(f"TRADE CLOSED (WIN): {signal.instrument} - TP3 reached")
        
        # Check Stop Loss
        if not signal.sl_hit and signal.stop_loss:
            sl_hit = (is_buy and current_price <= signal.stop_loss) or \
                    (not is_buy and current_price >= signal.stop_loss)
            if sl_hit:
                # If any TP was hit, close as win; otherwise as loss
                if signal.tp1_hit or signal.tp2_hit or signal.tp3_hit:
                    signal.is_active = False
                    signal.closed_at = datetime.now()
                    logger.info(f"TRADE CLOSED (WIN): {signal.instrument} - TP hit before SL")
                    # No notification needed - trade is a win
                else:
                    signal.sl_hit = True
                    signal.is_active = False
                    signal.closed_at = datetime.now()
                    logger.info(f"SL HIT: {signal.instrument} at {current_price}")
                    await self.send_sl_notification(signal, current_price)
                updated = True
        
        if updated:
            session.add(signal)
    
    async def send_tp_notification(self, signal, tp_level: int, current_price: float):
        """Send TP hit notification to user"""
        if not self.bot:
            return
            
        try:
            # Get user from signal
            session = self.session_maker()
            user = session.query(User).filter(User.id == signal.user_id).first()
            
            if not user:
                return
                
            message = f"🎯 **TP{tp_level} HIT!** 🎯\n\n"
            message += f"**Instrument:** {signal.instrument}\n"
            message += f"**Signal Type:** {signal.signal_type}\n"
            message += f"**Current Price:** ${current_price:.4f}\n"
            message += f"**TP{tp_level} Target:** ${getattr(signal, f'take_profit{tp_level}'):.4f}\n\n"
            message += f"✅ **Congratulations! Your trade is in profit!**\n\n"
            message += f"📊 GFX Trading Assistant"
            
            await self.bot.send_message(
                chat_id=user.telegram_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"TP{tp_level} notification sent to user {user.telegram_id}")
            
        except Exception as e:
            logger.error(f"Failed to send TP notification: {e}")
        finally:
            session.close()
    
    async def send_sl_notification(self, signal, current_price: float):
        """Send SL hit notification to user"""
        if not self.bot:
            return
            
        try:
            # Get user from signal
            session = self.session_maker()
            user = session.query(User).filter(User.id == signal.user_id).first()
            
            if not user:
                return
                
            message = f"🛑 **STOP LOSS HIT** 🛑\n\n"
            message += f"**Instrument:** {signal.instrument}\n"
            message += f"**Signal Type:** {signal.signal_type}\n"
            message += f"**Current Price:** ${current_price:.4f}\n"
            message += f"**Stop Loss:** ${signal.stop_loss:.4f}\n\n"
            message += f"📉 **Trade closed at loss. Better luck next time!**\n\n"
            message += f"📊 GFX Trading Assistant"
            
            await self.bot.send_message(
                chat_id=user.telegram_id,
                text=message,
                parse_mode='Markdown'
            )
            logger.info(f"SL notification sent to user {user.telegram_id}")
            
        except Exception as e:
            logger.error(f"Failed to send SL notification: {e}")
        finally:
            session.close()
    
    async def start_monitoring(self, interval=30):
        """Start the monitoring loop"""
        if not self.api_key:
            logger.error("Cannot start monitoring: No API key")
            return
            
        self.is_monitoring = True
        logger.info(f"Starting price monitoring every {interval} seconds")
        
        while self.is_monitoring:
            try:
                await self.check_all_signals()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)

async def main():
    """Main function to run the fixed price monitor"""
    monitor = FixedPriceMonitor()
    await monitor.start_monitoring()

if __name__ == "__main__":
    asyncio.run(main())