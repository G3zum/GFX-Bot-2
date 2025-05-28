"""
GFX Trading Bot - Professional Trade Signal & Education Bot

This bot provides:
1. User management with admin approval for access
2. Chart analysis with clear trade signals (Entry, SL, TP levels)
3. Risk management with lot size calculator based on signals
"""
import os
import sys
import logging
import re
import json
import base64
import asyncio
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler, 
    ContextTypes, filters
)

from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import requests
from openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_RISK_PERCENTAGE = 2.0
DEFAULT_ADMIN_ID = int(os.environ.get("ADMIN_TELEGRAM_ID", "2025152767"))  # Default admin ID

# Database setup
Base = declarative_base()
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before use
        pool_recycle=300,    # Recycle connections every 5 minutes
        connect_args={"sslmode": "require"}
    )
    Session = sessionmaker(bind=engine)
else:
    logger.error("DATABASE_URL environment variable not set")
    engine = None
    Session = None

class User(Base):
    """User model for the database"""
    __tablename__ = 'gfx_users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, unique=True, nullable=False)  # Changed to BigInteger for large Telegram IDs
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    is_admin = Column(Boolean, default=False)
    is_authorized = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    signals = relationship("Signal", back_populates="user")
    
    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, username={self.username})>"

class Signal(Base):
    """Signal model for the database"""
    __tablename__ = 'gfx_signals'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('gfx_users.id'), nullable=False)
    instrument = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit1 = Column(Float, nullable=True)
    take_profit2 = Column(Float, nullable=True)
    take_profit3 = Column(Float, nullable=True)
    analysis_text = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    tp1_hit = Column(Boolean, default=False)
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    sl_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    closed_at = Column(DateTime, nullable=True)
    
    user = relationship("User", back_populates="signals")
    
    def __repr__(self):
        return f"<Signal(id={self.id}, instrument={self.instrument}, type={self.signal_type})>"

class AccessRequest(Base):
    """Access request model for the database"""
    __tablename__ = 'gfx_access_requests'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(BigInteger, nullable=False)
    status = Column(String(20), default="pending")  # pending, approved, rejected
    requested_at = Column(DateTime, default=datetime.now)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(BigInteger, nullable=True)  # Admin's telegram_id
    
    def __repr__(self):
        return f"<AccessRequest(telegram_id={self.telegram_id}, status={self.status})>"


#############################
# USER MANAGEMENT AGENT #
#############################

class UserManagementAgent:
    """Agent for handling user access and management"""
    
    def __init__(self, admin_ids=None):
        """
        Initialize the user management agent
        
        Args:
            admin_ids (list, optional): List of admin Telegram IDs
        """
        self.admin_ids = admin_ids or [DEFAULT_ADMIN_ID]
        logger.info(f"UserManagementAgent initialized with admin IDs: {self.admin_ids}")
    
    def is_admin(self, telegram_id: int) -> bool:
        """Check if a user is an admin"""
        # First check hardcoded admin IDs
        if telegram_id in self.admin_ids:
            return True
            
        # Then check database
        if Session:
            session = Session()
            try:
                user = session.query(User).filter(User.telegram_id == telegram_id).first()
                return user and user.is_admin
            except Exception as e:
                logger.error(f"Error checking admin status: {e}")
                return False
            finally:
                session.close()
        
        return False
    
    def is_authorized(self, telegram_id: int) -> bool:
        """Check if a user is authorized to use the bot"""
        # Admins are always authorized
        if self.is_admin(telegram_id):
            return True
            
        # Check database for other authorized users
        if Session:
            session = Session()
            try:
                user = session.query(User).filter(User.telegram_id == telegram_id).first()
                return user and user.is_authorized
            except Exception as e:
                logger.error(f"Error checking authorization status: {e}")
                return False
            finally:
                session.close()
        
        return False
    
    def create_or_update_user(self, telegram_id: int, username: Optional[str] = None, 
                             first_name: Optional[str] = None, last_name: Optional[str] = None) -> Optional[User]:
        """Create or update a user in the database"""
        if not Session:
            logger.error("Database session not available")
            return None
            
        session = Session()
        try:
            user = session.query(User).filter(User.telegram_id == telegram_id).first()
            
            if user:
                # Update existing user
                user.username = username
                user.first_name = first_name
                user.last_name = last_name
                user.last_active = datetime.now()
            else:
                # Create new user
                user = User(
                    telegram_id=telegram_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    is_admin=False,
                    is_authorized=False
                )
                session.add(user)
                
            session.commit()
            return user
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            session.rollback()
            return None
        finally:
            session.close()
    
    def request_access(self, telegram_id: int) -> bool:
        """Request access for a user"""
        if not Session:
            logger.error("Database session not available")
            return False
            
        session = Session()
        try:
            # Check if there's already a pending request
            existing_request = session.query(AccessRequest).filter(
                AccessRequest.telegram_id == telegram_id,
                AccessRequest.status == "pending"
            ).first()
            
            if existing_request:
                logger.info(f"User {telegram_id} already has a pending access request")
                return True
                
            # Create new access request
            access_request = AccessRequest(
                telegram_id=telegram_id,
                status="pending",
                requested_at=datetime.now()
            )
            session.add(access_request)
            session.commit()
            logger.info(f"Created access request for user {telegram_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating access request: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_pending_requests(self) -> List[Dict]:
        """Get all pending access requests"""
        if not Session:
            logger.error("Database session not available")
            return []
            
        session = Session()
        try:
            requests = session.query(AccessRequest).filter(AccessRequest.status == "pending").all()
            
            result = []
            for req in requests:
                # Get user info if available
                user = session.query(User).filter(User.telegram_id == req.telegram_id).first()
                
                request_info = {
                    "id": req.id,
                    "telegram_id": req.telegram_id,
                    "requested_at": req.requested_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "username": user.username if user else None,
                    "first_name": user.first_name if user else None,
                    "last_name": user.last_name if user else None
                }
                result.append(request_info)
                
            return result
        except Exception as e:
            logger.error(f"Error fetching pending requests: {e}")
            return []
        finally:
            session.close()
    
    def approve_request(self, telegram_id: int, admin_id: int) -> bool:
        """Approve an access request"""
        if not Session:
            logger.error("Database session not available")
            return False
            
        session = Session()
        try:
            # Find the pending request
            request = session.query(AccessRequest).filter(
                AccessRequest.telegram_id == telegram_id,
                AccessRequest.status == "pending"
            ).first()
            
            if not request:
                logger.warning(f"No pending access request found for user {telegram_id}")
                return False
                
            # Update the request
            request.status = "approved"
            request.processed_at = datetime.now()
            request.processed_by = admin_id
            
            # Set user as authorized
            user = session.query(User).filter(User.telegram_id == telegram_id).first()
            if user:
                user.is_authorized = True
            else:
                # Create user if doesn't exist (rare case)
                user = User(
                    telegram_id=telegram_id,
                    is_authorized=True
                )
                session.add(user)
                
            session.commit()
            logger.info(f"Access request for user {telegram_id} approved by admin {admin_id}")
            return True
        except Exception as e:
            logger.error(f"Error approving access request: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def reject_request(self, telegram_id: int, admin_id: int) -> bool:
        """Reject an access request"""
        if not Session:
            logger.error("Database session not available")
            return False
            
        session = Session()
        try:
            # Find the pending request
            request = session.query(AccessRequest).filter(
                AccessRequest.telegram_id == telegram_id,
                AccessRequest.status == "pending"
            ).first()
            
            if not request:
                logger.warning(f"No pending access request found for user {telegram_id}")
                return False
                
            # Update the request
            request.status = "rejected"
            request.processed_at = datetime.now()
            request.processed_by = admin_id
            
            session.commit()
            logger.info(f"Access request for user {telegram_id} rejected by admin {admin_id}")
            return True
        except Exception as e:
            logger.error(f"Error rejecting access request: {e}")
            session.rollback()
            return False
        finally:
            session.close()


#############################
# SIGNAL GENERATION AGENT #
#############################

class SignalGenerationAgent:
    """Agent for analyzing charts and generating trading signals"""
    
    def __init__(self):
        """Initialize the signal generation agent"""
        self.api_key = os.environ.get('OPENAI_API_KEY_2') or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            logger.warning("OPENAI_API_KEY_2 or OPENAI_API_KEY not set - chart analysis will not work")
        else:
            logger.info("Using OpenAI API key for chart analysis")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        logger.info("SignalGenerationAgent initialized")
    
    def format_price(self, price, instrument):
        """
        Format price based on instrument type
        
        Args:
            price (float): Price value
            instrument (str): Trading instrument
            
        Returns:
            str: Formatted price string
        """
        if not price:
            return "N/A"
            
        # Convert to string for pattern matching
        instrument_upper = str(instrument).upper()
        
        # Commodities and crypto - 2 decimal places
        commodities_crypto = [
            'XAUUSD', 'XAGUSD', 'GOLD', 'SILVER', 'OIL', 'CRUDE',
            'BTCUSD', 'ETHUSD', 'BTC', 'ETH', 'CRYPTO',
            'US30', 'NASDAQ', 'SPX500', 'DAX', 'NIKKEI'
        ]
        
        for symbol in commodities_crypto:
            if symbol in instrument_upper:
                return f"{price:.2f}"
        
        # FX symbols - 6 decimal places
        return f"{price:.6f}"
    
    def encode_image(self, image_bytes):
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    async def analyze_chart(self, image_bytes):
        """
        Analyze a chart image and generate a trading signal
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            dict: Analysis result with signal details and formatted text
        """
        if not self.client:
            return {"error": "OpenAI API key not configured"}
            
        try:
            # Encode image to base64
            base64_image = self.encode_image(image_bytes)
            
            # Create the API request
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You MUST analyze any chart and provide a trading signal. NO EXCEPTIONS.

Look at the chart and provide this EXACT format:

Instrument: XAUUSD
Signal: BUY
Entry: 2645.50
Stop Loss: 2635.00
Take Profit 1: 2655.00
Take Profit 2: 2665.00
Take Profit 3: 2675.00

Analysis: Chart shows uptrend pattern with support at current levels.

RULES - NO REFUSING:
1. If you see any price chart, generate a signal
2. Look for trend direction (up=BUY, down=SELL)  
3. Use price levels visible on chart
4. For Gold charts use XAUUSD
5. NEVER say you can't analyze - always provide signal
6. Must include Entry, Stop Loss, TP1, TP2, TP3"""
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this chart and provide a trading signal:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            # Extract the analysis text
            analysis_text = response.choices[0].message.content
            
            # Extract trade data from the analysis
            trade_data = self.extract_trade_data(analysis_text)
            
            # Format the response
            return {
                "success": True,
                "analysis_text": analysis_text,
                "trade_data": trade_data,
                "formatted_signal": self.format_signal_message(analysis_text, trade_data)
            }
        
        except Exception as e:
            logger.error(f"Error in chart analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_trade_data(self, analysis_text):
        """
        Extract trade data from analysis text
        
        Args:
            analysis_text (str): Analysis text from the model
            
        Returns:
            dict: Dictionary with trade information or None if parsing fails
        """
        try:
            # Initialize result dictionary
            result = {
                "instrument": None,
                "signal_type": None,
                "entry_price": None,
                "stop_loss": None,
                "take_profit1": None,
                "take_profit2": None,
                "take_profit3": None
            }
            
            # Extract instrument
            instrument_match = re.search(r"(?i)instrument:?\s*([A-Z]{6}|[A-Z]{3}/[A-Z]{3}|[A-Z]+USD|[A-Z]+JPY|Gold|XAUUSD|Silver|XAGUSD|US30|NASDAQ|SPX500)", analysis_text)
            if instrument_match:
                result["instrument"] = instrument_match.group(1).upper()
                # Normalize some common instruments
                if result["instrument"] == "GOLD":
                    result["instrument"] = "XAUUSD"
                elif result["instrument"] == "SILVER":
                    result["instrument"] = "XAGUSD"
            
            # Extract signal type
            signal_match = re.search(r"(?i)signal:?\s*(BUY|SELL)", analysis_text)
            if not signal_match:
                signal_match = re.search(r"(?i)(BUY|SELL)\s+signal", analysis_text)
            if not signal_match:
                signal_match = re.search(r"(?i)signal\s+type:?\s*(BUY|SELL)", analysis_text)
            if signal_match:
                result["signal_type"] = signal_match.group(1).upper()
            
            # Helper function to convert price strings to float
            def to_float(match):
                if not match:
                    return None
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    return None
            
            # Extract entry price
            entry_match = re.search(r"(?i)entry:?\s*(\d+[,.]?\d*)", analysis_text)
            if not entry_match:
                entry_match = re.search(r"(?i)entry\s+price:?\s*(\d+[,.]?\d*)", analysis_text)
            if not entry_match:
                entry_match = re.search(r"(?i)entry\s+level:?\s*(\d+[,.]?\d*)", analysis_text)
            if not entry_match:
                entry_match = re.search(r"(?i)entry[\s:]*(\d+[,.]?\d*)", analysis_text)
            result["entry_price"] = to_float(entry_match)
            
            # Extract stop loss
            sl_match = re.search(r"(?i)stop\s*loss:?\s*(\d+[,.]?\d*)", analysis_text)
            if not sl_match:
                sl_match = re.search(r"(?i)SL:?\s*(\d+[,.]?\d*)", analysis_text)
            if not sl_match:
                sl_match = re.search(r"(?i)stop[\s:]*(\d+[,.]?\d*)", analysis_text)
            result["stop_loss"] = to_float(sl_match)
            
            # Extract take profits
            tp1_match = re.search(r"(?i)take\s*profit\s*1:?\s*(\d+[,.]?\d*)", analysis_text)
            if not tp1_match:
                tp1_match = re.search(r"(?i)TP1:?\s*(\d+[,.]?\d*)", analysis_text)
            if not tp1_match:
                tp1_match = re.search(r"(?i)take\s*profit:?\s*(\d+\.?\d*)", analysis_text)
            result["take_profit1"] = to_float(tp1_match)
            
            tp2_match = re.search(r"(?i)take\s*profit\s*2:?\s*(\d+[,.]?\d*)", analysis_text)
            if not tp2_match:
                tp2_match = re.search(r"(?i)TP2:?\s*(\d+[,.]?\d*)", analysis_text)
            result["take_profit2"] = to_float(tp2_match)
            
            tp3_match = re.search(r"(?i)take\s*profit\s*3:?\s*(\d+[,.]?\d*)", analysis_text)
            if not tp3_match:
                tp3_match = re.search(r"(?i)TP3:?\s*(\d+[,.]?\d*)", analysis_text)
            result["take_profit3"] = to_float(tp3_match)
            
            # Validate required fields - be more lenient
            if not (result["signal_type"] and result["entry_price"]):
                logger.warning("Failed to extract minimum required trade data")
                return None
            
            # Set default instrument if not found
            if not result["instrument"]:
                result["instrument"] = "XAUUSD"  # Default for most charts
                
            return result
            
        except Exception as e:
            logger.error(f"Error extracting trade data: {str(e)}")
            return None
    
    def format_signal_message(self, analysis_text, trade_data):
        """
        Format signal for Telegram message in GFX professional style
        
        Args:
            analysis_text (str): Raw analysis text
            trade_data (dict): Extracted trade data
            
        Returns:
            str: Formatted message for Telegram
        """
        if not trade_data:
            return "⚠️ Failed to generate a complete trading signal. Please try again with a clearer chart image."
            
        # Get trade details
        instrument = trade_data.get("instrument", "Unknown")
        signal_type = trade_data.get("signal_type", "Unknown")
        entry = trade_data.get("entry_price")
        sl = trade_data.get("stop_loss")
        tp1 = trade_data.get("take_profit1")
        tp2 = trade_data.get("take_profit2")
        tp3 = trade_data.get("take_profit3")
        
        # Format prices according to instrument type
        entry_formatted = self.format_price(entry, instrument)
        sl_formatted = self.format_price(sl, instrument)
        tp1_formatted = self.format_price(tp1, instrument) if tp1 else None
        tp2_formatted = self.format_price(tp2, instrument) if tp2 else None
        tp3_formatted = self.format_price(tp3, instrument) if tp3 else None
        
        # Calculate risk-reward ratio if possible
        rr_ratio = None
        if entry and sl and tp1:
            distance_to_sl = abs(entry - sl)
            distance_to_tp1 = abs(entry - tp1)
            if distance_to_sl > 0:
                rr_ratio = distance_to_tp1 / distance_to_sl
        
        # Create GFX professional formatted message
        message = "📊 *GFX CHART ANALYSIS*\n\n"
        message += "Thanks for sharing your chart! 📈\n\n"
        
        # Context Summary with checkmark
        message += "✅ *Context Summary:*\n"
        analysis_part = self.extract_analysis_section(analysis_text)
        if analysis_part:
            # Split analysis into bullet points
            analysis_lines = analysis_part.split('.')
            for line in analysis_lines[:3]:  # Take first 3 sentences
                if line.strip():
                    message += f"- {line.strip()}\n"
        message += "\n"
        
        # Trade Setup with target emoji
        signal_direction = "Long" if signal_type == "BUY" else "Short"
        message += f"🎯 *Trade Setup Idea ({signal_direction}):*\n"
        message += f"- {signal_type.title()} Zone: {entry_formatted}\n"
        message += f"- This is a {'demand' if signal_type == 'BUY' else 'supply'} zone with technical significance\n\n"
        
        # Targets with appropriate emoji
        target_emoji = "🟢" if signal_type == "BUY" else "🔴"
        message += f"{target_emoji} *Targets:*\n"
        if tp1_formatted:
            message += f"- TP1: {tp1_formatted}\n"
        if tp2_formatted:
            message += f"- TP2: {tp2_formatted}\n"
        if tp3_formatted:
            message += f"- TP3: {tp3_formatted}\n"
        message += "\n"
        
        # Stop Loss with red X emoji
        message += f"⛔ *Stop Loss:*\n"
        message += f"- {sl_formatted} (above {'support' if signal_type == 'BUY' else 'resistance'})\n\n"
        
        # Signal Summary with chart emoji
        message += f"📊 *Signal Summary:*\n"
        message += f"- {signal_type} {instrument} from {entry_formatted}\n"
        tp_list = [tp for tp in [tp1_formatted, tp2_formatted, tp3_formatted] if tp and tp != "N/A"]
        message += f"- SL: {sl_formatted} — TP: {', '.join(tp_list)}\n"
        
        # Add risk-reward ratio if calculated
        if rr_ratio:
            message += f"- Risk/Reward ratio: ~1:{rr_ratio:.1f}\n"
        message += "\n\n"
        
        # Disclaimer
        message += "This analysis is based on technical factors only and is not financial advice. Trade at your own risk."
        
        return message
    
    def extract_analysis_section(self, analysis_text):
        """Extract just the analysis part from the full text"""
        # Try to find a section labeled as Analysis
        analysis_match = re.search(r"(?i)analysis:?\s*(.*?)(?=\n\n|$)", analysis_text, re.DOTALL)
        if analysis_match:
            return analysis_match.group(1).strip()
        
        # If no explicit Analysis section, take the last paragraph
        paragraphs = analysis_text.split("\n\n")
        for paragraph in reversed(paragraphs):
            # Skip paragraphs that look like signal data
            if not re.search(r"(?i)(entry|stop loss|take profit|tp\d|sl):", paragraph):
                return paragraph.strip()
        
        # Fallback: return empty
        return ""
    
    def store_signal(self, user_id, trade_data, analysis_text):
        """
        Store a trade signal in the database
        
        Args:
            user_id (int): User's Telegram ID
            trade_data (dict): Trade data dictionary
            analysis_text (str): Full analysis text
            
        Returns:
            Signal: The created signal object or None if failed
        """
        if not Session:
            logger.error("Database session not available")
            return None
            
        session = Session()
        try:
            # Get database user ID from telegram ID
            user = session.query(User).filter(User.telegram_id == user_id).first()
            if not user:
                logger.error(f"User {user_id} not found in database")
                return None
                
            # Create signal
            signal = Signal(
                user_id=user.id,
                instrument=trade_data.get("instrument"),
                signal_type=trade_data.get("signal_type"),
                entry_price=trade_data.get("entry_price"),
                stop_loss=trade_data.get("stop_loss"),
                take_profit1=trade_data.get("take_profit1"),
                take_profit2=trade_data.get("take_profit2"),
                take_profit3=trade_data.get("take_profit3"),
                analysis_text=analysis_text,
                is_active=True,
                created_at=datetime.now()
            )
            
            session.add(signal)
            session.commit()
            logger.info(f"Signal created for user {user_id}: {signal.instrument} {signal.signal_type}")
            return signal
        except Exception as e:
            logger.error(f"Error storing signal: {str(e)}")
            session.rollback()
            return None
        finally:
            session.close()


#############################
# RISK MANAGEMENT AGENT #
#############################

class RiskManagementAgent:
    """Agent for calculating position sizes based on risk management principles"""
    
    def __init__(self):
        """Initialize the risk management agent"""
        # Define pip values and lot sizes for common instruments
        self.instrument_config = {
            # Format: "instrument": {"lot_multiplier": X, "pip_value": Y}
            # Major USD pairs
            "EURUSD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "GBPUSD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "AUDUSD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "NZDUSD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "USDCAD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "USDCHF": {"lot_multiplier": 100000, "pip_value": 0.0001},
            # JPY pairs
            "USDJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            "EURJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            "GBPJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            "AUDJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            "NZDJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            "CADJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            "CHFJPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            # Cross pairs
            "EURGBP": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "EURAUD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "EURNZD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "EURCAD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "EURCHF": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "GBPAUD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "GBPNZD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "GBPCAD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "GBPCHF": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "AUDCAD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "AUDCHF": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "AUDNZD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "NZDCAD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "NZDCHF": {"lot_multiplier": 100000, "pip_value": 0.0001},  # This was missing!
            "CADCHF": {"lot_multiplier": 100000, "pip_value": 0.0001},
            # Alternative symbol formats
            "NZD/CHF": {"lot_multiplier": 100000, "pip_value": 0.0001},  # Handle slash format
            "EUR/USD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "GBP/USD": {"lot_multiplier": 100000, "pip_value": 0.0001},
            "USD/JPY": {"lot_multiplier": 100000, "pip_value": 0.01},
            # Commodities
            "XAUUSD": {"lot_multiplier": 100, "pip_value": 0.1},  # Gold - 1 pip = $0.1
            "XAGUSD": {"lot_multiplier": 5000, "pip_value": 0.01},  # Silver
            # Crypto
            "BTCUSD": {"lot_multiplier": 1, "pip_value": 1.0},  # Bitcoin
            "ETHUSD": {"lot_multiplier": 1, "pip_value": 1.0},  # Ethereum
        }
        logger.info("RiskManagementAgent initialized")
    
    def get_latest_signal(self, telegram_id):
        """
        Get the latest active signal for a user
        
        Args:
            telegram_id (int): User's Telegram ID
            
        Returns:
            Signal: The latest signal or None if not found
        """
        if not Session:
            logger.error("Database session not available")
            return None
            
        session = Session()
        try:
            # Get user from database
            user = session.query(User).filter(User.telegram_id == telegram_id).first()
            if not user:
                logger.error(f"User {telegram_id} not found in database")
                return None
                
            # Get latest active signal
            signal = (session.query(Signal)
                     .filter(Signal.user_id == user.id, Signal.is_active == True)
                     .order_by(Signal.created_at.desc())
                     .first())
                     
            return signal
        except Exception as e:
            logger.error(f"Error getting latest signal: {str(e)}")
            return None
        finally:
            session.close()
    
    def calculate_lot_size(self, balance, risk_percent, entry_price=None, stop_loss=None, instrument="EURUSD"):
        """
        Calculate appropriate lot size based on account balance, entry/stop, and risk percentage
        
        Args:
            balance (float): Account balance in USD
            risk_percent (float): Risk percentage (1-5%)
            entry_price (float, optional): Entry price
            stop_loss (float, optional): Stop loss price
            instrument (str): Trading instrument
            
        Returns:
            dict: Dictionary with calculation details
        """
        try:
            # Validate inputs
            if not balance or balance <= 0:
                return {"error": "Invalid account balance"}
                
            if risk_percent <= 0 or risk_percent > 5:
                return {"error": "Risk percentage should be between 0.1 and 5%"}
                
            # Standardize instrument name
            instrument = instrument.upper()
            
            # Get configuration for this instrument
            config = self.instrument_config.get(instrument)
            if not config:
                # Default to EURUSD if instrument not found
                logger.warning(f"Instrument {instrument} not found in config, using EURUSD")
                config = self.instrument_config["EURUSD"]
                instrument = "EURUSD"
                
            # Calculate risk amount in dollars
            risk_amount = balance * (risk_percent / 100)
            
            # Calculate pips at risk if entry and stop loss provided
            pips_at_risk = None
            if entry_price is not None and stop_loss is not None:
                # Calculate pips based on instrument
                pip_size = config["pip_value"]
                price_difference = abs(entry_price - stop_loss)
                
                if instrument in ["BTCUSD", "ETHUSD", "BTC", "ETH"]:
                    # Crypto: 1 pip = 1 dollar
                    pips_at_risk = price_difference
                elif instrument in ["XAUUSD", "XAGUSD"]:
                    # Gold and Silver: 1 pip = 0.1, so divide by 0.1
                    pips_at_risk = price_difference / 0.1
                elif instrument in ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]:
                    # JPY pairs: 1 pip = 0.01
                    pips_at_risk = price_difference / 0.01
                else:
                    # Major forex pairs: 1 pip = 0.0001 (4th decimal place)
                    pips_at_risk = price_difference / 0.0001
                
                # Calculate lot size based on risk, pips, and instrument
                if instrument == "XAUUSD":
                    # For Gold: 1 pip = $1 per 0.1 lot, so $10 per 1.0 lot
                    pip_value_per_lot = 10.0
                    lot_size = risk_amount / (pips_at_risk * pip_value_per_lot)
                else:
                    # Standard forex calculation - simplified
                    if instrument in ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY"]:
                        # JPY pairs: 1 pip = $1 per 0.1 lot, so $10 per 1.0 lot
                        pip_value_per_lot = 10.0
                    else:
                        # Major forex pairs: 1 pip = $1 per 0.1 lot, so $10 per 1.0 lot
                        pip_value_per_lot = 10.0
                    lot_size = risk_amount / (pips_at_risk * pip_value_per_lot)
            else:
                # If no entry/stop, just provide a generic calculation
                # using a default 20 pips stop distance
                pips_at_risk = 20
                lot_multiplier = config["lot_multiplier"]
                pip_value = config["pip_value"] * lot_multiplier / 10
                lot_size = risk_amount / (pips_at_risk * pip_value * 10)
            
            # Format the result
            return {
                "account_balance": balance,
                "risk_percentage": risk_percent,
                "risk_amount": risk_amount,
                "instrument": instrument,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "pips_at_risk": pips_at_risk,
                "lot_size": round(lot_size, 2),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error calculating lot size: {str(e)}")
            return {"error": str(e)}
    
    def format_lot_calculation(self, result):
        """
        Format the lot size calculation for display
        
        Args:
            result (dict): Calculation result
            
        Returns:
            str: Formatted message
        """
        if "error" in result:
            return f"⚠️ *Error:* {result['error']}"
            
        # Extract values
        instrument = result.get("instrument", "Unknown")
        balance = result.get("account_balance", 0)
        risk_pct = result.get("risk_percentage", 0)
        risk_amount = result.get("risk_amount", 0)
        entry = result.get("entry_price")
        stop_loss = result.get("stop_loss")
        pips = result.get("pips_at_risk", 0)
        lot_size = result.get("lot_size", 0)
        
        # No need for pip display correction - calculation is now accurate
        
        # Create message
        message = f"🧮 *LOT SIZE CALCULATION* 🧮\n\n"
        
        message += f"*Instrument:* {instrument}\n\n"
        
        message += f"*Account Balance:* ${balance:,.2f}\n"
        message += f"*Risk:* {risk_pct:.1f}% (${risk_amount:,.2f})\n\n"
        
        if entry and stop_loss:
            message += f"*Entry Price:* {entry:,.5f}\n"
            message += f"*Stop Loss:* {stop_loss:,.5f}\n"
            message += f"*Stop Distance:* {pips:.1f} pips\n\n"
        else:
            message += f"*Default Stop Distance:* {pips:.1f} pips\n\n"
        
        message += f"*Recommended Lot Size:* {lot_size:.2f}\n\n"
        
        message += "Always verify calculations before trading."
        
        return message


#############################
# MAIN BOT INTEGRATION #
#############################

class GFXTradingBot:
    """Telegram bot for trading signals and education"""
    
    def __init__(self):
        """Initialize the bot"""
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN')
        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
            sys.exit(1)
            
        # Initialize agents
        self.user_manager = UserManagementAgent()
        self.signal_generator = SignalGenerationAgent()
        self.risk_manager = RiskManagementAgent()
        
        # Initialize database if needed
        self.init_database()
        
        logger.info("GFX Trading Bot initialized")
    
    def init_database(self):
        """Initialize database tables"""
        if not engine:
            logger.error("Database engine not available")
            return
            
        try:
            # Create tables
            Base.metadata.create_all(engine)
            logger.info("Database tables created successfully")
            
            # Initialize admin user if needed
            admin_id = DEFAULT_ADMIN_ID
            session = Session()
            admin = session.query(User).filter(User.telegram_id == admin_id).first()
            
            if not admin:
                admin = User(
                    telegram_id=admin_id,
                    username="admin",
                    is_admin=True,
                    is_authorized=True
                )
                session.add(admin)
                session.commit()
                logger.info(f"Admin user created with ID {admin_id}")
            
            session.close()
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    async def download_image(self, update, context):
        """Download an image from Telegram"""
        file = await context.bot.get_file(update.message.photo[-1].file_id)
        file_bytes = await file.download_as_bytearray()
        return bytes(file_bytes)
    
    async def handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        user = update.effective_user
        telegram_id = user.id
        
        # Create or update user in database
        self.user_manager.create_or_update_user(
            telegram_id=telegram_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        # Check if user is authorized
        is_authorized = self.user_manager.is_authorized(telegram_id)
        is_admin = self.user_manager.is_admin(telegram_id)
        
        if is_admin:
            # Admin welcome message
            await update.message.reply_text(
                f"👋 Welcome, Admin {user.first_name}!\n\n"
                f"You have full access to all bot features and admin commands.\n\n"
                f"Use /help to see available commands."
            )
        elif is_authorized:
            # Authorized user welcome message
            await update.message.reply_text(
                f"👋 Welcome back, {user.first_name}!\n\n"
                f"You have full access to GFX Trading Assistant. "
                f"Send a chart image for analysis or use /help to see available commands."
            )
        else:
            # Unauthorized user - show access request button
            keyboard = [
                [InlineKeyboardButton("Request Access", callback_data="request_access")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"👋 Hello, {user.first_name}!\n\n"
                f"To access GFX Trading Assistant's exclusive trading signals and tools, "
                f"please request access using the button below.",
                reply_markup=reply_markup
            )
    
    async def handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command"""
        user = update.effective_user
        telegram_id = user.id
        
        # Check if user is authorized
        is_authorized = self.user_manager.is_authorized(telegram_id)
        is_admin = self.user_manager.is_admin(telegram_id)
        
        if not is_authorized and not is_admin:
            # Unauthorized user help
            keyboard = [
                [InlineKeyboardButton("Request Access", callback_data="request_access")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                "⚠️ You need access to use this bot.\n\n"
                "Please request access using the button below.",
                reply_markup=reply_markup
            )
            return
            
        # Standard commands for all authorized users
        help_text = (
            "🔹 *GFX Trading Assistant* 🔹\n\n"
            "*Available Commands:*\n\n"
            "📊 *Chart Analysis:*\n"
            "Send any chart image to receive trading analysis\n\n"
            "🧮 *Lot Size Calculator:*\n"
            "/lot <balance> <risk%> - Calculate lot size based on recent signal\n"
            "/customlot <balance> <risk%> <entry> <stop> <instrument> - Custom calculation\n\n"
            "📈 *Performance Analytics:*\n"
            "/performance today - Today's trading performance\n"
            "/performance week - This week's performance\n"
            "/performance month - This month's performance\n"
            "/performance overall - All-time performance\n\n"
            "ℹ️ *General:*\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
        )
        
        # Add admin commands if applicable
        if is_admin:
            help_text += (
                "\n👑 *Admin Commands:*\n"
                "/admin - View admin dashboard\n"
                "/approve <user_id> - Approve user access\n"
                "/reject <user_id> - Reject user access\n"
                "/list - List pending access requests\n"
            )
        
        await update.message.reply_text(
            help_text,
            parse_mode=telegram.constants.ParseMode.MARKDOWN
        )
    
    async def handle_lot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /lot command"""
        user = update.effective_user
        telegram_id = user.id
        chat_id = update.effective_chat.id
        
        # Check if user is authorized
        if not self.user_manager.is_authorized(telegram_id):
            await update.message.reply_text(
                "⚠️ You need access to use this command.\n\n"
                "Use /start to request access."
            )
            return
            
        # Parse arguments
        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "⚠️ Please provide your account balance and risk percentage.\n\n"
                "*Example:* /lot 1000 2\n\n"
                "This would calculate lot size for $1000 balance with 2% risk.",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return
            
        try:
            # Parse balance and risk
            balance = float(args[0])
            risk_percent = float(args[1])
            
            # Get latest signal
            signal = self.risk_manager.get_latest_signal(telegram_id)
            
            if not signal:
                await update.message.reply_text(
                    "⚠️ No recent trading signals found.\n\n"
                    "Please send a chart image for analysis first, or use "
                    "/customlot to specify entry and stop loss manually."
                )
                return
                
            # Calculate lot size
            result = self.risk_manager.calculate_lot_size(
                balance=balance,
                risk_percent=risk_percent,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                instrument=signal.instrument
            )
            
            # Format and send response
            response = self.risk_manager.format_lot_calculation(result)
            await update.message.reply_text(
                response,
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid balance or risk percentage. Please provide numeric values.\n\n"
                "*Example:* /lot 1000 2",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Error handling lot command: {str(e)}")
            await update.message.reply_text(
                f"⚠️ Error calculating lot size: {str(e)}"
            )
    
    async def handle_custom_lot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /customlot command"""
        user = update.effective_user
        telegram_id = user.id
        
        # Check if user is authorized
        if not self.user_manager.is_authorized(telegram_id):
            await update.message.reply_text(
                "⚠️ You need access to use this command.\n\n"
                "Use /start to request access."
            )
            return
            
        # Parse arguments
        args = context.args
        if len(args) < 4:
            await update.message.reply_text(
                "⚠️ Please provide all required parameters.\n\n"
                "*Example:* /customlot 1000 2 1.08650 1.08500 EURUSD\n\n"
                "Parameters: balance risk% entry_price stop_loss [instrument]",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return
            
        try:
            # Parse arguments
            balance = float(args[0])
            risk_percent = float(args[1])
            entry_price = float(args[2])
            stop_loss = float(args[3])
            instrument = args[4].upper() if len(args) > 4 else "EURUSD"
            
            # Calculate lot size
            result = self.risk_manager.calculate_lot_size(
                balance=balance,
                risk_percent=risk_percent,
                entry_price=entry_price,
                stop_loss=stop_loss,
                instrument=instrument
            )
            
            # Format and send response
            response = self.risk_manager.format_lot_calculation(result)
            await update.message.reply_text(
                response,
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid parameters. Please provide numeric values.\n\n"
                "*Example:* /customlot 1000 2 1.08650 1.08500 EURUSD",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
        except Exception as e:
            logger.error(f"Error handling custom lot command: {str(e)}")
            await update.message.reply_text(
                f"⚠️ Error calculating lot size: {str(e)}"
            )
    
    async def handle_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /admin command"""
        user = update.effective_user
        telegram_id = user.id
        
        # Check if user is admin
        if not self.user_manager.is_admin(telegram_id):
            await update.message.reply_text(
                "⚠️ You don't have admin privileges."
            )
            return
            
        # Get pending access requests
        pending_requests = self.user_manager.get_pending_requests()
        
        if not pending_requests:
            await update.message.reply_text(
                "👑 *Admin Dashboard*\n\n"
                "No pending access requests at this time.",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return
            
        # Format message with pending requests
        message = "👑 *Admin Dashboard*\n\n"
        message += "*Pending Access Requests:*\n\n"
        
        for i, req in enumerate(pending_requests, 1):
            username = req.get("username", "Unknown")
            first_name = req.get("first_name", "")
            last_name = req.get("last_name", "")
            telegram_id = req.get("telegram_id")
            requested_at = req.get("requested_at")
            
            message += f"{i}. "
            if username:
                message += f"@{username} "
            if first_name or last_name:
                message += f"({first_name} {last_name}) "
            message += f"- ID: {telegram_id}\n"
            message += f"   Requested: {requested_at}\n"
            message += f"   */approve {telegram_id}* | */reject {telegram_id}*\n\n"
        
        message += "Use */list* to see this list again."
        
        await update.message.reply_text(
            message,
            parse_mode=telegram.constants.ParseMode.MARKDOWN
        )
    
    async def handle_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /list command for admins"""
        telegram_id = update.effective_user.id
        
        if not self.user_manager.is_admin(telegram_id):
            await update.message.reply_text("⚠️ You don't have admin privileges.")
            return
            
        try:
            # Get all users from database
            if Session:
                session = Session()
                try:
                    users = session.query(User).all()
                    
                    if not users:
                        await update.message.reply_text("📝 No users registered yet.")
                        return
                        
                    message = "👥 **REGISTERED USERS**\n\n"
                    
                    for user in users:
                        # Create user header with username and ID
                        if user.username:
                            user_header = f"**@{user.username}** (ID: `{user.telegram_id}`)"
                        else:
                            user_header = f"**User ID:** `{user.telegram_id}`"
                        
                        user_info = f"{user_header}\n"
                        
                        if user.first_name or user.last_name:
                            name = f"{user.first_name or ''} {user.last_name or ''}".strip()
                            user_info += f"**Name:** {name}\n"
                        
                        user_info += f"**Admin:** {'✅' if user.is_admin else '❌'}\n"
                        user_info += f"**Authorized:** {'✅' if user.is_authorized else '❌'}\n"
                        user_info += f"**Joined:** {user.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                        user_info += "─" * 30 + "\n\n"
                        
                        message += user_info
                        
                    await update.message.reply_text(message, parse_mode='Markdown')
                    
                finally:
                    session.close()
            else:
                await update.message.reply_text("❌ Database connection error.")
                
        except Exception as e:
            logger.error(f"Error in list command: {str(e)}")
            await update.message.reply_text("❌ An error occurred while fetching users.")
    
    async def handle_approve(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /approve command"""
        user = update.effective_user
        telegram_id = user.id
        
        # Check if user is admin
        if not self.user_manager.is_admin(telegram_id):
            await update.message.reply_text(
                "⚠️ You don't have admin privileges."
            )
            return
            
        # Parse arguments
        args = context.args
        if not args:
            await update.message.reply_text(
                "⚠️ Please specify the user ID to approve.\n\n"
                "*Example:* /approve 123456789",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return
            
        try:
            # Parse user ID
            user_id = int(args[0])
            
            # Approve the request
            success = self.user_manager.approve_request(user_id, telegram_id)
            
            if success:
                # Notify the approved user
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=(
                            "✅ *Access Approved*\n\n"
                            "Your request to access GFX Trading Assistant has been approved!\n\n"
                            "You now have full access to all features. Send a chart image for analysis "
                            "or use /help to see available commands."
                        ),
                        parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Error notifying approved user: {str(e)}")
                
                # Respond to admin
                await update.message.reply_text(
                    f"✅ User {user_id} has been approved successfully."
                )
            else:
                await update.message.reply_text(
                    f"⚠️ Failed to approve user {user_id}. No pending request found."
                )
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid user ID. Please provide a numeric ID."
            )
        except Exception as e:
            logger.error(f"Error handling approve command: {str(e)}")
            await update.message.reply_text(
                f"⚠️ Error approving user: {str(e)}"
            )
    
    async def handle_reject(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /reject command"""
        user = update.effective_user
        telegram_id = user.id
        
        # Check if user is admin
        if not self.user_manager.is_admin(telegram_id):
            await update.message.reply_text(
                "⚠️ You don't have admin privileges."
            )
            return
            
        # Parse arguments
        args = context.args
        if not args:
            await update.message.reply_text(
                "⚠️ Please specify the user ID to reject.\n\n"
                "*Example:* /reject 123456789",
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            return
            
        try:
            # Parse user ID
            user_id = int(args[0])
            
            # Reject the request
            success = self.user_manager.reject_request(user_id, telegram_id)
            
            if success:
                # Notify the rejected user
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=(
                            "❌ *Access Denied*\n\n"
                            "Your request to access GFX Trading Assistant has been denied.\n\n"
                            "If you believe this is a mistake, please contact an administrator."
                        ),
                        parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Error notifying rejected user: {str(e)}")
                
                # Respond to admin
                await update.message.reply_text(
                    f"✅ User {user_id} has been rejected successfully."
                )
            else:
                await update.message.reply_text(
                    f"⚠️ Failed to reject user {user_id}. No pending request found."
                )
        except ValueError:
            await update.message.reply_text(
                "⚠️ Invalid user ID. Please provide a numeric ID."
            )
        except Exception as e:
            logger.error(f"Error handling reject command: {str(e)}")
            await update.message.reply_text(
                f"⚠️ Error rejecting user: {str(e)}"
            )

    async def handle_requests(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /requests command to show pending access requests"""
        telegram_id = update.effective_user.id
        
        if not self.user_manager.is_admin(telegram_id):
            await update.message.reply_text("⚠️ You don't have admin privileges.")
            return
            
        try:
            pending_requests = self.user_manager.get_pending_requests()
            
            if not pending_requests:
                await update.message.reply_text("📝 No pending access requests.")
                return
                
            message = "📋 **PENDING ACCESS REQUESTS**\n\n"
            
            for req in pending_requests:
                # Create a user identifier that shows both username and ID
                if req['username']:
                    user_header = f"**@{req['username']}** (ID: `{req['telegram_id']}`)"
                else:
                    user_header = f"**User ID:** `{req['telegram_id']}`"
                
                user_info = f"{user_header}\n"
                
                if req['first_name'] or req['last_name']:
                    name = f"{req['first_name'] or ''} {req['last_name'] or ''}".strip()
                    user_info += f"**Name:** {name}\n"
                user_info += f"**Requested:** {req['requested_at']}\n"
                user_info += f"**Commands:** `/approve {req['telegram_id']}` | `/reject {req['telegram_id']}`\n"
                user_info += "─" * 30 + "\n\n"
                
                message += user_info
                
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in requests command: {str(e)}")
            await update.message.reply_text("❌ An error occurred while fetching requests.")

    async def handle_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /performance command with time periods"""
        user = update.effective_user
        telegram_id = user.id
        
        # Log that the command was received
        logger.info(f"Performance command received from user {telegram_id}")
        
        # Parse command arguments manually since context.args might be empty
        command_text = update.message.text.strip()
        parts = command_text.split()
        args = parts[1:] if len(parts) > 1 else []
        
        # Send immediate response to confirm command received
        await update.message.reply_text("📊 Processing your performance request...")
        
        # Check if user is authorized
        if not self.user_manager.is_authorized(telegram_id) and not self.user_manager.is_admin(telegram_id):
            await update.message.reply_text(
                "⚠️ You don't have permission to use this command."
            )
            return
        
        # Default to overall if no period specified
        period = args[0] if args else "overall"
        valid_periods = ["today", "week", "month", "overall"]
        
        if period not in valid_periods:
            await update.message.reply_text(
                "⚠️ Invalid period. Use: today, week, month, or overall\n\n"
                "*Example:* /performance today"
            )
            return
        
        try:
            from datetime import datetime, timedelta
            
            session = Session()
            logger.info(f"Performance command called by user {telegram_id} for period: {period}")
            
            # Get cutoff date based on period
            now = datetime.now()
            if period == "today":
                cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
                period_name = "Today's"
            elif period == "week":
                cutoff = now - timedelta(days=7)
                period_name = "This Week's"
            elif period == "month":
                cutoff = now - timedelta(days=30)
                period_name = "This Month's"
            else:
                cutoff = None
                period_name = "Overall"
            
            # Query user's signals
            user = session.query(User).filter(User.telegram_id == telegram_id).first()
            if not user:
                await update.message.reply_text(
                    "⚠️ User not found in database. Please send a chart first to create your profile."
                )
                session.close()
                return
            
            logger.info(f"Found user {user.id} for telegram_id {telegram_id}")
            query = session.query(Signal).filter(Signal.user_id == user.id)
            
            if cutoff:
                query = query.filter(Signal.created_at >= cutoff)
            
            signals = query.order_by(Signal.created_at.desc()).all()
            
            if not signals:
                await update.message.reply_text(
                    f"📊 No trading signals found for {period_name.lower()} period."
                )
                session.close()
                return
            
            # Calculate performance metrics
            total_signals = len(signals)
            winning_trades = 0
            losing_trades = 0
            total_pips = 0
            trade_details = []
            
            # TP breakdown counters
            tp1_hits = 0
            tp2_hits = 0
            tp3_hits = 0
            
            for signal in signals:
                # Determine pip multiplier
                instrument = signal.instrument.upper()
                if 'XAU' in instrument or 'GOLD' in instrument:
                    pip_multiplier = 10
                elif 'JPY' in instrument:
                    pip_multiplier = 100
                elif 'BTC' in instrument or 'ETH' in instrument:
                    pip_multiplier = 1
                else:
                    pip_multiplier = 10000
                
                signal_pips = 0
                status = "Active"
                
                if signal.sl_hit:
                    losing_trades += 1
                    signal_pips = -(abs(signal.entry_price - signal.stop_loss) * pip_multiplier)
                    status = "SL Hit"
                elif signal.tp1_hit or signal.tp2_hit or signal.tp3_hit:
                    winning_trades += 1
                    tp_level = "TP3" if signal.tp3_hit else ("TP2" if signal.tp2_hit else "TP1")
                    tp_price = signal.take_profit3 if signal.tp3_hit else (signal.take_profit2 if signal.tp2_hit else signal.take_profit1)
                    
                    if signal.signal_type == 'BUY':
                        signal_pips = (tp_price - signal.entry_price) * pip_multiplier
                    else:
                        signal_pips = (signal.entry_price - tp_price) * pip_multiplier
                    
                    status = f"{tp_level} Hit"
                
                total_pips += signal_pips
                
                trade_details.append({
                    'id': signal.id,
                    'instrument': signal.instrument,
                    'type': signal.signal_type,
                    'entry': signal.entry_price,
                    'status': status,
                    'pips': signal_pips,
                    'date': signal.created_at.strftime('%m/%d %H:%M')
                })
            
            # Calculate win rate
            completed_trades = winning_trades + losing_trades
            win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
            
            # Format response
            message = f"📊 *{period_name} Performance*\n\n"
            message += f"📈 *Summary:*\n"
            message += f"• Total Signals: {total_signals}\n"
            message += f"• Winning Trades: {winning_trades}\n"
            message += f"• Losing Trades: {losing_trades}\n"
            message += f"• Win Rate: {win_rate:.1f}%\n"
            message += f"• Total Pips: {total_pips:+.1f}\n\n"
            
            if trade_details:
                message += f"🔍 *Trade Details:*\n"
                for trade in trade_details[:10]:  # Show latest 10 trades
                    pips_str = f"{trade['pips']:+.1f}" if trade['pips'] != 0 else "0.0"
                    message += f"#{trade['id']} {trade['instrument']} {trade['type']} - {trade['status']} ({pips_str} pips) - {trade['date']}\n"
                
                if len(trade_details) > 10:
                    message += f"\n... and {len(trade_details) - 10} more trades"
            
            await update.message.reply_text(
                message,
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            
            session.close()
            
        except Exception as e:
            logger.error(f"Error handling performance command: {str(e)}")
            await update.message.reply_text(
                f"⚠️ Error retrieving performance data: {str(e)}"
            )

    async def handle_lot_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the interactive lot size calculation conversation"""
        user_text = update.message.text.strip()
        step = context.user_data.get('step')
        
        try:
            if step == 'balance':
                # Parse account balance
                balance = float(user_text.replace(',', '').replace('$', ''))
                context.user_data['balance'] = balance
                context.user_data['step'] = 'risk'
                
                await update.message.reply_text(
                    f"✅ Account Balance: ${balance:,.2f}\n\n"
                    "📊 **Step 2 of 2: Risk Percentage**\n\n"
                    "What percentage of your account are you willing to risk on this trade?\n\n"
                    "Please reply with just the number (e.g., 2 for 2% risk)\n"
                    "💡 *Recommended: 1-3% per trade*",
                    parse_mode=telegram.constants.ParseMode.MARKDOWN
                )
                
            elif step == 'risk':
                # Parse risk percentage
                risk_percent = float(user_text.replace('%', ''))
                balance = context.user_data['balance']
                signal_id = context.user_data['signal_id']
                
                # Get the signal details
                signal = None
                if Session:
                    session = Session()
                    try:
                        signal = session.query(Signal).filter(Signal.id == signal_id).first()
                    finally:
                        session.close()
                
                if signal:
                    # Calculate lot size
                    result = self.risk_manager.calculate_lot_size(
                        balance=balance,
                        risk_percent=risk_percent,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        instrument=signal.instrument
                    )
                    
                    # Format and send response
                    response = self.risk_manager.format_lot_calculation(result)
                    await update.message.reply_text(
                        f"✅ Risk Percentage: {risk_percent}%\n\n{response}",
                        parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )
                else:
                    await update.message.reply_text(
                        "❌ Signal not found. Please try again with a new chart analysis."
                    )
                
                # Clear conversation state
                context.user_data.clear()
                
        except ValueError:
            await update.message.reply_text(
                "❌ Please enter a valid number. Try again:"
            )
        except Exception as e:
            logger.error(f"Error in lot conversation: {str(e)}")
            await update.message.reply_text(
                "❌ An error occurred. Please try again with a new calculation."
            )
            context.user_data.clear()
    
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline buttons"""
        query = update.callback_query
        data = query.data
        user = update.effective_user
        telegram_id = user.id
        
        # Answer the callback query to stop the loading animation
        await query.answer()
        
        if data == "request_access":
            # Create or update user in database
            self.user_manager.create_or_update_user(
                telegram_id=telegram_id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name
            )
            
            # Submit access request
            success = self.user_manager.request_access(telegram_id)
            
            if success:
                # Update the message
                await query.edit_message_text(
                    f"✅ Your access request has been submitted successfully!\n\n"
                    f"Our administrators will review your request as soon as possible. "
                    f"You'll receive a notification when your access is approved."
                )
                
                # Notify admins
                admin_ids = self.user_manager.admin_ids
                for admin_id in admin_ids:
                    try:
                        # Create approval/rejection buttons
                        keyboard = [
                            [
                                InlineKeyboardButton("✅ Approve", callback_data=f"approve_{telegram_id}"),
                                InlineKeyboardButton("❌ Reject", callback_data=f"reject_{telegram_id}")
                            ]
                        ]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        await context.bot.send_message(
                            chat_id=admin_id,
                            text=(
                                f"🔔 *New Access Request*\n\n"
                                f"User: {user.first_name or ''} {user.last_name or ''}\n"
                                f"Username: @{user.username or 'None'}\n"
                                f"User ID: {telegram_id}\n\n"
                                f"Use the buttons below to approve or reject."
                            ),
                            parse_mode=telegram.constants.ParseMode.MARKDOWN,
                            reply_markup=reply_markup
                        )
                    except Exception as e:
                        logger.error(f"Error notifying admin {admin_id}: {str(e)}")
            else:
                await query.edit_message_text(
                    f"⚠️ There was an error submitting your access request. Please try again later."
                )
        elif data.startswith("approve_"):
            # Check if user is admin
            if not self.user_manager.is_admin(telegram_id):
                await query.edit_message_text(
                    "⚠️ You don't have admin privileges."
                )
                return
                
            # Extract user ID
            user_id = int(data.split("_")[1])
            
            # Approve the request
            success = self.user_manager.approve_request(user_id, telegram_id)
            
            if success:
                # Update the message
                await query.edit_message_text(
                    f"✅ User {user_id} has been approved successfully."
                )
                
                # Notify the approved user
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=(
                            "✅ *Access Approved*\n\n"
                            "Your request to access GFX Trading Assistant has been approved!\n\n"
                            "You now have full access to all features. Send a chart image for analysis "
                            "or use /help to see available commands."
                        ),
                        parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Error notifying approved user: {str(e)}")
            else:
                await query.edit_message_text(
                    f"⚠️ Failed to approve user {user_id}. No pending request found."
                )
        elif data.startswith("reject_"):
            # Check if user is admin
            if not self.user_manager.is_admin(telegram_id):
                await query.edit_message_text(
                    "⚠️ You don't have admin privileges."
                )
                return
                
            # Extract user ID
            user_id = int(data.split("_")[1])
            
            # Reject the request
            success = self.user_manager.reject_request(user_id, telegram_id)
            
            if success:
                # Update the message
                await query.edit_message_text(
                    f"❌ User {user_id} has been rejected."
                )
                
                # Notify the rejected user
                try:
                    await context.bot.send_message(
                        chat_id=user_id,
                        text=(
                            "❌ *Access Denied*\n\n"
                            "Your request to access GFX Trading Assistant has been denied.\n\n"
                            "If you believe this is a mistake, please contact an administrator."
                        ),
                        parse_mode=telegram.constants.ParseMode.MARKDOWN
                    )
                except Exception as e:
                    logger.error(f"Error notifying rejected user: {str(e)}")
            else:
                await query.edit_message_text(
                    f"⚠️ Failed to reject user {user_id}. No pending request found."
                )

    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular messages"""
        if not update.message or not update.effective_user:
            return
            
        user = update.effective_user
        telegram_id = user.id
        
        # Create or update user in database
        self.user_manager.create_or_update_user(
            telegram_id=telegram_id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name
        )
        
        # Check if user is authorized
        if not self.user_manager.is_authorized(telegram_id) and not self.user_manager.is_admin(telegram_id):
            # Only respond if message contains certain keywords
            message_text = update.message.text or ""
            if re.search(r"access|request|join|how|help|start", message_text, re.IGNORECASE):
                keyboard = [
                    [InlineKeyboardButton("Request Access", callback_data="request_access")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    f"👋 Hello, {user.first_name}!\n\n"
                    f"To access GFX Trading Assistant, please request access using the button below.",
                    reply_markup=reply_markup
                )
            return
            
        # Handle photo messages (chart analysis)
        if update.message.photo:
            # Send typing status
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
            
            try:
                # Download the image
                image_bytes = await self.download_image(update, context)
                
                # Analyze the chart
                analysis_result = await self.signal_generator.analyze_chart(image_bytes)
                
                if not analysis_result.get("success", False):
                    await update.message.reply_text(
                        f"⚠️ Error analyzing chart: {analysis_result.get('error', 'Unknown error')}\n\n"
                        f"Please try again with a clearer chart image."
                    )
                    return
                    
                # Extract data
                analysis_text = analysis_result.get("analysis_text", "")
                trade_data = analysis_result.get("trade_data")
                formatted_signal = analysis_result.get("formatted_signal", "")
                
                # Save signal to database if trade data is valid
                if trade_data and trade_data.get("signal_type") and trade_data.get("entry_price"):
                    try:
                        # Get or create user
                        session = Session()
                        user = session.query(User).filter(User.telegram_id == telegram_id).first()
                        
                        if user:
                            # Create new signal record
                            new_signal = Signal(
                                user_id=user.id,
                                instrument=trade_data.get("instrument", "XAUUSD"),
                                signal_type=trade_data.get("signal_type"),
                                entry_price=trade_data.get("entry_price"),
                                stop_loss=trade_data.get("stop_loss"),
                                take_profit1=trade_data.get("take_profit1"),
                                take_profit2=trade_data.get("take_profit2"),
                                take_profit3=trade_data.get("take_profit3"),
                                analysis_text=analysis_text[:1000],  # Limit text length
                                is_active=True
                            )
                            
                            session.add(new_signal)
                            session.commit()
                            logger.info(f"Saved signal {new_signal.id} for user {telegram_id}")
                            
                        session.close()
                        
                    except Exception as e:
                        logger.error(f"Error saving signal to database: {str(e)}")
                        if 'session' in locals():
                            session.close()
                
                if not trade_data:
                    await update.message.reply_text(
                        "⚠️ Failed to extract trade data from the analysis.\n\n"
                        "Please try again with a clearer chart image."
                    )
                    return
                    
                # Store the signal in the database
                signal = self.signal_generator.store_signal(telegram_id, trade_data, analysis_text)
                
                # Send the formatted signal
                sent_message = await update.message.reply_text(
                    formatted_signal,
                    parse_mode=telegram.constants.ParseMode.MARKDOWN
                )
                

                
            except Exception as e:
                logger.error(f"Error handling chart analysis: {str(e)}")
                await update.message.reply_text(
                    f"⚠️ An error occurred while analyzing your chart: {str(e)}\n\n"
                    f"Please try again later."
                )
        
        # Handle text messages (could add a trading education/Q&A feature here)
        elif update.message.text:
            pass  # We could add a trading education feature here in the future
    
    async def handle_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors in the dispatcher"""
        logger.error(f"Update {update} caused error {context.error}")
        
        # Send error message to the user if possible
        if update and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ An error occurred while processing your request. Please try again later."
            )
    
    async def run(self):
        """Run the bot"""
        # Create the Application
        application = Application.builder().token(self.token).build()
        
        # Add command handlers
        application.add_handler(CommandHandler("start", self.handle_start))
        application.add_handler(CommandHandler("help", self.handle_help))
        application.add_handler(CommandHandler("lot", self.handle_lot))
        application.add_handler(CommandHandler("customlot", self.handle_custom_lot))
        application.add_handler(CommandHandler("performance", self.handle_performance))
        
        # Add admin command handlers
        application.add_handler(CommandHandler("admin", self.handle_admin))
        application.add_handler(CommandHandler("list", self.handle_list))
        application.add_handler(CommandHandler("approve", self.handle_approve))
        application.add_handler(CommandHandler("reject", self.handle_reject))
        application.add_handler(CommandHandler("requests", self.handle_requests))
        
        # Add callback query handler
        application.add_handler(CallbackQueryHandler(self.handle_callback_query))
        
        # Add message handler for all other messages
        application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        application.add_error_handler(self.handle_error)
        
        # Start the bot
        await application.initialize()
        await application.start()
        
        logger.info("GFX Trading Bot is running!")
        
        # Run the bot until stopped
        try:
            # Start polling in a way that keeps the bot running
            await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            
            # Keep the process running indefinitely
            while True:
                await asyncio.sleep(1000)  # Sleep for a long time to keep the bot running
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by keyboard interrupt")
        except Exception as e:
            logger.error(f"Error in bot polling: {e}")
        finally:
            # Stop the bot when we're done
            await application.stop()


async def main():
    """Run the bot"""
    # Create and run the bot
    bot = GFXTradingBot()
    await bot.run()


if __name__ == "__main__":
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)