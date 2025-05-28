"""
Database models for GFX Trading Bot
"""

import os
import logging
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean,
    DateTime, ForeignKey, Text, and_, or_, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    filename="database_models.log"
)
logger = logging.getLogger(__name__)

# Get database URL from environment or use SQLite
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///gfx_trading_bot.db")

# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Constants
ADMIN_USER_ID = 2025152767  # @BigGTrades - Admin user Telegram ID

class User(Base):
    __tablename__ = 'bot_users'
    
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)
    is_admin = Column(Boolean, default=False)
    is_authorized = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    signals = relationship("Signal", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, telegram_id={self.telegram_id}, username={self.username})>"

class Signal(Base):
    __tablename__ = 'bot_signals'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('bot_users.id'), nullable=False)
    instrument = Column(String(50), nullable=False)
    signal_type = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit1 = Column(Float, nullable=True)
    take_profit2 = Column(Float, nullable=True)
    take_profit3 = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    tp1_hit = Column(Boolean, default=False)
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    sl_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    closed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="signals")
    
    def __repr__(self):
        return f"<Signal(id={self.id}, instrument={self.instrument}, signal_type={self.signal_type})>"

def is_user_authorized(user_id):
    """Check if a user is authorized to use the bot"""
    try:
        with Session() as session:
            user = session.query(User).filter(User.telegram_id == user_id).first()
            return user is not None and user.is_authorized
    except Exception as e:
        logger.error(f"Error checking user authorization: {e}")
        return False

def is_admin_user(user_id):
    """Check if a user is an admin"""
    # Simple admin check: either defined admin ID or in the database
    if user_id == ADMIN_USER_ID:
        return True
        
    try:
        with Session() as session:
            user = session.query(User).filter(
                User.telegram_id == user_id,
                User.is_admin == True
            ).first()
            return user is not None
    except Exception as e:
        logger.error(f"Error checking admin status: {e}")
        return False

def get_or_create_user(telegram_id, username=None, first_name=None, last_name=None):
    """Get or create a user in the database"""
    try:
        with Session() as session:
            user = session.query(User).filter(User.telegram_id == telegram_id).first()
            
            if not user:
                # Create new user
                user = User(
                    telegram_id=telegram_id,
                    username=username,
                    first_name=first_name,
                    last_name=last_name,
                    is_admin=(telegram_id == ADMIN_USER_ID)  # Set admin status based on ID
                )
                session.add(user)
                session.commit()
                logger.info(f"Created new user: {telegram_id} ({username})")
            else:
                # Update user details if they've changed
                if ((username and user.username != username) or
                    (first_name and user.first_name != first_name) or
                    (last_name and user.last_name != last_name)):
                    
                    user.username = username or user.username
                    user.first_name = first_name or user.first_name
                    user.last_name = last_name or user.last_name
                    user.last_active = datetime.now()
                    session.commit()
                    logger.info(f"Updated user details: {telegram_id} ({username})")
            
            return user
    except Exception as e:
        logger.error(f"Error getting/creating user: {e}")
        return None

def create_all_tables():
    """Create all necessary database tables"""
    Base.metadata.create_all(engine)
    logger.info("Created all database tables")

def initialize_admin_user():
    """Initialize the admin user in the database"""
    try:
        with Session() as session:
            admin = session.query(User).filter(User.telegram_id == ADMIN_USER_ID).first()
            
            if not admin:
                admin = User(
                    telegram_id=ADMIN_USER_ID,
                    username="BigGTrades",
                    first_name="GFX",
                    last_name="Admin",
                    is_admin=True,
                    is_authorized=True
                )
                session.add(admin)
                session.commit()
                logger.info(f"Created admin user: {ADMIN_USER_ID}")
            elif not admin.is_admin:
                admin.is_admin = True
                admin.is_authorized = True
                session.commit()
                logger.info(f"Updated admin status for user: {ADMIN_USER_ID}")
                
    except Exception as e:
        logger.error(f"Error initializing admin user: {e}")

# Initialize database
def init_database():
    """Initialize the database and tables"""
    create_all_tables()
    initialize_admin_user()
    logger.info("Database initialized")

# Initialize database when the module is imported
if __name__ == "__main__":
    init_database()
else:
    # Initialize the database when the module is imported
    try:
        create_all_tables()
        initialize_admin_user()
        logger.info("Database initialized during import")
    except Exception as e:
        logger.error(f"Error initializing database during import: {e}")