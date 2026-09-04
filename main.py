#!/usr/bin/env python3
"""
Hybrid Trading Bot - Main Entry Point
ML signals + Claude AI analysis + Discord approval
"""

import os
import sys
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

# Create data directories before logging opens its file handler
Path('data').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from src.discord_bot import TradingBot
from src.database import Database
from src.ml_engine import TradingSignalEngine

def main():
    """Main entry point"""
    print("=" * 60)
    print("🚀 HYBRID TRADING BOT - Starting")
    print("=" * 60)
    
    # Validate configuration
    # CHANNEL_ID is optional - leave it unset to have alerts DM'd to USER_ID
    required_env = ['DISCORD_TOKEN', 'USER_ID']
    missing = [var for var in required_env if not os.getenv(var)]
    
    if missing:
        logger.error(f"❌ Missing environment variables: {', '.join(missing)}")
        logger.error("Copy .env.example to .env and fill in your values")
        sys.exit(1)
    
    # Create data directories
    Path('data').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("✅ Configuration validated")
    logger.info(f"   Discord Token: {'*' * 40}...{os.getenv('DISCORD_TOKEN')[-4:]}")
    logger.info(f"   Delivery: {'channel ' + os.getenv('CHANNEL_ID') if (os.getenv('CHANNEL_ID') or '').strip('0 ') else 'direct message'}")
    logger.info(f"   User ID: {os.getenv('USER_ID')}")
    
    # Initialize database
    logger.info("📊 Initializing database...")
    db = Database()
    
    # Initialize ML engine
    logger.info("🧠 Initializing ML engine...")
    engine = TradingSignalEngine()
    
    # Initialize Discord bot
    logger.info("💬 Starting Discord bot...")
    bot = TradingBot(engine=engine, db=db)
    
    try:
        bot.run(os.getenv('DISCORD_TOKEN'))
    except KeyboardInterrupt:
        logger.info("⏸️  Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
