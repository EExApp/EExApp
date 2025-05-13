#!/usr/bin/env python3
import os
import sys
import signal
import logging
from eexapp import EExApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler("eexapp.log", maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EExAppMain")

def signal_handler(sig, frame):
    """Handle termination signals"""
    logger.info("Received termination signal. Shutting down...")
    if 'xapp' in globals():
        xapp.stop()
    sys.exit(0)

def main():
    """Main function to run the xApp"""
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize xApp
        logger.info("Initializing EExApp...")
        global xapp
        xapp = EExApp()
        
        # Start xApp
        logger.info("Starting EExApp...")
        xapp.start()
        
        # Keep the main thread alive
        while True:
            signal.pause()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        if 'xapp' in globals():
            xapp.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 