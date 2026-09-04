#!/usr/bin/env python3
"""
Claude AI Analyzer
Provides market analysis, risk assessment, and portfolio review
"""

import os
import logging
from anthropic import Anthropic, AsyncAnthropic
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Current-generation model. Thinking is on by default on Opus 5; max_tokens must
# leave room for it, so the prompts (not the cap) constrain output length.
MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")


def _text(response) -> str:
    """Join the text blocks of a response, skipping thinking blocks."""
    if getattr(response, "stop_reason", None) == "refusal":
        return "Claude declined to answer that request."
    return "\n".join(b.text for b in response.content if b.type == "text").strip()

class ClaudeAnalyzer:
    def __init__(self):
        api_key = os.getenv('CLAUDE_API_KEY')
        
        if not api_key:
            logger.warning("⚠️ CLAUDE_API_KEY not set - Claude analysis disabled")
            self.enabled = False
            return
        
        self.client = AsyncAnthropic(api_key=api_key)
        # ClaudeAnalyzerSync needs a blocking client of its own.
        self.sync_client = Anthropic(api_key=api_key)
        self.enabled = True
        self.conversation_history = []
    
    async def daily_market_analysis(self):
        """Claude analyzes daily market conditions"""
        if not self.enabled:
            return "Claude API not configured. Run `!daily_brief` after adding CLAUDE_API_KEY."
        
        prompt = f"""
        Today is {datetime.now().strftime('%A, %B %d, %Y')}.
        
        Provide a brief market analysis (2-3 sentences) covering:
        1. Current market sentiment (bullish/bearish/neutral)
        2. Any major economic events happening today
        3. Recommendation for trading activity (active/cautious/avoid)
        
        Be concise and actionable for a retail trader.
        """
        
        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = _text(response)
            logger.info("✅ Daily brief generated")
            return analysis
        
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            return f"Error analyzing market: {str(e)}"
    
    async def analyze_portfolio_risk(self, positions):
        """Claude assesses portfolio risk"""
        if not self.enabled:
            return "Claude API not configured."
        
        positions_str = json.dumps(positions, indent=2) if positions else "No positions"
        
        prompt = f"""
        Current portfolio positions:
        {positions_str}
        
        Provide a brief risk assessment (2-3 sentences) covering:
        1. Concentration risk (too much in one sector?)
        2. Position size appropriateness
        3. Recommended risk mitigation
        
        Be direct and actionable.
        """
        
        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = _text(response)
            logger.info("✅ Risk assessment generated")
            return analysis
        
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            return f"Error assessing risk: {str(e)}"
    
    async def signal_explanation(self, symbol, signal, price, confidence):
        """Claude explains why a signal was generated"""
        if not self.enabled:
            return f"Signal: {symbol} - {signal_type} at ${price:.2f}"
        
        signal_type = "BUY" if signal == 1 else "SELL"
        
        prompt = f"""
        A trading signal was generated:
        - Symbol: {symbol}
        - Signal: {signal_type}
        - Price: ${price:.2f}
        - Confidence: {confidence:.2%}
        
        In 1-2 sentences, explain what this signal might indicate 
        (technical conditions, momentum, etc). Keep it brief and factual.
        """
        
        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            explanation = _text(response)
            logger.info(f"✅ Signal explanation for {symbol}")
            return explanation
        
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            return f"Confidence: {confidence:.2%}"
    
    async def news_impact_analysis(self, symbol, news_headline):
        """Claude analyzes impact of news on a stock"""
        if not self.enabled:
            return "Claude API not configured."
        
        prompt = f"""
        News: {news_headline}
        Stock: {symbol}
        
        In 1-2 sentences, assess whether this news is likely to be:
        - Bullish (positive for stock)
        - Bearish (negative for stock)  
        - Neutral (no material impact)
        
        Also mention if trading recommendation should change.
        """
        
        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis = _text(response)
            logger.info(f"✅ News impact analysis for {symbol}")
            return analysis
        
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            return "Unable to analyze news impact"
    
    async def weekly_portfolio_review(self, trades, positions, performance):
        """Claude provides weekly portfolio review"""
        if not self.enabled:
            return "Claude API not configured."
        
        prompt = f"""
        Weekly Trading Review:
        
        Trades executed: {json.dumps(trades, indent=2)}
        Current positions: {json.dumps(positions, indent=2)}
        Performance: {performance}
        
        Provide a brief weekly review (3-4 sentences) covering:
        1. What went well this week
        2. What needs improvement
        3. Recommendation for next week
        
        Be constructive and actionable.
        """
        
        try:
            response = await self.client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            review = _text(response)
            logger.info("✅ Weekly review generated")
            return review
        
        except Exception as e:
            logger.error(f"❌ Claude API error: {e}")
            return "Error generating weekly review"
    
    def is_enabled(self):
        """Check if Claude API is enabled"""
        return self.enabled


# Synchronous wrapper for Discord async context
class ClaudeAnalyzerSync(ClaudeAnalyzer):
    """Synchronous version of ClaudeAnalyzer for non-async contexts"""
    
    def daily_market_analysis_sync(self):
        """Synchronous daily market analysis"""
        if not self.enabled:
            return "Claude API not configured."
        
        prompt = f"""
        Today is {datetime.now().strftime('%A, %B %d, %Y')}.
        
        Provide a brief market analysis (2-3 sentences) covering:
        1. Current market sentiment
        2. Any major economic events
        3. Trading recommendation (active/cautious/avoid)
        """
        
        try:
            response = self.sync_client.messages.create(
                model=MODEL,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}]
            )
            return _text(response)
        except Exception as e:
            return f"Error: {str(e)}"
