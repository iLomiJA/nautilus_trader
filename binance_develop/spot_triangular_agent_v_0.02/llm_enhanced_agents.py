# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import asyncio
import json
import os
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nautilus_trader.common.actor import Actor, ActorConfig
from nautilus_trader.common.component import Clock
from nautilus_trader.common.enums import LogColor
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument

class LLMClient:
    """
    Client for interacting with Large Language Models (Claude, GPT, etc.).
    
    Supports multiple LLM providers with unified interface for agent intelligence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get("provider", "claude")  # claude, openai, local
        self.model = config.get("model", "claude-3-sonnet-20240229")
        self.api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 30)
        self.max_tokens = config.get("max_tokens", 4000)
        self.temperature = config.get("temperature", 0.1)  # Low temperature for analytical tasks
        
        # Rate limiting
        self.requests_per_minute = config.get("requests_per_minute", 60)
        self.request_times = []
        
    def analyze_data(self, prompt: str, data: Dict[str, Any],
                          system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Send data to LLM for analysis and receive structured recommendations.
        
        Parameters
        ----------
        prompt : str
            The analysis request prompt
        data : Dict[str, Any]
            Market data and metrics to analyze
        system_prompt : Optional[str]
            System prompt for context and behavior
            
        Returns
        -------
        Dict[str, Any]
            LLM analysis results and recommendations
        """
        self._check_rate_limit()
        
        # Format data for LLM
        formatted_data = self._format_data_for_llm(data)
        
        # Construct messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_message = f"{prompt}\n\nData to analyze:\n{formatted_data}"
        messages.append({"role": "user", "content": user_message})
        
        # try:
        response = self._call_llm_api(messages)
        return self._parse_llm_response(response)
        # except Exception as e:
        #     # Fallback to rule-based analysis if LLM fails
        #     return self._fallback_analysis(data)
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = datetime.now()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
        
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def _format_data_for_llm(self, data: Dict[str, Any]) -> str:
        """Format data in a structured way for LLM analysis."""
        formatted = []
        
        for key, value in data.items():
            if isinstance(value, dict):
                formatted.append(f"{key.upper()}:")
                for sub_key, sub_value in value.items():
                    formatted.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                formatted.append(f"{key.upper()}: {len(value)} items")
                if value and len(value) <= 10:  # Show first few items
                    for i, item in enumerate(value[:5]):
                        formatted.append(f"  [{i}]: {item}")
            else:
                formatted.append(f"{key.upper()}: {value}")
        
        return "\n".join(formatted)
    
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API based on provider."""
        if self.provider == "claude":
            return self._call_claude_api(messages)
        elif self.provider == "openai":
            return self._call_openai_api(messages)
        # elif self.provider == "local":
        #     return self._call_local_api(messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _call_claude_api(self, messages: List[Dict[str, str]]) -> str:
        """Call Claude API via Anthropic."""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            
            # Extract system message if present
            system_message = None
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=user_messages
            )
            
            return response.content[0].text
            
        except ImportError:
            raise ImportError("anthropic package required for Claude API. Install with: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")
    
    def _call_openai_api(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            print(f"OpenAI response received: {response}")
            return response.choices[0].message.content
            
        except ImportError:
            raise ImportError("openai package required for OpenAI API. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    # def _call_local_api(self, messages: List[Dict[str, str]]) -> str:
    #     """Call local LLM API (e.g., Ollama, LlamaCpp)."""
    #     try:
    #         import aiohttp
    #
    #         # Format for local API (adjust based on your local setup)
    #         prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    #
    #         payload = {
    #             "model": self.model,
    #             "prompt": prompt,
    #             "stream": False,
    #             "options": {
    #                 "temperature": self.temperature,
    #                 "num_predict": self.max_tokens
    #             }
    #         }
    #
    #         async with aiohttp.ClientSession() as session:
    #             async with session.post(
    #                 f"{self.base_url}/api/generate",
    #                 json=payload,
    #                 timeout=self.timeout
    #             ) as response:
    #                 result = response.json()
    #                 return result.get("response", "")
    #
    #     except Exception as e:
    #         raise RuntimeError(f"Local LLM API error: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            
            # Fallback: parse key-value pairs
            lines = response.split('\n')
            result = {"analysis": response}
            
            for line in lines:
                if ':' in line and not line.strip().startswith('#'):
                    try:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        
                        # Try to convert to appropriate type
                        if value.lower() in ['true', 'false']:
                            result[key] = value.lower() == 'true'
                        elif value.replace('.', '').replace('-', '').isdigit():
                            result[key] = float(value) if '.' in value else int(value)
                        else:
                            result[key] = value
                    except:
                        continue
            
            return result
            
        except Exception:
            return {"analysis": response, "error": "Failed to parse structured response"}
    
    def _fallback_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based analysis if LLM fails."""
        return {
            "analysis": "LLM unavailable - using rule-based fallback",
            "confidence": 0.5,
            "recommendation": "maintain_current_strategy",
            "fallback": True
        }

class LLMEnhancedIntelligentAgent(Actor, ABC):
    """
    Base class for LLM-enhanced intelligent agents.
    
    Combines traditional quantitative analysis with LLM-powered insights
    for superior decision making in trading strategies.
    """
    
    def __init__(
        self,
        actor_config: ActorConfig,
        config: Dict[str, Any],
        msgbus,
        cache,
        clock: Clock,
    ):
        super().__init__(actor_config)
        self.__is_running = False
        self.update_interval_secs = config.get("update_interval_secs", 60)
        self.last_update_time = None
        
        # LLM configuration
        llm_config = config.get("llm_config", {})
        self.llm_client = LLMClient(llm_config) if llm_config.get("enabled", True) else None
        self.use_llm_fallback = config.get("use_llm_fallback", True)
        
        # Analysis history for learning
        self.analysis_history: List[Dict[str, Any]] = []
        self.max_history_size = config.get("max_history_size", 100)

        #else
        self.__msgbus = msgbus
        self.__cache = cache
        self.__clock = clock
        
    @abstractmethod
    def quantitative_analysis(self) -> Dict[str, Any]:
        """
        Perform traditional quantitative analysis.
        
        Returns
        -------
        Dict[str, Any]
            Quantitative analysis results
        """
        pass
    
    @abstractmethod
    def get_llm_system_prompt(self) -> str:
        """
        Get the system prompt for LLM analysis specific to this agent type.
        
        Returns
        -------
        str
            System prompt for the LLM
        """
        pass
    
    @abstractmethod
    def get_llm_analysis_prompt(self, quant_data: Dict[str, Any]) -> str:
        """
        Get the analysis prompt for LLM based on quantitative data.
        
        Parameters
        ----------
        quant_data : Dict[str, Any]
            Quantitative analysis results
            
        Returns
        -------
        str
            Analysis prompt for the LLM
        """
        pass
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis combining quantitative and LLM insights.
        
        Returns
        -------
        Dict[str, Any]
            Combined analysis results and recommendations
        """
        # Step 1: Perform quantitative analysis
        quant_analysis = self.quantitative_analysis()
        
        # Step 2: Enhance with LLM analysis if available
        llm_analysis = {}
        if self.llm_client:
            try:
                system_prompt = self.get_llm_system_prompt()
                analysis_prompt = self.get_llm_analysis_prompt(quant_analysis)
                
                llm_analysis = self.llm_client.analyze_data(
                    prompt=analysis_prompt,
                    data=quant_analysis,
                    system_prompt=system_prompt
                )
                
                self.log.info(f"LLM analysis completed for {self.__class__.__name__}", LogColor.CYAN)
                
            except Exception as e:
                self.log.error(f"LLM analysis failed: {e}", LogColor.YELLOW)
                if self.use_llm_fallback:
                    llm_analysis = {"error": str(e), "fallback": True}
        
        # Step 3: Combine analyses
        combined_analysis = self._combine_analyses(quant_analysis, llm_analysis)
        
        # Step 4: Store analysis for learning
        self._store_analysis_history(combined_analysis)
        
        return combined_analysis
    
    def _combine_analyses(self, quant_analysis: Dict[str, Any],
                               llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine quantitative and LLM analyses with intelligent weighting.
        
        Parameters
        ----------
        quant_analysis : Dict[str, Any]
            Quantitative analysis results
        llm_analysis : Dict[str, Any]
            LLM analysis results
            
        Returns
        -------
        Dict[str, Any]
            Combined analysis with confidence-weighted recommendations
        """
        combined = {
            "timestamp": self.__clock.timestamp_ns(),
            "quantitative_analysis": quant_analysis,
            "llm_analysis": llm_analysis,
            "agent_type": self.__class__.__name__,
        }
        
        # Determine analysis confidence and weighting
        quant_confidence = quant_analysis.get("confidence", 0.7)
        llm_confidence = llm_analysis.get("confidence", 0.8) if llm_analysis and not llm_analysis.get("fallback") else 0.0
        
        # Weight analyses based on confidence and availability
        if llm_confidence > 0:
            total_confidence = quant_confidence + llm_confidence
            quant_weight = quant_confidence / total_confidence
            llm_weight = llm_confidence / total_confidence
            
            combined["analysis_method"] = "hybrid"
            combined["quant_weight"] = quant_weight
            combined["llm_weight"] = llm_weight
            combined["overall_confidence"] = (quant_confidence + llm_confidence) / 2
        else:
            combined["analysis_method"] = "quantitative_only"
            combined["quant_weight"] = 1.0
            combined["llm_weight"] = 0.0
            combined["overall_confidence"] = quant_confidence
        
        # Combine recommendations
        combined["recommendations"] = self._merge_recommendations(
            quant_analysis.get("recommendations", {}),
            llm_analysis.get("recommendations", {}),
            combined["quant_weight"],
            combined["llm_weight"]
        )
        
        return combined
    
    def _merge_recommendations(self, quant_recs: Dict[str, Any],
                                    llm_recs: Dict[str, Any],
                                    quant_weight: float, llm_weight: float) -> Dict[str, Any]:
        """
        Intelligently merge recommendations from quantitative and LLM analyses.
        
        Parameters
        ----------
        quant_recs : Dict[str, Any]
            Quantitative recommendations
        llm_recs : Dict[str, Any]
            LLM recommendations
        quant_weight : float
            Weight for quantitative analysis
        llm_weight : float
            Weight for LLM analysis
            
        Returns
        -------
        Dict[str, Any]
            Merged recommendations
        """
        merged = {}
        
        # Get all recommendation keys
        all_keys = set(quant_recs.keys()) | set(llm_recs.keys())
        
        for key in all_keys:
            quant_val = quant_recs.get(key)
            llm_val = llm_recs.get(key)
            
            if quant_val is not None and llm_val is not None:
                # Both analyses have this recommendation
                if isinstance(quant_val, (int, float)) and isinstance(llm_val, (int, float)):
                    # Numeric values: weighted average
                    merged[key] = quant_val * quant_weight + llm_val * llm_weight
                elif quant_val == llm_val:
                    # Same recommendation: use it
                    merged[key] = quant_val
                else:
                    # Different recommendations: favor higher confidence
                    merged[key] = quant_val if quant_weight > llm_weight else llm_val
                    merged[f"{key}_conflict"] = True
            elif quant_val is not None:
                # Only quantitative analysis has this
                merged[key] = quant_val
            elif llm_val is not None:
                # Only LLM analysis has this
                merged[key] = llm_val
        
        return merged
    
    def _store_analysis_history(self, analysis: Dict[str, Any]):
        """Store analysis results for historical learning."""
        self.analysis_history.append({
            "timestamp": analysis["timestamp"],
            "analysis": analysis,
            "performance": None  # Will be updated later with actual performance
        })
        
        # Limit history size
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history = self.analysis_history[-self.max_history_size:]

class LLMEnhancedTripletSelectionAgent(LLMEnhancedIntelligentAgent):
    """
    LLM-enhanced agent for selecting optimal cryptocurrency triplets for triangular arbitrage.
    
    Combines quantitative analysis (volume, volatility, spreads) with LLM insights
    on market conditions, news sentiment, and cross-asset correlations.
    """
    
    def __init__(
        self,
        actor_config: ActorConfig,
        config: Dict[str, Any],
        msgbus,
        cache,
        clock: Clock,
    ):
        super().__init__(actor_config, config, msgbus, cache, clock)
        self.base_currency = config.get("base_currency", "USDT")
        self.candidate_currencies = config.get("candidate_currencies", [
            "BTC", "ETH", "BNB", "ADA", "DOT", "LINK", "SOL", "MATIC", "AVAX", "ATOM"
        ])
        self.volume_threshold = config.get("volume_threshold", 1000000)
        self.volatility_window = config.get("volatility_window", 24)
        
        # Analysis data
        self.triplet_scores: Dict[Tuple[str, str, str], float] = {}
        self.volume_data: Dict[str, List[float]] = defaultdict(list)
        self.arbitrage_history: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    
    def get_llm_system_prompt(self) -> str:
        """Get system prompt for triplet selection analysis."""
        return """You are an expert cryptocurrency arbitrage analyst specializing in triangular arbitrage opportunities. 

Your role is to analyze market data and provide intelligent recommendations for optimal cryptocurrency triplets for triangular arbitrage trading.

Key areas of expertise:
1. Market microstructure and liquidity analysis
2. Cross-currency correlation patterns
3. Volatility regime analysis
4. News and sentiment impact on specific cryptocurrencies
5. Exchange-specific trading dynamics

Please provide analysis in the following JSON format:
{
    "primary_recommendation": "triplet_name",
    "confidence": 0.85,
    "reasoning": "detailed explanation",
    "market_conditions": "assessment of current conditions",
    "risk_factors": ["factor1", "factor2"],
    "recommendations": {
        "best_triplet": "BTC-ETH-USDT",
        "risk_level": "medium",
        "expected_opportunity_frequency": "high",
        "optimal_trade_size": 0.02
    }
}

Focus on actionable insights that combine quantitative metrics with market intelligence."""
    
    def get_llm_analysis_prompt(self, quant_data: Dict[str, Any]) -> str:
        """Get analysis prompt for triplet selection."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""Analyze the following triangular arbitrage triplet data and provide recommendations for optimal triplet selection.

Current timestamp: {current_time}
Base currency: {self.base_currency}
Candidate currencies: {', '.join(self.candidate_currencies)}

Please analyze the quantitative data and provide:

1. Assessment of market conditions for triangular arbitrage
2. Identification of the most promising triplet(s)
3. Risk assessment for current market environment
4. Recommended position sizing and timing
5. Key factors to monitor

Consider:
- Current volatility levels and their impact on arbitrage opportunities
- Liquidity characteristics across different triplets
- Market correlation patterns that might affect arbitrage efficiency
- Any recent news or events that could impact specific cryptocurrencies
- Optimal trading windows and frequency expectations

Provide specific, actionable recommendations with confidence levels."""
    
    def quantitative_analysis(self) -> Dict[str, Any]:
        """Perform quantitative analysis of triplet opportunities."""
        available_triplets = self._generate_triplets()
        triplet_analysis = {}
        
        for triplet in available_triplets:
            try:
                score = self._calculate_triplet_score(triplet)
                triplet_analysis[str(triplet)] = {
                    "score": score,
                    "volume_score": self._get_volume_score(triplet),
                    "volatility_score": self._get_volatility_score(triplet),
                    "spread_score": self._get_spread_score(triplet),
                    "historical_score":  self._get_historical_performance_score(triplet),
                    "liquidity_score":  self._get_liquidity_score(triplet),
                }
            except Exception as e:
                self.log.error(f"Error analyzing triplet {triplet}: {e}")
                continue
        
        # Sort triplets by score
        sorted_triplets = sorted(triplet_analysis.items(), key=lambda x: x[1]["score"], reverse=True)
        
        return {
            "best_triplet": sorted_triplets[0][0] if sorted_triplets else None,
            "triplet_rankings": sorted_triplets,
            "market_conditions": self._assess_market_conditions(),
            "average_score": np.mean([data["score"] for data in triplet_analysis.values()]) if triplet_analysis else 0,
            "total_triplets_analyzed": len(triplet_analysis),
            "confidence": min(1.0, len(triplet_analysis) / len(available_triplets)) if available_triplets else 0,
            "recommendations": {
                "best_triplet": sorted_triplets[0][0] if sorted_triplets else None,
                "risk_level": "medium",
                "trade_frequency": "normal"
            }
        }
    
    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        """Generate all possible triplets for analysis."""
        triplets = []
        base = self.base_currency
        
        # Generate combinations of 2 currencies from candidates
        import itertools
        for combo in itertools.combinations(self.candidate_currencies, 2):
            triplet = (base, combo[0], combo[1])
            triplets.append(triplet)
            
        return triplets
    
    def _calculate_triplet_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate composite score for a triplet."""
        volume_score = self._get_volume_score(triplet)
        volatility_score = self._get_volatility_score(triplet)
        spread_score = self._get_spread_score(triplet)
        historical_score = self._get_historical_performance_score(triplet)
        liquidity_score = self._get_liquidity_score(triplet)
        
        # Weighted composite score
        weights = {
            "volume": 0.20,
            "volatility": 0.15,
            "spread": 0.25,
            "historical": 0.25,
            "liquidity": 0.15,
        }
        
        score = (
            weights["volume"] * volume_score +
            weights["volatility"] * volatility_score +
            weights["spread"] * spread_score +
            weights["historical"] * historical_score +
            weights["liquidity"] * liquidity_score
        )
        
        return score

    def _estimate_pair_volume(self, pair: str) -> float:
        """Estimate volume score for a trading pair."""
        # Major pairs get higher scores
        major_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
        if any(major in pair for major in major_pairs):
            return np.random.uniform(0.7, 1.0)
        else:
            return np.random.uniform(0.3, 0.7)

    def _get_volume_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate volume score for a triplet."""
        try:
            base, currency1, currency2 = triplet
            volumes = []
            
            # Estimate volume for each pair in the triplet
            for pair in [f"{currency1}{base}", f"{currency2}{base}", f"{currency2}{currency1}"]:
                volume_score = self._estimate_pair_volume(pair)
                volumes.append(volume_score)
            
            # Return minimum volume (weakest link)
            return min(volumes)
            
        except Exception as e:
            self.log.error(f"Error calculating volume score for {triplet}: {e}")
            return 0.0
    
    def _get_volatility_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate volatility score for a triplet."""
        try:
            # Simulate volatility calculation
            volatilities = [np.random.uniform(0.02, 0.08) for _ in range(3)]
            avg_volatility = np.mean(volatilities)
            return min(1.0, avg_volatility / 0.10)
        except Exception:
            return 0.0
    
    def _get_spread_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate spread score for a triplet."""
        try:
            # Simulate spread calculation
            spreads = [np.random.uniform(0.001, 0.01) for _ in range(3)]
            avg_spread = np.mean(spreads)
            return max(0.0, 1.0 - avg_spread / 0.01)
        except Exception:
            return 0.0
    
    def _get_historical_performance_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate historical performance score for a triplet."""
        if triplet in self.arbitrage_history and len(self.arbitrage_history[triplet]) >= 10:
            history = self.arbitrage_history[triplet]
            successful_trades = [h for h in history if h.get("profit", 0) > 0]
            success_rate = len(successful_trades) / len(history)
            
            if successful_trades:
                avg_profit = np.mean([h["profit"] for h in successful_trades])
                return (success_rate * 0.6) + (min(avg_profit / 0.005, 1.0) * 0.4)
                
        return 0.5  # Default for new triplets
    
    def _get_liquidity_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate liquidity score for a triplet."""
        try:
            base, currency1, currency2 = triplet
            major_currencies = ["BTC", "ETH", "BNB", "ADA", "DOT"]
            
            score = 0.0
            for currency in [currency1, currency2]:
                if currency in major_currencies:
                    score += 0.5
                else:
                    score += 0.2
            
            return min(1.0, score)
        except Exception:
            return 0.0
    
    def _assess_market_conditions(self) -> Dict[str, Any]:
        """Assess overall market conditions."""
        return {
            "market_phase": "normal",
            "overall_liquidity": "good",
            "volatility_regime": "medium",
            "correlation_environment": "normal",
        }


class LLMEnhancedMarketSituationAgent(LLMEnhancedIntelligentAgent):
    """
    LLM-enhanced agent for market situation awareness using external information sources.
    
    Combines quantitative market analysis with LLM interpretation of news,
    social sentiment, and macroeconomic factors for superior market intelligence.
    """
    
    def __init__(
        self,
        actor_config: ActorConfig,
        config: Dict[str, Any],
        msgbus,
        cache,
        clock: Clock,
    ):
        super().__init__(actor_config, config, msgbus, cache, clock)
        self.search_interval_hours = config.get("search_interval_hours", 1)
        self.news_sources = config.get("news_sources", ["coindesk", "cointelegraph", "cryptonews"])
        self.sentiment_threshold = config.get("sentiment_threshold", 0.3)
        
        # Analysis data
        self.news_data: List[Dict] = []
        self.sentiment_scores: List[float] = []
        self.market_events: List[Dict] = []
        self.last_search_time = None
    
    def get_llm_system_prompt(self) -> str:
        """Get system prompt for market situation analysis."""
        return """You are an expert cryptocurrency market analyst with deep knowledge of market dynamics, sentiment analysis, and macroeconomic factors affecting digital assets.

Your role is to interpret market data, news, and external factors to provide actionable trading insights for algorithmic arbitrage strategies.

Key areas of expertise:
1. Cryptocurrency market sentiment analysis
2. News impact assessment on trading strategies
3. Macroeconomic event interpretation
4. Risk regime identification
5. Market timing and positioning advice

Please provide analysis in the following JSON format:
{
    "market_sentiment": "bullish|bearish|neutral",
    "confidence": 0.85,
    "volatility_forecast": "low|medium|high|extreme",
    "risk_level": "low|medium|high",
    "reasoning": "detailed market assessment",
    "key_factors": ["factor1", "factor2"],
    "recommendations": {
        "trading_activity": "increase|decrease|pause|normal",
        "position_size_multiplier": 1.2,
        "profit_threshold_multiplier": 1.0,
        "risk_multiplier": 0.8
    }
}

Focus on how current market conditions specifically impact arbitrage opportunities and risk management."""
    
    def get_llm_analysis_prompt(self, quant_data: Dict[str, Any]) -> str:
        """Get analysis prompt for market situation."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""Analyze the current cryptocurrency market situation and provide recommendations for arbitrage trading strategies.

Current timestamp: {current_time}
Recent news sources monitored: {', '.join(self.news_sources)}

Please analyze the quantitative market data and provide:

1. Overall market sentiment assessment
2. Volatility regime identification and forecast
3. Key risk factors and opportunities
4. Impact on arbitrage trading strategies
5. Recommended adjustments to trading parameters

Consider:
- Recent news and events affecting cryptocurrency markets
- Regulatory developments and their implications
- Macroeconomic factors (interest rates, inflation, global markets)
- Technical market structure and correlation patterns
- Liquidity conditions across major exchanges

Focus particularly on how current conditions affect:
- Arbitrage opportunity frequency and magnitude
- Risk levels and appropriate position sizing
- Optimal profit thresholds for current volatility
- Market timing considerations

Provide specific, actionable recommendations for algorithmic arbitrage systems."""
    
    def quantitative_analysis(self) -> Dict[str, Any]:
        """Perform quantitative market situation analysis."""
        current_time = datetime.now()
        
        # Check if we need to gather new information
        if (self.last_search_time is None or 
            current_time - self.last_search_time > timedelta(hours=self.search_interval_hours)):
            
            self._gather_market_information()
            self.last_search_time = current_time
        
        # Analyze collected information
        news_analysis = self._analyze_news_sentiment()
        market_analysis = self._analyze_market_events()
        technical_analysis = self._analyze_technical_patterns()
        
        # Combine analyses
        overall_analysis = self._combine_market_analyses(news_analysis, market_analysis, technical_analysis)
        
        return {
            "overall_sentiment": overall_analysis["sentiment"],
            "volatility_level": overall_analysis["volatility"],
            "market_phase": overall_analysis["phase"],
            "risk_level": overall_analysis["risk"],
            "news_analysis": news_analysis,
            "market_events": market_analysis,
            "technical_analysis": technical_analysis,
            "confidence": overall_analysis["confidence"],
            "recommendations": self._generate_trading_recommendations(overall_analysis),
        }

    def _gather_market_information(self) -> None:
        """Gather market information from various sources."""
        self.log.info("Gathering market information from external sources")
        
        # Simulate gathering news and market data
        self._simulate_news_gathering()
        self._simulate_market_events_gathering()
    
    def _simulate_news_gathering(self) -> None:
        """Simulate gathering news from various sources."""
        # Sample news data
        sample_news = [
            {
                "source": "coindesk",
                "title": "Bitcoin Shows Resilience Despite Market Volatility",
                "sentiment": 0.2,
                "relevance": 0.8,
                "timestamp": datetime.now() - timedelta(hours=2),
            },
            {
                "source": "cointelegraph",
                "title": "Ethereum 2.0 Upgrades Continue to Drive Adoption",
                "sentiment": 0.6,
                "relevance": 0.7,
                "timestamp": datetime.now() - timedelta(hours=1),
            },
            {
                "source": "cryptonews",
                "title": "Regulatory Uncertainty Continues to Impact Crypto Markets",
                "sentiment": -0.4,
                "relevance": 0.6,
                "timestamp": datetime.now() - timedelta(minutes=30),
            },
        ]
        
        self.news_data = sample_news
        self.sentiment_scores = [news["sentiment"] for news in sample_news]
    
    def _simulate_market_events_gathering(self) -> None:
        """Simulate gathering market events."""
        sample_events = [
            {
                "type": "economic_data",
                "event": "Federal Reserve Interest Rate Decision",
                "impact": "medium",
                "timestamp": datetime.now() - timedelta(hours=6),
            },
            {
                "type": "exchange_announcement",
                "event": "New cryptocurrency listings on major exchange",
                "impact": "low",
                "timestamp": datetime.now() - timedelta(hours=4),
            },
        ]
        
        self.market_events = sample_events
    
    def _analyze_news_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment from news sources."""
        if not self.news_data:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        # Calculate weighted sentiment
        total_weight = sum(news["relevance"] for news in self.news_data)
        if total_weight == 0:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        weighted_sentiment = sum(
            news["sentiment"] * news["relevance"] for news in self.news_data
        ) / total_weight
        
        # Classify sentiment
        if weighted_sentiment > self.sentiment_threshold:
            sentiment_label = "bullish"
        elif weighted_sentiment < -self.sentiment_threshold:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"
        
        confidence = min(1.0, abs(weighted_sentiment) / 0.5)
        
        return {
            "sentiment": sentiment_label,
            "confidence": confidence,
            "score": weighted_sentiment,
            "news_count": len(self.news_data),
        }
    
    def _analyze_market_events(self) -> Dict[str, Any]:
        """Analyze market events and their potential impact."""
        if not self.market_events:
            return {"impact": "none", "events": []}
        
        # Categorize events by impact
        high_impact = [e for e in self.market_events if e["impact"] == "high"]
        medium_impact = [e for e in self.market_events if e["impact"] == "medium"]
        low_impact = [e for e in self.market_events if e["impact"] == "low"]
        
        # Determine overall impact
        if high_impact:
            overall_impact = "high"
        elif medium_impact:
            overall_impact = "medium"
        else:
            overall_impact = "low"
        
        return {
            "impact": overall_impact,
            "events": self.market_events,
            "high_impact_count": len(high_impact),
            "medium_impact_count": len(medium_impact),
            "low_impact_count": len(low_impact),
        }
    
    def _analyze_technical_patterns(self) -> Dict[str, Any]:
        """Analyze technical patterns and indicators."""
        return {
            "trend": np.random.choice(["uptrend", "downtrend", "sideways"]),
            "volatility": np.random.choice(["low", "medium", "high", "extreme"]),
            "support_resistance": "neutral",
            "momentum": np.random.choice(["bullish", "bearish", "neutral"]),
        }
    
    def _combine_market_analyses(self, news_analysis: Dict, market_analysis: Dict,
                                     technical_analysis: Dict) -> Dict[str, Any]:
        """Combine all market analyses."""
        sentiment_score = 0.0
        
        # Weight news sentiment
        if news_analysis["sentiment"] == "bullish":
            sentiment_score += 0.3
        elif news_analysis["sentiment"] == "bearish":
            sentiment_score -= 0.3
        
        # Weight market events
        if market_analysis["impact"] == "high":
            sentiment_score += 0.2 if np.random.random() > 0.5 else -0.2
        
        # Weight technical analysis
        if technical_analysis["trend"] == "uptrend":
            sentiment_score += 0.1
        elif technical_analysis["trend"] == "downtrend":
            sentiment_score -= 0.1
        
        # Determine overall sentiment
        if sentiment_score > 0.2:
            overall_sentiment = "bullish"
        elif sentiment_score < -0.2:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        # Determine volatility and risk
        volatility_level = technical_analysis["volatility"]
        if market_analysis["impact"] == "high":
            volatility_level = "high"
        
        risk_level = "low"
        if volatility_level in ["high", "extreme"] or market_analysis["impact"] == "high":
            risk_level = "high"
        elif volatility_level == "medium" or market_analysis["impact"] == "medium":
            risk_level = "medium"
        
        return {
            "sentiment": overall_sentiment,
            "volatility": volatility_level,
            "phase": f"{overall_sentiment}_market",
            "risk": risk_level,
            "confidence": news_analysis["confidence"],
        }
    
    def _generate_trading_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis."""
        recommendations = {}
        
        # Position sizing recommendations
        if analysis["risk"] == "high":
            recommendations["position_size_multiplier"] = 0.3
        elif analysis["risk"] == "medium":
            recommendations["position_size_multiplier"] = 0.7
        else:
            recommendations["position_size_multiplier"] = 1.0
        
        # Profit threshold recommendations
        if analysis["volatility"] == "extreme":
            recommendations["profit_threshold_multiplier"] = 2.0
        elif analysis["volatility"] == "high":
            recommendations["profit_threshold_multiplier"] = 1.5
        elif analysis["volatility"] == "low":
            recommendations["profit_threshold_multiplier"] = 0.8
        else:
            recommendations["profit_threshold_multiplier"] = 1.0
        
        # Trading activity recommendations
        if analysis["risk"] == "high" and analysis["volatility"] == "extreme":
            recommendations["trading_activity"] = "pause"
        elif analysis["sentiment"] == "bullish" and analysis["volatility"] == "low":
            recommendations["trading_activity"] = "increase"
        elif analysis["sentiment"] == "bearish" and analysis["volatility"] == "high":
            recommendations["trading_activity"] = "decrease"
        else:
            recommendations["trading_activity"] = "normal"
        
        return recommendations


class LLMEnhancedParameterOptimizationAgent(LLMEnhancedIntelligentAgent):
    """
    LLM-enhanced agent for optimizing strategy parameters in real-time.
    
    Combines quantitative performance analysis with LLM insights on market
    regimes, risk management, and adaptive strategy optimization.
    """
    
    def __init__(
        self,
        actor_config: ActorConfig,
        config: Dict[str, Any],
        msgbus,
        cache,
        clock: Clock,
    ):
        super().__init__(actor_config, config, msgbus, cache, clock)
        self.optimization_window = config.get("optimization_window", 50)
        self.risk_tolerance = config.get("risk_tolerance", 0.02)
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.parameter_history: List[Dict] = []
    
    def get_llm_system_prompt(self) -> str:
        """Get system prompt for parameter optimization analysis."""
        return """You are an expert quantitative trading system optimizer with deep expertise in algorithmic trading parameter tuning and risk management.

Your role is to analyze trading performance metrics and recommend optimal parameter adjustments for cryptocurrency arbitrage strategies.

Key areas of expertise:
1. Performance metrics interpretation (Sharpe ratio, drawdown, win rate)
2. Parameter sensitivity analysis and optimization
3. Risk-adjusted return optimization
4. Market regime-aware parameter tuning
5. Adaptive strategy configuration

Please provide analysis in the following JSON format:
{
    "optimization_priority": "high|medium|low",
    "confidence": 0.85,
    "performance_assessment": "excellent|good|fair|poor",
    "key_issues": ["issue1", "issue2"],
    "improvement_areas": ["area1", "area2"],
    "recommendations": {
        "min_profit_threshold": 0.002,
        "trade_size": 0.015,
        "position_timeout_mins": 8,
        "risk_multiplier": 0.9,
        "parameter_changes_explanation": "detailed reasoning for each change"
    }
}

Focus on providing specific, actionable parameter adjustments that balance profitability with risk management in current market conditions."""
    
    def get_llm_analysis_prompt(self, quant_data: Dict[str, Any]) -> str:
        """Get analysis prompt for parameter optimization."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return f"""Analyze the following trading performance metrics and recommend optimal parameter adjustments for the arbitrage strategy.

Current timestamp: {current_time}
Optimization window: {self.optimization_window} trades
Risk tolerance: {self.risk_tolerance:.1%}

Please analyze the performance data and provide:

1. Assessment of current strategy performance
2. Identification of key performance bottlenecks
3. Specific parameter optimization recommendations
4. Risk-adjusted optimization suggestions
5. Adaptive tuning for current market conditions

Consider:
- Performance metrics trend analysis
- Risk-return optimization opportunities
- Market regime sensitivity of current parameters
- Transaction cost impact on profitability
- Drawdown management and position sizing
- Optimal trade frequency and timing

Provide specific parameter values with detailed reasoning for each recommendation. 
Focus on improvements that enhance risk-adjusted returns while maintaining strategy robustness."""
    
    def quantitative_analysis(self) -> Dict[str, Any]:
        """Perform quantitative parameter optimization analysis."""
        current_params = self._get_current_parameters()
        performance_metrics = self._calculate_performance_metrics()
        
        # Analyze parameter performance relationship
        param_sensitivity = self._analyze_parameter_sensitivity()
        
        # Optimize parameters based on performance
        optimized_params = self._optimize_parameters(current_params, performance_metrics)
        
        return {
            "current_parameters": current_params,
            "performance_metrics": performance_metrics,
            "parameter_sensitivity": param_sensitivity,
            "optimized_parameters": optimized_params,
            "parameter_changes": self._calculate_parameter_changes(current_params, optimized_params),
            "confidence": self._calculate_optimization_confidence(),
            "recommendations": {
                "optimization_priority": self._assess_optimization_priority(performance_metrics),
                "performance_trend": self._analyze_performance_trend(),
                "risk_assessment": self._assess_current_risk_level(performance_metrics)
            }
        }
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters."""
        return {
            "min_profit_threshold": 0.002,
            "trade_size": 0.01,
            "max_position_size": 10.0,
            "position_timeout_mins": 5,
            "transaction_cost_bps": 10,
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate recent performance metrics."""
        return {
            "total_trades": np.random.randint(20, 100),
            "successful_trades": np.random.randint(15, 80),
            "win_rate": np.random.uniform(0.6, 0.8),
            "avg_profit_per_trade": np.random.uniform(0.001, 0.005),
            "max_drawdown": np.random.uniform(0.01, 0.05),
            "sharpe_ratio": np.random.uniform(0.8, 2.5),
            "profit_factor": np.random.uniform(1.2, 3.0),
            "avg_trade_duration_mins": np.random.uniform(2, 10),
            "total_return": np.random.uniform(0.05, 0.25),
            "volatility": np.random.uniform(0.02, 0.08),
        }
    
    def _analyze_parameter_sensitivity(self) -> Dict[str, float]:
        """Analyze sensitivity of performance to parameter changes."""
        return {
            "profit_threshold_sensitivity": np.random.uniform(0.3, 0.8),
            "trade_size_sensitivity": np.random.uniform(0.2, 0.6),
            "timeout_sensitivity": np.random.uniform(0.1, 0.4),
            "overall_stability": np.random.uniform(0.6, 0.9),
        }
    
    def _optimize_parameters(self, current_params: Dict[str, Any],
                                  performance: Dict[str, float]) -> Dict[str, Any]:
        """Optimize strategy parameters based on performance."""
        optimized = current_params.copy()
        
        # Optimize profit threshold
        if performance["win_rate"] < 0.6:
            optimized["min_profit_threshold"] = min(
                0.005, current_params["min_profit_threshold"] * 1.2
            )
        elif performance["win_rate"] > 0.8 and performance["avg_profit_per_trade"] > 0.003:
            optimized["min_profit_threshold"] = max(
                0.001, current_params["min_profit_threshold"] * 0.9
            )
        
        # Optimize trade size based on Sharpe ratio and drawdown
        if performance["sharpe_ratio"] > 2.0 and performance["max_drawdown"] < 0.02:
            optimized["trade_size"] = min(
                0.05, current_params["trade_size"] * 1.1
            )
        elif performance["max_drawdown"] > 0.04 or performance["sharpe_ratio"] < 1.0:
            optimized["trade_size"] = max(
                0.005, current_params["trade_size"] * 0.8
            )
        
        # Optimize position timeout
        if performance["avg_trade_duration_mins"] > 8:
            optimized["position_timeout_mins"] = min(
                15, current_params["position_timeout_mins"] * 1.5
            )
        elif performance["avg_trade_duration_mins"] < 3:
            optimized["position_timeout_mins"] = max(
                2, current_params["position_timeout_mins"] * 0.8
            )
        
        return optimized
    
    def _calculate_parameter_changes(self, current: Dict[str, Any],
                                          optimized: Dict[str, Any]) -> Dict[str, float]:
        """Calculate percentage changes in parameters."""
        changes = {}
        
        for param, current_value in current.items():
            optimized_value = optimized[param]
            if current_value != 0:
                change_pct = (optimized_value - current_value) / current_value * 100
                changes[param] = change_pct
            else:
                changes[param] = 0.0
        
        return changes
    
    def _calculate_optimization_confidence(self) -> float:
        """Calculate confidence in optimization recommendations."""
        sample_size_factor = min(1.0, len(self.performance_history) / 50)
        base_confidence = 0.7
        return base_confidence * sample_size_factor
    
    def _assess_optimization_priority(self, performance: Dict[str, float]) -> str:
        """Assess the priority level for optimization."""
        if performance["sharpe_ratio"] < 1.0 or performance["max_drawdown"] > 0.05:
            return "high"
        elif performance["win_rate"] < 0.6 or performance["profit_factor"] < 1.5:
            return "medium"
        else:
            return "low"
    
    def _analyze_performance_trend(self) -> str:
        """Analyze recent performance trend."""
        trends = ["improving", "stable", "declining"]
        return np.random.choice(trends)
    
    def _assess_current_risk_level(self, performance: Dict[str, float]) -> str:
        """Assess current risk level based on performance."""
        if performance["max_drawdown"] > 0.04 or performance["volatility"] > 0.06:
            return "high"
        elif performance["max_drawdown"] > 0.02 or performance["volatility"] > 0.04:
            return "medium"
        else:
            return "low"