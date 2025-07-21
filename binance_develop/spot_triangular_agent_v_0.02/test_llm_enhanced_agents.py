#!/usr/bin/env python3
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

"""
Comprehensive test suite for LLM-enhanced intelligent agents.

This test suite covers:
1. LLMClient functionality with different providers
2. Agent quantitative analysis
3. LLM integration and fallback mechanisms
4. Performance and error handling
5. Integration testing with mock trading environment
"""

import asyncio
import json
import os
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pytest

from nautilus_trader.common.component import TestClock
# from nautilus_trader.common.msgbus import MessageBus
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.identifiers import TraderId
# from nautilus_trader.test_kit.mocks import MockCacheImpl
from nautilus_trader.test_kit.mocks.cache_database import MockCacheDatabase

# Import the LLM-enhanced agents
from llm_enhanced_agents import (
    LLMClient,
    LLMEnhancedTripletSelectionAgent,
    LLMEnhancedMarketSituationAgent,
    LLMEnhancedParameterOptimizationAgent,
)


class TestLLMClient(unittest.TestCase):
    """Test cases for LLMClient functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            "provider": "claude",
            "model": "claude-3-sonnet-20240229",
            "api_key": "test_key",
            "timeout": 30,
            "max_tokens": 4000,
            "temperature": 0.1,
            "requests_per_minute": 60,
        }
        
    def test_client_initialization(self):
        """Test LLMClient initialization with different configurations."""
        client = LLMClient(self.mock_config)
        
        self.assertEqual(client.provider, "claude")
        self.assertEqual(client.model, "claude-3-sonnet-20240229")
        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_tokens, 4000)
        self.assertEqual(client.temperature, 0.1)
        
    def test_data_formatting(self):
        """Test data formatting for LLM analysis."""
        client = LLMClient(self.mock_config)
        
        test_data = {
            "simple_value": 42,
            "float_value": 3.14159,
            "dict_value": {"sub_key": "sub_value"},
            "list_value": [1, 2, 3, 4, 5],
            "string_value": "test_string"
        }
        
        formatted = client._format_data_for_llm(test_data)
        
        self.assertIn("SIMPLE_VALUE: 42", formatted)
        self.assertIn("FLOAT_VALUE: 3.14159", formatted)
        self.assertIn("DICT_VALUE:", formatted)
        self.assertIn("LIST_VALUE: 5 items", formatted)
        self.assertIn("STRING_VALUE: test_string", formatted)
        
    def test_response_parsing(self):
        """Test LLM response parsing."""
        client = LLMClient(self.mock_config)
        
        # Test JSON response
        json_response = '''
        Some text before the JSON.
        
        {
            "recommendation": "buy",
            "confidence": 0.85,
            "risk_level": "medium",
            "reasoning": "Market conditions are favorable"
        }
        
        Some text after the JSON.
        '''
        
        parsed = client._parse_llm_response(json_response)
        
        self.assertEqual(parsed["recommendation"], "buy")
        self.assertEqual(parsed["confidence"], 0.85)
        self.assertEqual(parsed["risk_level"], "medium")
        
    def test_fallback_analysis(self):
        """Test fallback analysis when LLM is unavailable."""
        client = LLMClient(self.mock_config)
        
        test_data = {"market_data": "test"}
        fallback = client._fallback_analysis(test_data)
        
        self.assertIn("analysis", fallback)
        self.assertIn("confidence", fallback)
        self.assertIn("fallback", fallback)
        self.assertTrue(fallback["fallback"])
        
    @patch('llm_enhanced_agents.asyncio.sleep')
    async def test_rate_limiting(self, mock_sleep):
        """Test rate limiting functionality."""
        client = LLMClient(self.mock_config)
        
        # Simulate many requests
        for _ in range(5):
            client.request_times.append(datetime.now())
        
        # This should not trigger rate limiting
        await client._check_rate_limit()
        mock_sleep.assert_not_called()
        
        # Simulate exceeding rate limit
        client.requests_per_minute = 3
        await client._check_rate_limit()
        
        # Should trigger rate limiting
        self.assertTrue(len(client.request_times) <= client.requests_per_minute)


class TestLLMEnhancedTripletSelectionAgent(unittest.TestCase):
    """Test cases for LLMEnhancedTripletSelectionAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TEST-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
            config=None,
        )
        self.cache = MockCacheDatabase()
        
        self.config = {
            "update_interval_secs": 60,
            "base_currency": "USDT",
            "candidate_currencies": ["BTC", "ETH", "BNB"],
            "volume_threshold": 1000000,
            "volatility_window": 24,
            "llm_config": {
                "enabled": True,
                "provider": "claude",
                "model": "claude-3-sonnet-20240229",
                "api_key": "test_key",
                "timeout": 30,
                "max_tokens": 4000,
                "temperature": 0.1,
                "requests_per_minute": 60,
            },
            "use_llm_fallback": True,
            "max_history_size": 100,
        }
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = LLMEnhancedTripletSelectionAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        self.assertEqual(agent.base_currency, "USDT")
        self.assertEqual(agent.candidate_currencies, ["BTC", "ETH", "BNB"])
        self.assertIsNotNone(agent.llm_client)
        self.assertTrue(agent.use_llm_fallback)
        
    def test_triplet_generation(self):
        """Test triplet generation from candidate currencies."""
        agent = LLMEnhancedTripletSelectionAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        triplets = agent._generate_triplets()
        
        # With 3 candidate currencies, should generate 3 triplets
        # (USDT, BTC, ETH), (USDT, BTC, BNB), (USDT, ETH, BNB)
        self.assertEqual(len(triplets), 3)
        
        # Check that all triplets start with base currency
        for triplet in triplets:
            self.assertEqual(triplet[0], "USDT")
            
    async def test_quantitative_analysis(self):
        """Test quantitative analysis functionality."""
        agent = LLMEnhancedTripletSelectionAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        analysis = await agent.quantitative_analysis()
        
        self.assertIn("best_triplet", analysis)
        self.assertIn("triplet_rankings", analysis)
        self.assertIn("market_conditions", analysis)
        self.assertIn("confidence", analysis)
        self.assertIn("recommendations", analysis)
        
    def test_system_prompt_generation(self):
        """Test LLM system prompt generation."""
        agent = LLMEnhancedTripletSelectionAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        system_prompt = agent.get_llm_system_prompt()
        
        self.assertIn("triangular arbitrage", system_prompt.lower())
        self.assertIn("cryptocurrency", system_prompt.lower())
        self.assertIn("json", system_prompt.lower())
        
    def test_analysis_prompt_generation(self):
        """Test LLM analysis prompt generation."""
        agent = LLMEnhancedTripletSelectionAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        mock_quant_data = {
            "best_triplet": "('USDT', 'BTC', 'ETH')",
            "total_triplets_analyzed": 3,
            "average_score": 0.75,
        }
        
        analysis_prompt = agent.get_llm_analysis_prompt(mock_quant_data)
        
        self.assertIn("triangular arbitrage", analysis_prompt.lower())
        self.assertIn("USDT", analysis_prompt)
        self.assertIn("BTC", analysis_prompt)
        self.assertIn("ETH", analysis_prompt)


class TestLLMEnhancedMarketSituationAgent(unittest.TestCase):
    """Test cases for LLMEnhancedMarketSituationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TEST-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
            config=None,
        )
        self.cache = MockCacheDatabase()
        
        self.config = {
            "update_interval_secs": 180,
            "search_interval_hours": 1,
            "news_sources": ["coindesk", "cointelegraph", "cryptonews"],
            "sentiment_threshold": 0.3,
            "llm_config": {
                "enabled": True,
                "provider": "claude",
                "model": "claude-3-sonnet-20240229",
                "api_key": "test_key",
                "timeout": 30,
                "max_tokens": 4000,
                "temperature": 0.1,
                "requests_per_minute": 60,
            },
            "use_llm_fallback": True,
            "max_history_size": 50,
        }
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = LLMEnhancedMarketSituationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        self.assertEqual(agent.search_interval_hours, 1)
        self.assertEqual(agent.news_sources, ["coindesk", "cointelegraph", "cryptonews"])
        self.assertEqual(agent.sentiment_threshold, 0.3)
        self.assertIsNotNone(agent.llm_client)
        
    async def test_quantitative_analysis(self):
        """Test quantitative market analysis."""
        agent = LLMEnhancedMarketSituationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        analysis = await agent.quantitative_analysis()
        
        self.assertIn("overall_sentiment", analysis)
        self.assertIn("volatility_level", analysis)
        self.assertIn("market_phase", analysis)
        self.assertIn("risk_level", analysis)
        self.assertIn("recommendations", analysis)
        
    async def test_news_sentiment_analysis(self):
        """Test news sentiment analysis."""
        agent = LLMEnhancedMarketSituationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        # Mock news data
        agent.news_data = [
            {
                "source": "coindesk",
                "title": "Bitcoin Surges to New Heights",
                "sentiment": 0.8,
                "relevance": 0.9,
                "timestamp": datetime.now(),
            },
            {
                "source": "cointelegraph",
                "title": "Market Uncertainty Continues",
                "sentiment": -0.2,
                "relevance": 0.7,
                "timestamp": datetime.now(),
            },
        ]
        
        analysis = await agent._analyze_news_sentiment()
        
        self.assertIn("sentiment", analysis)
        self.assertIn("confidence", analysis)
        self.assertIn("score", analysis)
        self.assertIn("news_count", analysis)
        self.assertEqual(analysis["news_count"], 2)
        
    def test_system_prompt_generation(self):
        """Test LLM system prompt generation."""
        agent = LLMEnhancedMarketSituationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        system_prompt = agent.get_llm_system_prompt()
        
        self.assertIn("market analyst", system_prompt.lower())
        self.assertIn("cryptocurrency", system_prompt.lower())
        self.assertIn("sentiment", system_prompt.lower())
        self.assertIn("json", system_prompt.lower())


class TestLLMEnhancedParameterOptimizationAgent(unittest.TestCase):
    """Test cases for LLMEnhancedParameterOptimizationAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TEST-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
            config=None,
        )
        self.cache = MockCacheDatabase()
        
        self.config = {
            "update_interval_secs": 600,
            "optimization_window": 100,
            "risk_tolerance": 0.03,
            "llm_config": {
                "enabled": True,
                "provider": "claude",
                "model": "claude-3-sonnet-20240229",
                "api_key": "test_key",
                "timeout": 30,
                "max_tokens": 4000,
                "temperature": 0.1,
                "requests_per_minute": 60,
            },
            "use_llm_fallback": True,
            "max_history_size": 200,
        }
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = LLMEnhancedParameterOptimizationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        self.assertEqual(agent.optimization_window, 100)
        self.assertEqual(agent.risk_tolerance, 0.03)
        self.assertIsNotNone(agent.llm_client)
        
    async def test_quantitative_analysis(self):
        """Test quantitative parameter optimization analysis."""
        agent = LLMEnhancedParameterOptimizationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        analysis = await agent.quantitative_analysis()
        
        self.assertIn("current_parameters", analysis)
        self.assertIn("performance_metrics", analysis)
        self.assertIn("optimized_parameters", analysis)
        self.assertIn("parameter_changes", analysis)
        self.assertIn("confidence", analysis)
        
    async def test_parameter_optimization(self):
        """Test parameter optimization logic."""
        agent = LLMEnhancedParameterOptimizationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        current_params = {
            "min_profit_threshold": 0.002,
            "trade_size": 0.01,
            "position_timeout_mins": 5,
        }
        
        # Test with poor performance (should increase thresholds)
        poor_performance = {
            "win_rate": 0.4,
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.06,
            "avg_trade_duration_mins": 10,
        }
        
        optimized = await agent._optimize_parameters(current_params, poor_performance)
        
        # Should increase profit threshold due to low win rate
        self.assertGreater(optimized["min_profit_threshold"], current_params["min_profit_threshold"])
        
        # Should decrease trade size due to high drawdown
        self.assertLess(optimized["trade_size"], current_params["trade_size"])
        
    def test_system_prompt_generation(self):
        """Test LLM system prompt generation."""
        agent = LLMEnhancedParameterOptimizationAgent(
            config=self.config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        system_prompt = agent.get_llm_system_prompt()
        
        self.assertIn("parameter", system_prompt.lower())
        self.assertIn("optimization", system_prompt.lower())
        self.assertIn("trading", system_prompt.lower())
        self.assertIn("json", system_prompt.lower())


class TestIntegration(unittest.TestCase):
    """Integration tests for LLM-enhanced agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TEST-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
            config=None,
        )
        self.cache = MockCacheDatabase()
        
        # Disable LLM for integration tests
        self.base_config = {
            "update_interval_secs": 60,
            "llm_config": {"enabled": False},
            "use_llm_fallback": True,
            "max_history_size": 50,
        }
        
    async def test_agent_coordination(self):
        """Test coordination between multiple agents."""
        triplet_config = {
            **self.base_config,
            "base_currency": "USDT",
            "candidate_currencies": ["BTC", "ETH", "BNB"],
            "volume_threshold": 1000000,
            "volatility_window": 24,
        }
        
        market_config = {
            **self.base_config,
            "search_interval_hours": 1,
            "news_sources": ["coindesk", "cointelegraph"],
            "sentiment_threshold": 0.3,
        }
        
        param_config = {
            **self.base_config,
            "optimization_window": 50,
            "risk_tolerance": 0.02,
        }
        
        # Create agents
        triplet_agent = LLMEnhancedTripletSelectionAgent(
            config=triplet_config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        market_agent = LLMEnhancedMarketSituationAgent(
            config=market_config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        param_agent = LLMEnhancedParameterOptimizationAgent(
            config=param_config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        # Test analysis
        triplet_analysis = await triplet_agent.analyze()
        market_analysis = await market_agent.analyze()
        param_analysis = await param_agent.analyze()
        
        # Verify all agents produced valid analyses
        self.assertIsNotNone(triplet_analysis)
        self.assertIsNotNone(market_analysis)
        self.assertIsNotNone(param_analysis)
        
        # Verify analyses contain expected keys
        self.assertIn("recommendations", triplet_analysis)
        self.assertIn("recommendations", market_analysis)
        self.assertIn("recommendations", param_analysis)
        
    async def test_performance_monitoring(self):
        """Test performance monitoring and history tracking."""
        agent = LLMEnhancedTripletSelectionAgent(
            config={
                **self.base_config,
                "base_currency": "USDT",
                "candidate_currencies": ["BTC", "ETH"],
                "volume_threshold": 1000000,
                "volatility_window": 24,
            },
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        # Generate multiple analyses
        for _ in range(5):
            analysis = await agent.analyze()
            self.assertIsNotNone(analysis)
            
        # Check history tracking
        self.assertEqual(len(agent.analysis_history), 5)
        
        # Verify history contains required fields
        for history_item in agent.analysis_history:
            self.assertIn("timestamp", history_item)
            self.assertIn("analysis", history_item)
            self.assertIn("performance", history_item)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and resilience."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TEST-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
            config=None,
        )
        self.cache = MockCacheDatabase()
        
    async def test_llm_failure_fallback(self):
        """Test fallback to quantitative analysis when LLM fails."""
        config = {
            "update_interval_secs": 60,
            "base_currency": "USDT",
            "candidate_currencies": ["BTC", "ETH"],
            "volume_threshold": 1000000,
            "volatility_window": 24,
            "llm_config": {
                "enabled": True,
                "provider": "claude",
                "model": "claude-3-sonnet-20240229",
                "api_key": "invalid_key",  # This will cause failure
                "timeout": 30,
                "max_tokens": 4000,
                "temperature": 0.1,
                "requests_per_minute": 60,
            },
            "use_llm_fallback": True,
            "max_history_size": 50,
        }
        
        agent = LLMEnhancedTripletSelectionAgent(
            config=config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        # Mock LLM client to raise exception
        agent.llm_client._call_llm_api = AsyncMock(side_effect=Exception("API Error"))
        
        # Analysis should still complete using fallback
        analysis = await agent.analyze()
        
        self.assertIsNotNone(analysis)
        self.assertIn("recommendations", analysis)
        self.assertEqual(analysis["analysis_method"], "quantitative_only")
        
    async def test_invalid_configuration(self):
        """Test handling of invalid configurations."""
        invalid_config = {
            "update_interval_secs": -1,  # Invalid
            "llm_config": {"enabled": False},
            "use_llm_fallback": True,
        }
        
        # Should handle gracefully
        agent = LLMEnhancedTripletSelectionAgent(
            config=invalid_config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        # Agent should still initialize
        self.assertIsNotNone(agent)


# Performance benchmarks
class TestPerformance(unittest.TestCase):
    """Performance tests for LLM-enhanced agents."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clock = TestClock()
        self.trader_id = TraderId("TEST-001")
        self.msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=self.clock,
            config=None,
        )
        self.cache = MockCacheDatabase()
        
    async def test_analysis_performance(self):
        """Test analysis performance without LLM."""
        config = {
            "update_interval_secs": 60,
            "base_currency": "USDT",
            "candidate_currencies": ["BTC", "ETH", "BNB", "ADA", "DOT"],
            "volume_threshold": 1000000,
            "volatility_window": 24,
            "llm_config": {"enabled": False},
            "use_llm_fallback": True,
            "max_history_size": 50,
        }
        
        agent = LLMEnhancedTripletSelectionAgent(
            config=config,
            msgbus=self.msgbus,
            cache=self.cache,
            clock=self.clock,
        )
        
        # Measure analysis time
        import time
        start_time = time.time()
        
        analysis = await agent.analyze()
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(analysis_time, 1.0)  # 1 second
        self.assertIsNotNone(analysis)


if __name__ == "__main__":
    # Run specific test suites
    if len(os.sys.argv) > 1:
        if os.sys.argv[1] == "unit":
            # Run unit tests only
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            suite.addTests(loader.loadTestsFromTestCase(TestLLMClient))
            suite.addTests(loader.loadTestsFromTestCase(TestLLMEnhancedTripletSelectionAgent))
            suite.addTests(loader.loadTestsFromTestCase(TestLLMEnhancedMarketSituationAgent))
            suite.addTests(loader.loadTestsFromTestCase(TestLLMEnhancedParameterOptimizationAgent))
            
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
            
        elif os.sys.argv[1] == "integration":
            # Run integration tests only
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
            suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
            
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
            
        elif os.sys.argv[1] == "performance":
            # Run performance tests only
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
            
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
            
    else:
        # Run all tests
        unittest.main(verbosity=2)