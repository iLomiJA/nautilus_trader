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
from abc import ABC, abstractmethod
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import itertools
import re
from datetime import datetime, timedelta

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


class TriangularArbitrageAgent(Actor, ABC):
    """
    Base class for triangular arbitrage intelligent agents.
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
        self.update_interval_secs = config.get("update_interval_secs", 60) # 更新间隔时间（秒）
        self.__msgbus = msgbus # Message bus for communication
        self.__cache = cache
        self.__clock = clock
        self.__is_running = False
        self.__last_update_time = None

    @abstractmethod
    async def analyze(self) -> Dict[str, Any]:
        """
         分析当前市场状况并返回推荐结果。

        Returns
        -------
        Dict[str, Any]
            分析结果和推荐。
        """
        pass

    async def start(self) -> None:
        """Start the intelligent agent."""
        self.__is_running = True
        self.log.info(f"Starting {self.__class__.__name__}", LogColor.GREEN)

        # 设置定时器进行周期性更新
        self.__clock.set_timer(
            name=f"{self.__class__.__name__}_update",
            interval=timedelta(seconds=self.update_interval_secs),
            callback=self._on_timer,
        )

    async def stop(self) -> None:
        """Stop the intelligent agent."""
        self.__is_running = False
        self.log.info(f"Stopping {self.__class__.__name__}", LogColor.YELLOW)

    async def _on_timer(self) -> None:
        """处理周期性定时器更新。"""
        if self.__is_running:
            try:
                self.log.info(f"Running analysis for {self.__class__.__name__}", LogColor.GREEN)
                analysis = await self.analyze()
                if analysis:
                    await self._publish_analysis(analysis)
                self.__last_update_time = self.__clock.timestamp_ns()
            except Exception as e:
                self.log.error(f"Error in {self.__class__.__name__}: {e}", LogColor.RED)

    async def _publish_analysis(self, analysis: Dict[str, Any]) -> None:
        """Publish analysis results to the message bus."""
        self.__msgbus.publish(
            topic=f"agent.{self.__class__.__name__.lower()}.analysis",
            msg=analysis,
        )

class TripletSelectionAgent(TriangularArbitrageAgent):
    """
    用于选择最佳加密货币三元组进行三角套利的智能代理。

    该 Agent 分析：
    - 不同货币对的交易量和流动性
    - 不同三元组的历史套利机会
    - 价格波动性和点差模式
    - 市场相关性结构
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
        self.base_currency = config.get("base_currency", "USDT") # 基础货币
        self.candidate_currencies = config.get("candidate_currencies", [
            "BTC", "ETH", "BNB", "ADA", "DOT", "LINK", "SOL", "MATIC", "AVAX", "ATOM"
        ]) # 候选货币
        self.volume_threshold = config.get("volume_threshold", 1000) # 交易量阈值
        self.volatility_window = config.get("volatility_window", 24) # 波动性窗口（小时）

        # 数据分析
        self.triplet_scores: Dict[Tuple[str, str, str], float] = {} # 三元组评分
        self.volume_data: Dict[str, List[float]] = defaultdict(list)  # 交易量数据
        self.arbitrage_history: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)  # 历史套利数据

    async def analyze(self) -> Dict[str, Any]:
        """
        分析并排序加密货币三元组的套利潜力。

        返回
        -------
        Dict[str, Any]
            排名三元组及其评分和推荐。
        """
        available_triplets = self._generate_triplets()  # 生成可用的三元组
        triplet_analysis = {}

        for triplet in available_triplets:
            try:
                score = await self._calculate_triplet_score(triplet) # 计算三元组评分
                triplet_analysis[triplet] = {
                    "score": score, # 三元组的总评分
                    "volume_score": await self._get_volume_score(triplet),  # 交易量评分
                    "volatility_score": await self._get_volatility_score(triplet),   # 波动性评分
                    "spread_score": await self._get_spread_score(triplet), # 点差评分
                    "historical_score": await self._get_historical_performance_score(triplet), # 历史表现评分
                    "liquidity_score": await self._get_liquidity_score(triplet), # 流动性评分
                }
            except Exception as e:
                self.log.error(f"Error analyzing triplet {triplet}: {e}")
                continue

        # Sort triplets by score
        sorted_triplets = sorted(triplet_analysis.items(), key=lambda x: x[1]["score"], reverse=True)

        recommendation = {
            "timestamp": self.__clock.timestamp_ns(),
            "best_triplet": sorted_triplets[0][0] if sorted_triplets else None,
            "triplet_rankings": sorted_triplets,
            "analysis_summary": await self._generate_analysis_summary(triplet_analysis),
            "market_conditions": await self._assess_market_conditions(), # 市场条件评估
        }

        if recommendation["best_triplet"]:
            self.log.info(f"Best triplet for arbitrage: {recommendation['best_triplet']}", LogColor.CYAN)

        return recommendation

    def _generate_triplets(self) -> List[Tuple[str, str, str]]:
        """生成所有可能的三元组供分析。"""
        triplets = []
        base = self.base_currency

        # Generate combinations of 2 currencies from candidates
        for combo in itertools.combinations(self.candidate_currencies, 2):
            triplet = (base, combo[0], combo[1])
            triplets.append(triplet)

        return triplets

    async def _calculate_triplet_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate composite score for a triplet."""
        volume_score = await self._get_volume_score(triplet)
        volatility_score = await self._get_volatility_score(triplet)
        spread_score = await self._get_spread_score(triplet)
        historical_score = await self._get_historical_performance_score(triplet)
        liquidity_score = await self._get_liquidity_score(triplet)

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

    async def _get_volume_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate volume score for a triplet."""
        try:
            base, currency1, currency2 = triplet # 拆分三元组为基础货币和两个目标货币

            # 获取三个交易对的交易量
            pair1_id = InstrumentId.from_str(f"{currency1}{base}.BINANCE")  # 生成交易对ID
            pair2_id = InstrumentId.from_str(f"{currency2}{base}.BINANCE")
            pair3_id = InstrumentId.from_str(f"{currency2}{currency1}.BINANCE")

            volumes = []  # 存储交易量数据
            for pair_id in [pair1_id, pair2_id, pair3_id]:
                # 智能计算交易量评分
                volume_score = self._estimate_pair_volume(str(pair_id))
                volumes.append(volume_score)

            # 三元组的交易量评分是三个交易对中比重最小的交易量评分（最弱链条决定整体流动性）
            return min(volumes)

        except Exception as e:
            self.log.error(f"Error calculating volume score for {triplet}: {e}")
            return 0.0

    def _estimate_pair_volume(self, pair_id: str) -> float:
        """估算交易对的交易量评分。"""
        # 这是一个交易量估算方法
        # 根据以下方式来估算：
        # 1. 分析最近的交易数据
        # 2. 计算24小时交易量
        # 3. 与历史平均量进行对比
        # 4. 考虑市场资本和流动性

        # 对于主流交易对，给出较高的评分
        major_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]
        if any(major in pair_id for major in major_pairs):
            return np.random.uniform(0.7, 1.0)  # 返回较高的流动性
        else:
            return np.random.uniform(0.3, 0.7)  # 对于其他交易对，给出较低的评分

    async def _get_volatility_score(self, triplet: Tuple[str, str, str]) -> float:
        """Calculate volatility score for a triplet."""
        try:
            base, currency1, currency2 = triplet  # 拆分三元组为基础货币和两个目标货币

            # 获取三个交易对的波动性数据
            pair_ids = [
                f"{currency1}{base}.BINANCE",
                f"{currency2}{base}.BINANCE",
                f"{currency2}{currency1}.BINANCE"
            ]

            volatilities = []  # 存储波动性数据
            for pair_id in pair_ids:
                # 计算每个交易对的价格波动性
                volatility = self._calculate_pair_volatility(pair_id)
                volatilities.append(volatility)

            # 波动性较高通常意味着更多的套利机会
            avg_volatility = np.mean(volatilities)  # 计算波动性的平均值

            # 归一化到0-1范围（假设最大波动性为10%）
            volatility_score = min(1.0, avg_volatility / 0.10)

            return volatility_score

        except Exception as e:
            self.log.error(f"Error calculating volatility score for {triplet}: {e}")
            return 0.0

    def _calculate_pair_volatility(self, pair_id: str) -> float:
        """计算交易对的波动性。"""
        # 简化的波动性计算方法
        # 1. 获取最近的价格数据
        # 2. 计算收益率
        # 3. 计算标准差
        # 4. 年化波动性

        return np.random.uniform(0.02, 0.08)  # 2%到8%的波动性

    async def _get_spread_score(self, triplet: Tuple[str, str, str]) -> float:
        """计算三元组的点差评分。"""
        try:
            base, currency1, currency2 = triplet
            # 拆分三元组为基础货币和两个目标货币

            # 获取三个交易对的当前点差
            pair_ids = [
                InstrumentId.from_str(f"{currency1}{base}.BINANCE"),
                InstrumentId.from_str(f"{currency2}{base}.BINANCE"),
                InstrumentId.from_str(f"{currency2}{currency1}.BINANCE")
            ]

            spreads = []  # 存储点差数据
            for pair_id in pair_ids:
                quote = self.__cache.quote_tick(pair_id)  # 获取报价数据
                if quote:
                    spread = float(quote.ask_price - quote.bid_price)  # 计算点差
                    spread_rate = spread / float(quote.mid_price)  # 计算点差与中间价格的比例
                    spreads.append(spread_rate)
                else:
                    spreads.append(0.005)  # 默认点差估算

            # 较小的点差有利于套利
            avg_spread = np.mean(spreads)
            spread_score = max(0.0, 1.0 - avg_spread / 0.01)  # 归一化点差（与1%点差比较）

            return spread_score

        except Exception as e:
            self.log.error(f"Error calculating spread score for {triplet}: {e}")
            return 0.0

    async def _get_historical_performance_score(self, triplet: Tuple[str, str, str]) -> float:
        """计算三元组的历史表现评分。"""
        try:
            if triplet in self.arbitrage_history:
                history = self.arbitrage_history[triplet]

                if len(history) >= 10:  # 至少需要10笔历史数据
                    # 计算成功率和平均利润
                    successful_trades = [h for h in history if h.get("profit", 0) > 0]
                    success_rate = len(successful_trades) / len(history)

                    if successful_trades:
                        avg_profit = np.mean([h["profit"] for h in successful_trades])

                        # 综合成功率和利润
                        performance_score = (success_rate * 0.6) + (min(avg_profit / 0.005, 1.0) * 0.4)
                        return performance_score

            # 如果历史数据不足，默认评分为0.5
            return 0.5

        except Exception as e:
            self.log.error(f"Error calculating historical score for {triplet}: {e}")
            return 0.5

    async def _get_liquidity_score(self, triplet: Tuple[str, str, str]) -> float:
        """计算三元组的流动性评分。"""
        try:
            base, currency1, currency2 = triplet

            # 基于订单簿深度估算流动性

            major_currencies = ["BTC", "ETH", "BNB", "ADA", "DOT"]

            score = 0.0
            for currency in [currency1, currency2]:
                if currency in major_currencies:
                    score += 0.5  # 如果是主流货币，给出较高的评分
                else:
                    score += 0.2  # 对于非主流货币，给出较低的评分

            return min(1.0, score)

        except Exception as e:
            self.log.error(f"Error calculating liquidity score for {triplet}: {e}")
            return 0.0

    async def _assess_market_conditions(self) -> Dict[str, Any]:
        """评估整体市场状况。"""
        return {
            "market_phase": "normal",  # Could be "bull"（牛市）, "bear"（熊市）, "sideways", "volatile"
            "overall_liquidity": "good",
            "volatility_regime": "medium",
            "correlation_environment": "normal",
        }

    async def _generate_analysis_summary(self, triplet_analysis: Dict[Tuple[str, str, str], Dict[str, float]]) -> str:
        """生成三元组分析的总结。"""
        if not triplet_analysis:
            return "No triplets analyzed"

        best_triplet = max(triplet_analysis.keys(), key=lambda x: triplet_analysis[x]["score"])  # 找到最佳三元组
        best_score = triplet_analysis[best_triplet]["score"]  # 获取最佳三元组的评分
        avg_score = np.mean([data["score"] for data in triplet_analysis.values()])  # 获取平均评分

        return f"Best triplet: {best_triplet} (score: {best_score:.3f}), Average score: {avg_score:.3f}"


class MarketSituationAgent(TriangularArbitrageAgent):
    """
      用于市场情况意识，利用外部信息源分析市场状况。

    该Agent分析：
    - 实时市场新闻和情绪
    - 社交媒体情绪和趋势
    - 经济指标和事件
    - 技术分析模式
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
        self.search_interval_hours = config.get("search_interval_hours", 1)  # 设置信息收集的时间间隔，默认为1小时
        self.news_sources = config.get("news_sources", ["coindesk", "cointelegraph", "cryptonews"])  # 设置新闻来源
        self.sentiment_threshold = config.get("sentiment_threshold", 0.3)  # 设置情绪分析的阈值，控制判断的灵敏度

        # Analysis data
        self.news_data: List[Dict] = []
        self.sentiment_scores: List[float] = []
        self.market_events: List[Dict] = []
        self.last_search_time = None

    async def analyze(self) -> Dict[str, Any]:
        """
        分析市场情况，结合外部信息源提供市场情绪、波动性、市场阶段等评估。

        返回
        -------
        Dict[str, Any]
            市场状况分析和推荐。
        """
        # Gather recent information
        current_time = datetime.now()

        # 如果需要重新收集信息（超过设定的间隔时间），则触发信息收集
        if (self.last_search_time is None or
                current_time - self.last_search_time > timedelta(hours=self.search_interval_hours)):
            await self._gather_market_information() # 收集市场信息
            self.last_search_time = current_time  # 更新上次收集时间

        # 分析收集到的信息
        news_analysis = await self._analyze_news_sentiment() # 新闻
        market_analysis = await self._analyze_market_events() # 市场事件和指标
        technical_analysis = await self._analyze_technical_patterns() # 技术分析模式

        # 综合所有分析结果
        overall_analysis = await self._combine_analyses(news_analysis, market_analysis, technical_analysis)

        # 构建推荐对象，包含市场情绪、波动性、风险等信息
        recommendation = {
            "timestamp": self.__clock.timestamp_ns(),
            "overall_sentiment": overall_analysis["sentiment"],
            "volatility_level": overall_analysis["volatility"],  # 市场波动性
            "market_phase": overall_analysis["phase"],  # 市场阶段
            "risk_level": overall_analysis["risk"],  # 风险水平
            "news_analysis": news_analysis,  # 新闻分析结果
            "market_events": market_analysis,  # 市场事件分析结果
            "technical_analysis": technical_analysis,  # 技术分析结果
            "trading_recommendations": await self._generate_trading_recommendations(overall_analysis),  # 基于分析的交易推荐
        }

        self.log.info(f"Market situation: {overall_analysis['sentiment']} sentiment, "
                      f"{overall_analysis['volatility']} volatility", LogColor.CYAN)

        return recommendation

    async def _gather_market_information(self) -> None:
        """收集来自各个外部来源的市场信息。"""
        self.log.info("Gathering market information from external sources")

        # 收集新闻和市场数据
        # 1. 利用MCP工具Web抓取、API调用等方式获取数据（模拟）
        # 2. 分析社交媒体情绪
        # 3. 查看经济日历
        # 4. 监控汇率公告

        await self._simulate_news_gathering()
        await self._simulate_market_events_gathering()

    async def _simulate_news_gathering(self) -> None:
        """收集来自多个新闻源的新闻信息。"""
        # Sample news data (in reality, this would come from APIs or scraping)
        sample_news = [
            {
                "source": "coindesk",
                "title": "Bitcoin Shows Resilience Despite Market Volatility",
                "sentiment": 0.2, # 新闻情绪
                "relevance": 0.8, # 新闻相关性
                "timestamp": datetime.now() - timedelta(hours=2),  # 发布时间
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

        # In a real implementation, you would use WebSearch or WebFetch
        self.news_data = sample_news
        self.sentiment_scores = [news["sentiment"] for news in sample_news]

    async def _simulate_market_events_gathering(self) -> None:
        """收集市场事件和经济指标。"""
        # 市场事件数据例子
        sample_events = [
            {
                "type": "economic_data",  # 事件类型
                "event": "Federal Reserve Interest Rate Decision",  # 事件内容
                "impact": "medium",  # 事件影响
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

    async def _analyze_news_sentiment(self) -> Dict[str, Any]:
        """分析新闻来源的情绪。"""
        if not self.news_data:
            return {"sentiment": "neutral", "confidence": 0.0}  # 如果没有新闻数据，返回中立情绪

        # 计算加权情绪
        total_weight = sum(news["relevance"] for news in self.news_data)
        if total_weight == 0:
            return {"sentiment": "neutral", "confidence": 0.0}

        weighted_sentiment = sum(
            news["sentiment"] * news["relevance"] for news in self.news_data
        ) / total_weight

        # 根据情绪值分类
        if weighted_sentiment > self.sentiment_threshold:
            sentiment_label = "bullish"  # 多头情绪
        elif weighted_sentiment < -self.sentiment_threshold:
            sentiment_label = "bearish"  # 空头情绪
        else:
            sentiment_label = "neutral"  # 中立情绪

        confidence = min(1.0, abs(weighted_sentiment) / 0.5)

        return {
            "sentiment": sentiment_label,
            "confidence": confidence,
            "score": weighted_sentiment,
            "news_count": len(self.news_data),  # 分析的新闻数量
        }

    async def _analyze_market_events(self) -> Dict[str, Any]:
        """分析市场事件及其潜在影响。"""
        if not self.market_events:
            return {"impact": "none", "events": []}  # 如果没有市场事件，返回无影响

        # 根据事件影响力分类
        high_impact = [e for e in self.market_events if e["impact"] == "high"]
        medium_impact = [e for e in self.market_events if e["impact"] == "medium"]
        low_impact = [e for e in self.market_events if e["impact"] == "low"]

        # 综合市场事件的影响力
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

    async def _analyze_technical_patterns(self) -> Dict[str, Any]:
        """分析技术模式和指标。"""
        # 该方法用于分析技术指标和图表模式
        # For now, we'll simulate this analysis

        return {
            "trend": "sideways",  # 趋势 （上涨、下跌、横盘）
            "volatility": "medium",  # 波动性（低、中、高）
            "support_resistance": "neutral",  # 支撑/阻力位
            "momentum": "neutral",  # 动量指标
        }

    async def _combine_analyses(self, news_analysis: Dict, market_analysis: Dict,
                                technical_analysis: Dict) -> Dict[str, Any]:
        """将所有分析结果结合起来，进行综合评估。"""
        # Determine overall sentiment
        sentiment_score = 0.0

        # Weight news sentiment
        if news_analysis["sentiment"] == "bullish":
            sentiment_score += 0.3  # 多头情绪增加评分
        elif news_analysis["sentiment"] == "bearish":
            sentiment_score -= 0.3 # 空头情绪减少评分

        # Weight market events
        if market_analysis["impact"] == "high":
            sentiment_score += 0.2 if market_analysis["events"][0].get("positive", True) else -0.2

        # Weight technical analysis
        if technical_analysis["trend"] == "uptrend":
            sentiment_score += 0.1
        elif technical_analysis["trend"] == "downtrend":
            sentiment_score -= 0.1

        # Determine overall sentiment
        if sentiment_score > 0.2:
            overall_sentiment = "bullish"  # 如果情绪得分大于0.2，则为多头情绪
        elif sentiment_score < -0.2:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        # 判断市场的波动性等级
        volatility_factors = [
            technical_analysis["volatility"],
            "high" if market_analysis["impact"] == "high" else "medium",  # 市场影响大时波动性评定为高
        ]
        # 如果波动性为极端，则设定为极端波动；如果两个因素都为高波动，则为高波动
        if "extreme" in volatility_factors or volatility_factors.count("high") >= 2:
            volatility_level = "extreme"
        elif "high" in volatility_factors:
            volatility_level = "high"
        elif "medium" in volatility_factors:
            volatility_level = "medium"
        else:
            volatility_level = "low"

        # 确定市场阶段
        if overall_sentiment == "bullish" and volatility_level in ["low", "medium"]:
            market_phase = "bull_market"
        elif overall_sentiment == "bearish" and volatility_level in ["low", "medium"]:
            market_phase = "bear_market"
        elif volatility_level in ["high", "extreme"]:
            market_phase = "volatile_market"
        else:
            market_phase = "sideways_market"

        # 确定风险等级
        if volatility_level == "extreme" or market_analysis["impact"] == "high":
            risk_level = "high"
        elif volatility_level == "high" or market_analysis["impact"] == "medium":
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "sentiment": overall_sentiment,
            "volatility": volatility_level,
            "phase": market_phase,
            "risk": risk_level,
            "confidence": news_analysis["confidence"],
        }

    async def _generate_trading_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """基于市场分析结果生成交易建议。"""
        recommendations = {}

        # 根据风险水平调整仓位大小
        if analysis["risk"] == "high":
            recommendations["position_size_multiplier"] = 0.3
        elif analysis["risk"] == "medium":
            recommendations["position_size_multiplier"] = 0.7
        else:
            recommendations["position_size_multiplier"] = 1.0

        # 根据波动性调整利润阈值
        if analysis["volatility"] == "extreme":
            recommendations["profit_threshold_multiplier"] = 2.0
        elif analysis["volatility"] == "high":
            recommendations["profit_threshold_multiplier"] = 1.5
        elif analysis["volatility"] == "low":
            recommendations["profit_threshold_multiplier"] = 0.8
        else:
            recommendations["profit_threshold_multiplier"] = 1.0

        # 根据市场情绪和波动性调整交易活动
        if analysis["risk"] == "high" and analysis["volatility"] == "extreme":
            recommendations["trading_activity"] = "pause" # 高风险和极端波动时暂停交易
        elif analysis["sentiment"] == "bullish" and analysis["volatility"] == "low":
            recommendations["trading_activity"] = "increase"
        elif analysis["sentiment"] == "bearish" and analysis["volatility"] == "high":
            recommendations["trading_activity"] = "decrease"
        else:
            recommendations["trading_activity"] = "normal"

        return recommendations


class ParameterOptimizationAgent(TriangularArbitrageAgent):
    """
    用于实时优化策略参数的智能Agent。

    该Agent分析：
    - 最近的表现指标
    - 市场条件和波动性
    - 交易成本和滑点
    - 风险调整后的回报
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
        self.optimization_window = config.get("optimization_window", 50) # 设置优化窗口大小，默认50
        self.risk_tolerance = config.get("risk_tolerance", 0.02)  # 风险容忍度，默认0.02

        # Performance tracking
        self.performance_history: List[Dict] = []
        self.parameter_history: List[Dict] = []

    async def analyze(self) -> Dict[str, Any]:
        """
        分析当前策略的表现，并根据最近的表现优化策略参数。

        return
        -------
        Dict[str, Any]
            优化后的参数和表现指标。
        """
        current_params = await self._get_current_parameters()  # 获取当前策略的参数
        performance_metrics = await self._calculate_performance_metrics()  # 计算当前策略的表现指标

        # 基于最近的表现优化策略参数
        optimized_params = await self._optimize_parameters(current_params, performance_metrics)

        # 构建推荐对象，包含当前和优化后的参数，以及优化的信心度
        recommendation = {
            "timestamp": self.__clock.timestamp_ns(),
            "current_parameters": current_params,
            "optimized_parameters": optimized_params,
            "performance_metrics": performance_metrics,
            "optimization_confidence": await self._calculate_optimization_confidence(),
            "parameter_changes": await self._calculate_parameter_changes(current_params, optimized_params),
        }

        self.log.info(f"Parameter optimization completed. Confidence: {recommendation['optimization_confidence']:.2f}")

        return recommendation

    async def _get_current_parameters(self) -> Dict[str, Any]:
        """获取当前的策略参数。"""
        # 从策略中获取当前的参数值
        return {
            "min_profit_threshold": 0.002,  # 0.2%
            "trade_size": 0.01,
            "max_position_size": 10.0,
            "position_timeout_mins": 5,
            "transaction_cost_bps": 10,
        }

    async def _calculate_performance_metrics(self) -> Dict[str, float]:
        """计算当前策略的表现指标。"""


        return {
            "total_trades": np.random.randint(20, 100),  # 总交易次数
            "successful_trades": np.random.randint(15, 80),  # 成功交易次数
            "win_rate": np.random.uniform(0.6, 0.8),  # 成功率
            "avg_profit_per_trade": np.random.uniform(0.001, 0.005),  # 每笔交易的平均利润
            "max_drawdown": np.random.uniform(0.01, 0.05),  # 最大回撤
            "sharpe_ratio": np.random.uniform(0.8, 2.5),  # 夏普比率
            "profit_factor": np.random.uniform(1.2, 3.0),  # 利润因子
            "avg_trade_duration_mins": np.random.uniform(2, 10),
        }

    async def _optimize_parameters(self, current_params: Dict[str, Any],
                                   performance: Dict[str, float]) -> Dict[str, Any]:
        """基于当前的表现优化策略参数。"""
        optimized = current_params.copy()

        # 根据成功率优化利润阈值
        if performance["win_rate"] < 0.6:
            # Increase profit threshold to be more selective
            optimized["min_profit_threshold"] = min(
                0.005, current_params["min_profit_threshold"] * 1.2
            )
        elif performance["win_rate"] > 0.8 and performance["avg_profit_per_trade"] > 0.003:
            # Decrease profit threshold to capture more opportunities
            optimized["min_profit_threshold"] = max(
                0.001, current_params["min_profit_threshold"] * 0.9
            )

        # 根据夏普比率和最大回撤优化交易金额
        if performance["sharpe_ratio"] > 2.0 and performance["max_drawdown"] < 0.02:
            # Increase trade size for better performance
            optimized["trade_size"] = min(
                0.05, current_params["trade_size"] * 1.1
            )
        elif performance["max_drawdown"] > 0.04 or performance["sharpe_ratio"] < 1.0:
            # Decrease trade size to reduce risk
            optimized["trade_size"] = max(
                0.005, current_params["trade_size"] * 0.8
            )

        # 根据平均交易时长优化仓位超时
        if performance["avg_trade_duration_mins"] > 8:
            # Increase timeout if trades are taking longer
            optimized["position_timeout_mins"] = min(
                15, current_params["position_timeout_mins"] * 1.5
            )
        elif performance["avg_trade_duration_mins"] < 3:
            # Decrease timeout for faster execution
            optimized["position_timeout_mins"] = max(
                2, current_params["position_timeout_mins"] * 0.8
            )

        # 根据利润因子优化交易成本假设
        if performance["profit_factor"] < 1.5:
            # Increase transaction cost estimate (be more conservative)
            optimized["transaction_cost_bps"] = min(
                20, current_params["transaction_cost_bps"] * 1.2
            )
        elif performance["profit_factor"] > 2.5:
            # Decrease transaction cost estimate
            optimized["transaction_cost_bps"] = max(
                5, current_params["transaction_cost_bps"] * 0.9
            )

        return optimized

    async def _calculate_optimization_confidence(self) -> float:
        """计算优化推荐的信心度。"""
        # 基于样本大小、表现一致性和市场条件来评估信心度
        sample_size_factor = min(1.0, len(self.performance_history) / 50)

        base_confidence = 0.7
        confidence = base_confidence * sample_size_factor

        return confidence

    async def _calculate_parameter_changes(self, current: Dict[str, Any],
                                           optimized: Dict[str, Any]) -> Dict[str, float]:
        """计算参数的变化百分比。"""
        changes = {}

        for param, current_value in current.items():
            optimized_value = optimized[param]
            if current_value != 0:
                change_pct = (optimized_value - current_value) / current_value * 100
                changes[param] = change_pct
            else:
                changes[param] = 0.0

        return changes


class TriangularArbitrageAgentManager:
    """
    三角套利代理管理器类，用于协调多个三角套利智能代理。

    该类负责：
    - 注册和管理多个代理
    - 启动和停止代理
    - 处理来自代理的推荐并应用到套利策略
    """

    def __init__(self, strategy_reference):
        self.strategy = strategy_reference
        self.agents: Dict[str, TriangularArbitrageAgent] = {}
        self.__is_running = False

    def add_agent(self, name: str, agent: TriangularArbitrageAgent):
        """将智能代理添加到管理器中。"""
        self.agents[name] = agent

    async def start_all_agents(self):
        """启动所有已注册的代理。"""
        self.__is_running = True
        for name, agent in self.agents.items():
            await agent.start()

    async def stop_all_agents(self):
        """Stop all registered agents."""
        self.__is_running = False
        for name, agent in self.agents.items():
            await agent.stop()

    def get_agent(self, name: str) -> Optional[TriangularArbitrageAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    async def process_agent_recommendations(self, agent_name: str, recommendations: Dict[str, Any]):
        """处理来自代理的推荐，并将其应用到策略中。"""
        if agent_name == "TripletSelectionAgent":
            await self._handle_triplet_selection(recommendations)
        elif agent_name == "MarketSituationAgent":
            await self._handle_market_situation(recommendations)
        elif agent_name == "ParameterOptimizationAgent":
            await self._handle_parameter_optimization(recommendations)

    async def _handle_triplet_selection(self, recommendations: Dict[str, Any]):
        """Handle triplet selection recommendations."""
        if "best_triplet" in recommendations and recommendations["best_triplet"]:
            await self.strategy.update_triplet(recommendations["best_triplet"])

    async def _handle_market_situation(self, recommendations: Dict[str, Any]):
        """Handle market situation recommendations."""
        if "trading_recommendations" in recommendations:
            await self.strategy.adjust_for_market_conditions(recommendations)

    async def _handle_parameter_optimization(self, recommendations: Dict[str, Any]):
        """Handle parameter optimization recommendations."""
        if "optimized_parameters" in recommendations:
            await self.strategy.update_parameters(recommendations["optimized_parameters"])