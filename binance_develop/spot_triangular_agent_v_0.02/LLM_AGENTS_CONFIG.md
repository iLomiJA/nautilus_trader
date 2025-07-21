# LLM-Enhanced Intelligent Agents Configuration Guide

This document provides configuration examples for integrating LLM models (Claude, GPT, local models) with NautilusTrader intelligent agents for cryptocurrency arbitrage strategies.

## Environment Setup

### Required Environment Variables

```bash
# Anthropic Claude API
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# OpenAI GPT API
export OPENAI_API_KEY="your_openai_api_key_here"

# Binance API (for trading)
export BINANCE_API_KEY="your_binance_api_key_here"
export BINANCE_API_SECRET="your_binance_api_secret_here"
```

### Required Python Packages

```bash
# For Claude integration
pip install anthropic

# For OpenAI integration  
pip install openai

# For local LLM integration
pip install aiohttp

# For Ollama local models
# Install Ollama: https://ollama.ai/
# Then run: ollama run llama2
```

## LLM Configuration Options

### 1. Claude Configuration (Recommended)

```python
LLM_CONFIG_CLAUDE = {
    "enabled": True,
    "provider": "claude",
    "model": "claude-3-sonnet-20240229",  # or "claude-3-haiku-20240307" for faster responses
    "api_key": None,  # Uses ANTHROPIC_API_KEY env var
    "timeout": 30,
    "max_tokens": 4000,
    "temperature": 0.1,  # Low temperature for analytical tasks
    "requests_per_minute": 60,
}
```

### 2. OpenAI GPT Configuration

```python
LLM_CONFIG_OPENAI = {
    "enabled": True,
    "provider": "openai",
    "model": "gpt-4-turbo-preview",  # or "gpt-3.5-turbo" for cost efficiency
    "api_key": None,  # Uses OPENAI_API_KEY env var
    "timeout": 30,
    "max_tokens": 4000,
    "temperature": 0.1,
    "requests_per_minute": 60,
}
```

### 3. Local LLM Configuration (Ollama)

```python
LLM_CONFIG_LOCAL = {
    "enabled": True,
    "provider": "local",
    "model": "llama2",  # or "mistral", "codellama", etc.
    "base_url": "http://localhost:11434",  # Ollama default
    "timeout": 60,  # Longer timeout for local models
    "max_tokens": 4000,
    "temperature": 0.1,
    "requests_per_minute": 120,  # No API limits for local
}
```

### 4. Disabled Configuration (Fallback Only)

```python
LLM_CONFIG_DISABLED = {
    "enabled": False,
}
```

## Agent Configuration Examples

### Triplet Selection Agent Configuration

```python
TRIPLET_SELECTION_AGENT_CONFIG = {
    "update_interval_secs": 300,  # 5 minutes
    "base_currency": "USDT",
    "candidate_currencies": [
        "BTC", "ETH", "BNB", "ADA", "DOT", "LINK", 
        "SOL", "MATIC", "AVAX", "ATOM", "UNI", "AAVE"
    ],
    "volume_threshold": 2000000,  # $2M daily volume
    "volatility_window": 24,  # 24 hours
    "llm_config": LLM_CONFIG_CLAUDE,
    "use_llm_fallback": True,
    "max_history_size": 100,
}
```

### Market Situation Agent Configuration

```python
MARKET_SITUATION_AGENT_CONFIG = {
    "update_interval_secs": 180,  # 3 minutes
    "search_interval_hours": 1,
    "news_sources": [
        "coindesk", "cointelegraph", "cryptonews", 
        "decrypt", "theblock", "coinbase"
    ],
    "sentiment_threshold": 0.3,
    "llm_config": LLM_CONFIG_CLAUDE,
    "use_llm_fallback": True,
    "max_history_size": 50,
}
```

### Parameter Optimization Agent Configuration

```python
PARAMETER_OPTIMIZATION_AGENT_CONFIG = {
    "update_interval_secs": 600,  # 10 minutes
    "optimization_window": 100,  # Number of trades to analyze
    "risk_tolerance": 0.03,  # 3% risk tolerance
    "llm_config": LLM_CONFIG_CLAUDE,
    "use_llm_fallback": True,
    "max_history_size": 200,
}
```

## Usage Examples

### Basic Usage

```python
from llm_enhanced_agents import (
    LLMEnhancedTripletSelectionAgent,
    LLMEnhancedMarketSituationAgent,
    LLMEnhancedParameterOptimizationAgent,
)

# Create agents
triplet_agent = LLMEnhancedTripletSelectionAgent(
    config=TRIPLET_SELECTION_AGENT_CONFIG,
    msgbus=strategy.msgbus,
    cache=strategy.cache,
    clock=strategy.clock,
)

market_agent = LLMEnhancedMarketSituationAgent(
    config=MARKET_SITUATION_AGENT_CONFIG,
    msgbus=strategy.msgbus,
    cache=strategy.cache,
    clock=strategy.clock,
)

param_agent = LLMEnhancedParameterOptimizationAgent(
    config=PARAMETER_OPTIMIZATION_AGENT_CONFIG,
    msgbus=strategy.msgbus,
    cache=strategy.cache,
    clock=strategy.clock,
)

# Start agents
await triplet_agent.start()
await market_agent.start()
await param_agent.start()
```

### Advanced Configuration

```python
# High-frequency trading configuration
HIGH_FREQUENCY_CONFIG = {
    "triplet_selection": {
        "update_interval_secs": 60,  # 1 minute updates
        "llm_config": LLM_CONFIG_CLAUDE,
        "use_llm_fallback": True,
    },
    "market_situation": {
        "update_interval_secs": 30,  # 30 second updates
        "llm_config": LLM_CONFIG_CLAUDE,
        "use_llm_fallback": True,
    },
    "parameter_optimization": {
        "update_interval_secs": 300,  # 5 minute updates
        "llm_config": LLM_CONFIG_CLAUDE,
        "use_llm_fallback": True,
    },
}

# Conservative trading configuration
CONSERVATIVE_CONFIG = {
    "triplet_selection": {
        "update_interval_secs": 3600,  # 1 hour updates
        "llm_config": LLM_CONFIG_CLAUDE,
        "use_llm_fallback": True,
    },
    "market_situation": {
        "update_interval_secs": 600,  # 10 minute updates
        "llm_config": LLM_CONFIG_CLAUDE,
        "use_llm_fallback": True,
    },
    "parameter_optimization": {
        "update_interval_secs": 1800,  # 30 minute updates
        "llm_config": LLM_CONFIG_CLAUDE,
        "use_llm_fallback": True,
    },
}
```

## Performance Optimization

### Model Selection Guidelines

1. **Claude 3 Sonnet**: Best balance of speed and intelligence for financial analysis
2. **Claude 3 Haiku**: Fastest responses, good for high-frequency updates
3. **GPT-4 Turbo**: Excellent analytical capabilities, slightly slower
4. **GPT-3.5 Turbo**: Cost-effective option with good performance
5. **Local Models**: No API costs, full privacy, but requires local compute

### Rate Limiting

```python
# Adjust based on your API plan
RATE_LIMITS = {
    "claude": 60,      # requests per minute
    "openai": 60,      # requests per minute
    "local": 120,      # no API limits
}
```

### Error Handling

```python
# Enable fallback to quantitative analysis if LLM fails
FALLBACK_CONFIG = {
    "use_llm_fallback": True,
    "fallback_confidence": 0.7,  # Lower confidence for fallback
    "retry_attempts": 3,
    "retry_delay_seconds": 5,
}
```

## Monitoring and Logging

### Agent Performance Metrics

```python
# Monitor agent performance
def monitor_agent_performance(agent):
    history = agent.analysis_history
    
    if len(history) >= 10:
        llm_analyses = [h for h in history if not h.get("llm_analysis", {}).get("fallback")]
        fallback_rate = 1 - (len(llm_analyses) / len(history))
        
        print(f"Agent: {agent.__class__.__name__}")
        print(f"Fallback Rate: {fallback_rate:.1%}")
        print(f"Average Confidence: {np.mean([h['overall_confidence'] for h in history]):.2f}")
```

### Log Configuration

```python
# Enhanced logging for LLM agents
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_llm_requests": True,
    "log_llm_responses": True,  # Set to False in production
    "log_performance_metrics": True,
}
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Rate Limiting**: Implement proper rate limiting to avoid API abuse
3. **Error Handling**: Implement graceful degradation when LLM is unavailable
4. **Data Privacy**: Consider using local models for sensitive trading data
5. **Cost Management**: Monitor API usage and costs regularly

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure environment variables are set correctly
2. **Rate Limiting**: Adjust `requests_per_minute` based on your API plan
3. **Timeout Issues**: Increase timeout for slower local models
4. **Package Dependencies**: Install required packages for your chosen LLM provider

### Debug Mode

```python
# Enable debug logging
DEBUG_CONFIG = {
    "log_level": "DEBUG",
    "log_llm_requests": True,
    "log_llm_responses": True,
    "log_fallback_triggers": True,
}
```

## Cost Optimization

### Claude API Pricing (as of 2024)

- **Claude 3 Sonnet**: $3 per 1M input tokens, $15 per 1M output tokens
- **Claude 3 Haiku**: $0.25 per 1M input tokens, $1.25 per 1M output tokens

### OpenAI API Pricing (as of 2024)

- **GPT-4 Turbo**: $10 per 1M input tokens, $30 per 1M output tokens
- **GPT-3.5 Turbo**: $0.50 per 1M input tokens, $1.50 per 1M output tokens

### Cost Estimation

```python
# Estimate daily costs
def estimate_daily_costs(agent_configs):
    # Assume average 500 input tokens, 200 output tokens per request
    daily_requests = sum(
        24 * 60 * 60 / config["update_interval_secs"] 
        for config in agent_configs
    )
    
    # Claude 3 Sonnet pricing
    input_cost = daily_requests * 500 * 3 / 1_000_000
    output_cost = daily_requests * 200 * 15 / 1_000_000
    
    return input_cost + output_cost
```

## Examples

See `llm_enhanced_triangular_arbitrage_live.py` for a complete working example with all three LLM-enhanced agents integrated into a triangular arbitrage strategy.