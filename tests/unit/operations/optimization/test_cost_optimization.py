"""Tests for AI-Powered Cost Optimization System"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.operations.optimization.cost_optimization import (
    AIOperationsCostOptimizer,
    AIProviderCostOptimizer,
    InfrastructureCostOptimizer,
    CostForecastingEngine,
    CostOptimization,
    OptimizationStrategy,
    CostCategory
)

@pytest.fixture
def ai_provider_optimizer():
    return AIProviderCostOptimizer()

@pytest.fixture
def infrastructure_optimizer():
    return InfrastructureCostOptimizer()

@pytest.fixture
def forecasting_engine():
    return CostForecastingEngine()

@pytest.fixture
def cost_optimizer():
    return AIOperationsCostOptimizer()

class TestAIProviderCostOptimizer:
    """Test AIProviderCostOptimizer class"""
    
    @pytest.mark.asyncio
    async def test_optimize_ai_provider_usage(self, ai_provider_optimizer):
        """Test AI provider cost optimization"""
        with patch.object(ai_provider_optimizer, '_analyze_ai_usage_patterns', AsyncMock()) as mock_analyze:
            with patch.object(ai_provider_optimizer, '_calculate_current_ai_costs', AsyncMock()) as mock_current:
                with patch.object(ai_provider_optimizer, '_identify_ai_optimizations', AsyncMock()) as mock_identify:
                    with patch.object(ai_provider_optimizer, '_calculate_optimized_ai_costs', AsyncMock()) as mock_optimized:
                        with patch.object(ai_provider_optimizer, '_generate_ai_optimization_steps', AsyncMock()) as mock_steps:
                            
                            # Setup mocks
                            mock_analyze.return_value = {'total_tokens_per_day': 500000}
                            mock_current.return_value = 1500.0
                            mock_identify.return_value = [{'type': 'caching', 'potential_savings': 0.3}]
                            mock_optimized.return_value = 1200.0
                            mock_steps.return_value = ['Implement caching']
                            
                            result = await ai_provider_optimizer.optimize_ai_provider_usage()
                            
                            assert isinstance(result, CostOptimization)
                            assert result.strategy == OptimizationStrategy.PROVIDER_OPTIMIZATION
                            assert result.category == CostCategory.AI_SERVICES
                            assert result.current_cost == 1500.0
                            assert result.optimized_cost == 1200.0
                            assert result.savings_amount == 300.0
                            assert result.savings_percentage == 20.0
    
    @pytest.mark.asyncio
    async def test_analyze_ai_usage_patterns(self, ai_provider_optimizer):
        """Test AI usage pattern analysis"""
        usage = await ai_provider_optimizer._analyze_ai_usage_patterns()
        
        assert 'total_tokens_per_day' in usage
        assert 'model_distribution' in usage
        assert 'request_types' in usage
        assert 'cache_potential' in usage
        
        # Check that percentages sum to 1
        model_dist = usage['model_distribution']
        assert abs(sum(model_dist.values()) - 1.0) < 0.01
        
        request_types = usage['request_types']
        assert abs(sum(request_types.values()) - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_calculate_current_ai_costs(self, ai_provider_optimizer):
        """Test current AI cost calculation"""
        usage = {
            'total_tokens_per_day': 100000,
            'model_distribution': {
                'gpt-4': 0.3,
                'gpt-3.5-turbo': 0.7
            }
        }
        
        cost = await ai_provider_optimizer._calculate_current_ai_costs(usage)
        
        assert cost > 0
        assert isinstance(cost, float)
        
        # Verify cost calculation logic
        expected_daily = (100000 * 0.3 / 1000 * 0.03) + (100000 * 0.7 / 1000 * 0.002)
        expected_monthly = expected_daily * 30
        assert abs(cost - expected_monthly) < 1.0
    
    @pytest.mark.asyncio
    async def test_identify_ai_optimizations(self, ai_provider_optimizer):
        """Test AI optimization identification"""
        usage = {
            'cache_potential': 0.4,
            'quality_requirements': {'reasoning': 'high', 'conversation': 'medium'}
        }
        
        optimizations = await ai_provider_optimizer._identify_ai_optimizations(usage)
        
        assert len(optimizations) > 0
        assert all('type' in opt for opt in optimizations)
        assert all('potential_savings' in opt for opt in optimizations)
        
        # Check for expected optimization types
        opt_types = [opt['type'] for opt in optimizations]
        assert 'intelligent_model_selection' in opt_types
        assert 'enhanced_caching' in opt_types

class TestInfrastructureCostOptimizer:
    """Test InfrastructureCostOptimizer class"""
    
    @pytest.mark.asyncio
    async def test_optimize_infrastructure_costs(self, infrastructure_optimizer):
        """Test comprehensive infrastructure optimization"""
        with patch.object(infrastructure_optimizer, '_analyze_infrastructure_utilization', AsyncMock()) as mock_analyze:
            with patch.object(infrastructure_optimizer, '_calculate_current_infrastructure_costs', AsyncMock()) as mock_current:
                with patch.object(infrastructure_optimizer, '_optimize_compute_costs', AsyncMock()) as mock_compute:
                    with patch.object(infrastructure_optimizer, '_optimize_storage_costs', AsyncMock()) as mock_storage:
                        with patch.object(infrastructure_optimizer, '_optimize_database_costs', AsyncMock()) as mock_database:
                            with patch.object(infrastructure_optimizer, '_optimize_network_costs', AsyncMock()) as mock_network:
                                
                                # Setup mocks
                                mock_analyze.return_value = {'compute': {}, 'storage': {}, 'database': {}, 'network': {}}
                                mock_current.return_value = {'compute': 1000, 'storage': 500, 'database': 300, 'network': 200}
                                
                                # Create sample optimizations
                                sample_opt = CostOptimization(
                                    strategy=OptimizationStrategy.RIGHT_SIZING,
                                    category=CostCategory.COMPUTE,
                                    current_cost=1000,
                                    optimized_cost=800,
                                    savings_amount=200,
                                    savings_percentage=20,
                                    implementation_steps=[],
                                    risk_assessment="Low",
                                    implementation_time="1 week",
                                    confidence=0.8
                                )
                                
                                mock_compute.return_value = sample_opt
                                mock_storage.return_value = sample_opt._replace(category=CostCategory.STORAGE, current_cost=500, optimized_cost=400)
                                mock_database.return_value = sample_opt._replace(category=CostCategory.DATABASE, current_cost=300, optimized_cost=250)
                                mock_network.return_value = sample_opt._replace(category=CostCategory.NETWORK, current_cost=200, optimized_cost=180)
                                
                                result = await infrastructure_optimizer.optimize_infrastructure_costs()
                                
                                assert result.total_current_cost == 2000
                                assert result.total_optimized_cost < result.total_current_cost
                                assert result.total_savings > 0
                                assert len(result.service_optimizations) == 4
    
    @pytest.mark.asyncio
    async def test_optimize_compute_costs(self, infrastructure_optimizer):
        """Test compute cost optimization"""
        compute_data = {
            'instances': 10,
            'avg_cpu_utilization': 0.3,  # Low utilization
            'on_demand_percentage': 0.9   # High on-demand usage
        }
        
        optimization = await infrastructure_optimizer._optimize_compute_costs(compute_data)
        
        assert isinstance(optimization, CostOptimization)
        assert optimization.category == CostCategory.COMPUTE
        assert optimization.savings_amount > 0
        assert optimization.savings_percentage > 0
        assert len(optimization.implementation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_storage_costs(self, infrastructure_optimizer):
        """Test storage cost optimization"""
        storage_data = {
            'ssd_gb': 2000,
            'hdd_gb': 1000,
            'access_patterns': {'hot': 0.2, 'warm': 0.3, 'cold': 0.5}
        }
        
        optimization = await infrastructure_optimizer._optimize_storage_costs(storage_data)
        
        assert isinstance(optimization, CostOptimization)
        assert optimization.category == CostCategory.STORAGE
        assert optimization.savings_amount > 0
        assert 'archive' in ' '.join(optimization.implementation_steps).lower()

class TestCostForecastingEngine:
    """Test CostForecastingEngine class"""
    
    @pytest.mark.asyncio
    async def test_predict_and_budget_costs(self, forecasting_engine):
        """Test cost forecasting and budgeting"""
        with patch.object(forecasting_engine, '_load_cost_history', AsyncMock()) as mock_history:
            with patch.object(forecasting_engine, '_analyze_cost_trends', AsyncMock()) as mock_trends:
                with patch.object(forecasting_engine, '_predict_baseline_costs', AsyncMock()) as mock_baseline:
                    with patch.object(forecasting_engine, '_identify_future_optimizations', AsyncMock()) as mock_optimizations:
                        
                        # Setup mocks
                        mock_history.return_value = []
                        mock_trends.return_value = {}
                        mock_baseline.return_value = {'compute': 5000, 'ai_services': 3000}
                        mock_optimizations.return_value = []
                        
                        forecast = await forecasting_engine.predict_and_budget_costs(6)
                        
                        assert forecast.forecast_period == "6 months"
                        assert forecast.baseline_cost > 0
                        assert isinstance(forecast.confidence_interval, tuple)
                        assert len(forecast.confidence_interval) == 2
    
    @pytest.mark.asyncio
    async def test_load_cost_history(self, forecasting_engine):
        """Test historical cost data loading"""
        history = await forecasting_engine._load_cost_history()
        
        assert len(history) == 12  # 12 months
        assert all('date' in month for month in history)
        assert all('total' in month for month in history)
        
        # Check that totals are calculated
        for month in history:
            calculated_total = sum(v for k, v in month.items() if k not in ['date', 'total'])
            assert abs(month['total'] - calculated_total) < 0.01
    
    @pytest.mark.asyncio
    async def test_predict_baseline_costs(self, forecasting_engine):
        """Test baseline cost prediction"""
        # Create sample history
        history = []
        for i in range(12):
            history.append({
                'date': datetime.now() - timedelta(days=30*i),
                'compute': 1000 + i * 100,  # Growing trend
                'ai_services': 500 + i * 50
            })
        
        predictions = await forecasting_engine._predict_baseline_costs(history, 12)
        
        assert 'compute' in predictions
        assert 'ai_services' in predictions
        assert all(cost > 0 for cost in predictions.values())
        
        # AI services should have higher growth
        assert predictions['ai_services'] > history[-1]['ai_services']

class TestAIOperationsCostOptimizer:
    """Test main AIOperationsCostOptimizer class"""
    
    @pytest.mark.asyncio
    async def test_comprehensive_cost_optimization(self, cost_optimizer):
        """Test comprehensive cost optimization"""
        with patch.object(cost_optimizer, 'optimize_ai_provider_usage', AsyncMock()) as mock_ai:
            with patch.object(cost_optimizer, 'optimize_infrastructure_costs', AsyncMock()) as mock_infra:
                with patch.object(cost_optimizer, 'predict_and_budget_costs', AsyncMock()) as mock_forecast:
                    
                    # Setup mocks
                    ai_opt = CostOptimization(
                        strategy=OptimizationStrategy.PROVIDER_OPTIMIZATION,
                        category=CostCategory.AI_SERVICES,
                        current_cost=2000,
                        optimized_cost=1500,
                        savings_amount=500,
                        savings_percentage=25,
                        implementation_steps=[],
                        risk_assessment="Low",
                        implementation_time="2 weeks",
                        confidence=0.8
                    )
                    mock_ai.return_value = ai_opt
                    
                    # Mock infrastructure optimization
                    from src.operations.optimization.cost_optimization import InfrastructureOptimization
                    infra_opt = InfrastructureOptimization(
                        service_optimizations={},
                        total_current_cost=5000,
                        total_optimized_cost=4000,
                        total_savings=1000,
                        payback_period="3 months",
                        implementation_priority=[]
                    )
                    mock_infra.return_value = infra_opt
                    
                    # Mock forecast
                    from src.operations.optimization.cost_optimization import CostForecast
                    forecast = CostForecast(
                        forecast_period="12 months",
                        baseline_cost=10000,
                        optimized_cost=8000,
                        projected_savings=2000,
                        cost_trends={},
                        optimization_opportunities=[],
                        confidence_interval=(8000, 12000)
                    )
                    mock_forecast.return_value = forecast
                    
                    result = await cost_optimizer.comprehensive_cost_optimization()
                    
                    assert 'ai_optimization' in result
                    assert 'infrastructure_optimization' in result
                    assert 'cost_forecast' in result
                    assert 'total_savings' in result
                    assert 'savings_percentage' in result
                    assert result['total_savings'] == 1500  # 500 + 1000
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_recommendations(self, cost_optimizer):
        """Test comprehensive recommendation generation"""
        # Create mock optimizations
        ai_opt = Mock()
        ai_opt.savings_percentage = 25.0
        ai_opt.savings_amount = 500.0
        
        infra_opt = Mock()
        infra_opt.total_savings = 1200.0
        infra_opt.implementation_priority = ['compute', 'storage']
        
        forecast = Mock()
        forecast.projected_savings = 2000.0
        forecast.baseline_cost = 10000.0
        
        recommendations = await cost_optimizer._generate_comprehensive_recommendations(ai_opt, infra_opt, forecast)
        
        assert len(recommendations) > 0
        assert any('AI optimization' in rec for rec in recommendations)
        assert any('Infrastructure optimization' in rec for rec in recommendations)
        assert any('compute, storage' in rec for rec in recommendations)

@pytest.mark.asyncio
async def test_get_cost_optimizer():
    """Test cost optimizer singleton getter"""
    from src.operations.optimization.cost_optimization import get_cost_optimizer
    
    optimizer = await get_cost_optimizer()
    assert optimizer is not None
    assert isinstance(optimizer, AIOperationsCostOptimizer)
