"""
AI-Powered Cost Optimization System

This module implements intelligent cost optimization with machine learning-based
analysis, predictive cost modeling, and automated optimization strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import json
import redis.asyncio as redis

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class OptimizationStrategy(Enum):
    """Cost optimization strategies"""
    RIGHT_SIZING = "right_sizing"
    SPOT_INSTANCES = "spot_instances"
    RESERVED_CAPACITY = "reserved_capacity"
    AUTO_SCALING = "auto_scaling"
    INTELLIGENT_CACHING = "intelligent_caching"
    WORKLOAD_SCHEDULING = "workload_scheduling"
    PROVIDER_OPTIMIZATION = "provider_optimization"

class CostCategory(Enum):
    """Cost categories for optimization"""
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"
    AI_SERVICES = "ai_services"
    DATABASE = "database"
    MONITORING = "monitoring"
    SECURITY = "security"

@dataclass
class CostOptimization:
    """Cost optimization result"""
    strategy: OptimizationStrategy
    category: CostCategory
    current_cost: float
    optimized_cost: float
    savings_amount: float
    savings_percentage: float
    implementation_steps: List[str]
    risk_assessment: str
    implementation_time: str
    confidence: float

@dataclass
class CostForecast:
    """Cost forecast with predictions"""
    forecast_period: str
    baseline_cost: float
    optimized_cost: float
    projected_savings: float
    cost_trends: Dict[str, List[float]]
    optimization_opportunities: List[CostOptimization]
    confidence_interval: Tuple[float, float]

@dataclass
class InfrastructureOptimization:
    """Infrastructure cost optimization"""
    service_optimizations: Dict[str, CostOptimization]
    total_current_cost: float
    total_optimized_cost: float
    total_savings: float
    payback_period: str
    implementation_priority: List[str]

class AIProviderCostOptimizer:
    """Optimizes AI provider costs through intelligent routing and caching"""
    
    def __init__(self):
        self.provider_costs = self._initialize_provider_costs()
        self.usage_patterns = {}
        self.cache_hit_rates = {}
        self.model_performance = {}
    
    def _initialize_provider_costs(self) -> Dict[str, Dict[str, float]]:
        """Initialize AI provider cost structure"""
        return {
            'openai': {
                'gpt-4': 0.03,          # per 1K tokens
                'gpt-3.5-turbo': 0.002,  # per 1K tokens
                'embeddings': 0.0001     # per 1K tokens
            },
            'anthropic': {
                'claude-3': 0.025,      # per 1K tokens
                'claude-instant': 0.003  # per 1K tokens
            },
            'azure': {
                'gpt-4': 0.028,         # per 1K tokens
                'gpt-3.5-turbo': 0.0018  # per 1K tokens
            }
        }
    
    async def optimize_ai_provider_usage(self) -> CostOptimization:
        """Optimize AI provider costs"""
        try:
            # Analyze current usage patterns
            usage_analysis = await self._analyze_ai_usage_patterns()
            
            # Calculate current costs
            current_cost = await self._calculate_current_ai_costs(usage_analysis)
            
            # Identify optimization opportunities
            optimizations = await self._identify_ai_optimizations(usage_analysis)
            
            # Calculate optimized costs
            optimized_cost = await self._calculate_optimized_ai_costs(usage_analysis, optimizations)
            
            # Calculate savings
            savings_amount = current_cost - optimized_cost
            savings_percentage = (savings_amount / current_cost * 100) if current_cost > 0 else 0
            
            return CostOptimization(
                strategy=OptimizationStrategy.PROVIDER_OPTIMIZATION,
                category=CostCategory.AI_SERVICES,
                current_cost=current_cost,
                optimized_cost=optimized_cost,
                savings_amount=savings_amount,
                savings_percentage=savings_percentage,
                implementation_steps=await self._generate_ai_optimization_steps(optimizations),
                risk_assessment="Low - Quality maintained through intelligent routing",
                implementation_time="1-2 weeks",
                confidence=0.85
            )
            
        except Exception as e:
            logger.error(f"AI provider optimization failed: {e}")
            return await self._get_fallback_ai_optimization()
    
    async def _analyze_ai_usage_patterns(self) -> Dict[str, Any]:
        """Analyze AI usage patterns for optimization"""
        # Simulated usage analysis - in production, this would analyze real usage
        return {
            'total_tokens_per_day': 500000,
            'model_distribution': {
                'gpt-4': 0.3,           # 30% of usage
                'gpt-3.5-turbo': 0.6,   # 60% of usage
                'claude-3': 0.1        # 10% of usage
            },
            'request_types': {
                'reasoning': 0.4,       # Complex reasoning tasks
                'conversation': 0.4,    # Regular conversations
                'simple_queries': 0.2   # Simple Q&A
            },
            'peak_hours': [9, 10, 11, 14, 15, 16],
            'cache_potential': 0.35,    # 35% of requests could be cached
            'quality_requirements': {
                'reasoning': 'high',     # Requires GPT-4
                'conversation': 'medium', # Can use GPT-3.5
                'simple_queries': 'low'  # Can use cheapest models
            }
        }
    
    async def _calculate_current_ai_costs(self, usage: Dict[str, Any]) -> float:
        """Calculate current AI service costs"""
        total_tokens = usage['total_tokens_per_day']
        model_distribution = usage['model_distribution']
        
        daily_cost = 0
        for model, percentage in model_distribution.items():
            tokens_used = total_tokens * percentage
            
            if model == 'gpt-4':
                cost_per_k = self.provider_costs['openai']['gpt-4']
            elif model == 'gpt-3.5-turbo':
                cost_per_k = self.provider_costs['openai']['gpt-3.5-turbo']
            elif model == 'claude-3':
                cost_per_k = self.provider_costs['anthropic']['claude-3']
            else:
                cost_per_k = 0.02  # Default cost
            
            daily_cost += (tokens_used / 1000) * cost_per_k
        
        return daily_cost * 30  # Monthly cost
    
    async def _identify_ai_optimizations(self, usage: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify AI cost optimization opportunities"""
        optimizations = []
        
        # Optimization 1: Intelligent model selection
        optimizations.append({
            'type': 'intelligent_model_selection',
            'description': 'Route requests to optimal models based on complexity',
            'potential_savings': 0.25,  # 25% cost reduction
            'implementation': 'request_routing_engine'
        })
        
        # Optimization 2: Enhanced caching
        cache_potential = usage.get('cache_potential', 0.3)
        if cache_potential > 0.2:
            optimizations.append({
                'type': 'enhanced_caching',
                'description': 'Implement intelligent response caching',
                'potential_savings': cache_potential * 0.9,  # 90% of cacheable requests
                'implementation': 'semantic_cache_system'
            })
        
        # Optimization 3: Batch processing
        optimizations.append({
            'type': 'batch_processing',
            'description': 'Batch non-urgent requests for efficiency',
            'potential_savings': 0.15,  # 15% cost reduction
            'implementation': 'request_batching_system'
        })
        
        # Optimization 4: Provider arbitrage
        optimizations.append({
            'type': 'provider_arbitrage',
            'description': 'Route to most cost-effective providers',
            'potential_savings': 0.20,  # 20% cost reduction
            'implementation': 'multi_provider_routing'
        })
        
        return optimizations
    
    async def _calculate_optimized_ai_costs(self, usage: Dict[str, Any], optimizations: List[Dict[str, Any]]) -> float:
        """Calculate optimized AI costs after applying optimizations"""
        current_cost = await self._calculate_current_ai_costs(usage)
        
        total_savings_rate = 0
        for optimization in optimizations:
            savings_rate = optimization.get('potential_savings', 0)
            # Apply diminishing returns for multiple optimizations
            effective_savings = savings_rate * (1 - total_savings_rate)
            total_savings_rate += effective_savings
        
        # Cap total savings at 60% to be realistic
        total_savings_rate = min(total_savings_rate, 0.6)
        
        return current_cost * (1 - total_savings_rate)
    
    async def _generate_ai_optimization_steps(self, optimizations: List[Dict[str, Any]]) -> List[str]:
        """Generate implementation steps for AI optimizations"""
        steps = []
        
        for optimization in optimizations:
            opt_type = optimization['type']
            
            if opt_type == 'intelligent_model_selection':
                steps.extend([
                    "Implement request complexity analysis",
                    "Create model routing rules based on task requirements",
                    "Add quality monitoring for routed requests"
                ])
            elif opt_type == 'enhanced_caching':
                steps.extend([
                    "Implement semantic similarity caching",
                    "Add cache invalidation strategies",
                    "Monitor cache hit rates and optimize"
                ])
            elif opt_type == 'batch_processing':
                steps.extend([
                    "Identify batchable request types",
                    "Implement request queuing and batching",
                    "Add batch processing scheduling"
                ])
            elif opt_type == 'provider_arbitrage':
                steps.extend([
                    "Set up multi-provider infrastructure",
                    "Implement cost-aware routing logic",
                    "Add failover and quality monitoring"
                ])
        
        return steps
    
    async def _get_fallback_ai_optimization(self) -> CostOptimization:
        """Get fallback AI optimization when analysis fails"""
        return CostOptimization(
            strategy=OptimizationStrategy.PROVIDER_OPTIMIZATION,
            category=CostCategory.AI_SERVICES,
            current_cost=1500.0,
            optimized_cost=1200.0,
            savings_amount=300.0,
            savings_percentage=20.0,
            implementation_steps=["Review AI provider usage", "Implement basic caching"],
            risk_assessment="Low",
            implementation_time="2-3 weeks",
            confidence=0.6
        )

class InfrastructureCostOptimizer:
    """Optimizes infrastructure costs through intelligent resource management"""
    
    def __init__(self):
        self.resource_costs = self._initialize_resource_costs()
        self.utilization_data = {}
        self.optimization_history = []
    
    def _initialize_resource_costs(self) -> Dict[str, Dict[str, float]]:
        """Initialize infrastructure cost structure"""
        return {
            'compute': {
                'on_demand_hourly': 0.10,
                'reserved_hourly': 0.06,
                'spot_hourly': 0.03
            },
            'storage': {
                'ssd_gb_month': 0.10,
                'hdd_gb_month': 0.045,
                'archive_gb_month': 0.004
            },
            'network': {
                'data_transfer_gb': 0.09,
                'load_balancer_hour': 0.025
            },
            'database': {
                'instance_hour': 0.15,
                'storage_gb_month': 0.20,
                'iops_month': 0.10
            }
        }
    
    async def optimize_infrastructure_costs(self) -> InfrastructureOptimization:
        """Optimize infrastructure costs comprehensively"""
        try:
            # Analyze current infrastructure utilization
            utilization = await self._analyze_infrastructure_utilization()
            
            # Calculate current costs
            current_costs = await self._calculate_current_infrastructure_costs(utilization)
            
            # Generate service-specific optimizations
            service_optimizations = {}
            
            # Compute optimization
            compute_opt = await self._optimize_compute_costs(utilization.get('compute', {}))
            service_optimizations['compute'] = compute_opt
            
            # Storage optimization
            storage_opt = await self._optimize_storage_costs(utilization.get('storage', {}))
            service_optimizations['storage'] = storage_opt
            
            # Database optimization
            database_opt = await self._optimize_database_costs(utilization.get('database', {}))
            service_optimizations['database'] = database_opt
            
            # Network optimization
            network_opt = await self._optimize_network_costs(utilization.get('network', {}))
            service_optimizations['network'] = network_opt
            
            # Calculate totals
            total_current_cost = sum(current_costs.values())
            total_optimized_cost = sum(opt.optimized_cost for opt in service_optimizations.values())
            total_savings = total_current_cost - total_optimized_cost
            
            # Determine implementation priority
            implementation_priority = self._prioritize_optimizations(service_optimizations)
            
            # Calculate payback period
            payback_period = await self._calculate_payback_period(service_optimizations, total_savings)
            
            return InfrastructureOptimization(
                service_optimizations=service_optimizations,
                total_current_cost=total_current_cost,
                total_optimized_cost=total_optimized_cost,
                total_savings=total_savings,
                payback_period=payback_period,
                implementation_priority=implementation_priority
            )
            
        except Exception as e:
            logger.error(f"Infrastructure optimization failed: {e}")
            return await self._get_fallback_infrastructure_optimization()
    
    async def _analyze_infrastructure_utilization(self) -> Dict[str, Dict[str, Any]]:
        """Analyze current infrastructure utilization"""
        # Simulated utilization data - in production, this would gather real metrics
        return {
            'compute': {
                'instances': 12,
                'avg_cpu_utilization': 0.45,
                'avg_memory_utilization': 0.52,
                'peak_usage_hours': [9, 10, 11, 14, 15, 16],
                'instance_types': {'m5.large': 8, 'm5.xlarge': 4},
                'on_demand_percentage': 0.8,
                'reserved_percentage': 0.2
            },
            'storage': {
                'total_gb': 5000,
                'ssd_gb': 3000,
                'hdd_gb': 2000,
                'utilization_percentage': 0.65,
                'growth_rate_monthly': 0.15,
                'access_patterns': {'hot': 0.3, 'warm': 0.4, 'cold': 0.3}
            },
            'database': {
                'instance_hours_monthly': 720,  # 24/7 operation
                'storage_gb': 1000,
                'avg_cpu_utilization': 0.35,
                'iops_provisioned': 3000,
                'iops_utilized': 1200
            },
            'network': {
                'data_transfer_gb_monthly': 2000,
                'load_balancer_hours': 720,
                'cdn_usage': 0.3  # 30% of traffic uses CDN
            }
        }
    
    async def _calculate_current_infrastructure_costs(self, utilization: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate current infrastructure costs"""
        costs = {}
        
        # Compute costs
        compute_data = utilization.get('compute', {})
        compute_instances = compute_data.get('instances', 0)
        on_demand_pct = compute_data.get('on_demand_percentage', 1.0)
        costs['compute'] = compute_instances * 720 * (on_demand_pct * 0.10 + (1-on_demand_pct) * 0.06)
        
        # Storage costs
        storage_data = utilization.get('storage', {})
        ssd_gb = storage_data.get('ssd_gb', 0)
        hdd_gb = storage_data.get('hdd_gb', 0)
        costs['storage'] = ssd_gb * 0.10 + hdd_gb * 0.045
        
        # Database costs
        db_data = utilization.get('database', {})
        db_hours = db_data.get('instance_hours_monthly', 720)
        db_storage = db_data.get('storage_gb', 0)
        costs['database'] = db_hours * 0.15 + db_storage * 0.20
        
        # Network costs
        network_data = utilization.get('network', {})
        data_transfer = network_data.get('data_transfer_gb_monthly', 0)
        lb_hours = network_data.get('load_balancer_hours', 720)
        costs['network'] = data_transfer * 0.09 + lb_hours * 0.025
        
        return costs
    
    async def _optimize_compute_costs(self, compute_data: Dict[str, Any]) -> CostOptimization:
        """Optimize compute costs"""
        instances = compute_data.get('instances', 12)
        avg_cpu = compute_data.get('avg_cpu_utilization', 0.45)
        on_demand_pct = compute_data.get('on_demand_percentage', 0.8)
        
        # Current cost
        current_cost = instances * 720 * (on_demand_pct * 0.10 + (1-on_demand_pct) * 0.06)
        
        # Optimization strategies
        optimizations = []
        
        # Right-sizing: reduce instances if utilization is low
        if avg_cpu < 0.5:
            right_sized_instances = max(6, int(instances * 0.75))  # 25% reduction
            optimizations.append(('right_sizing', right_sized_instances - instances))
        else:
            right_sized_instances = instances
        
        # Reserved instances: increase reserved percentage
        optimized_reserved_pct = min(0.6, on_demand_pct + 0.3)  # Increase reserved to 60%
        
        # Spot instances for non-critical workloads (20% of capacity)
        spot_percentage = 0.2
        
        # Calculate optimized cost
        optimized_instances = right_sized_instances
        optimized_cost = optimized_instances * 720 * (
            (1 - optimized_reserved_pct - spot_percentage) * 0.10 +  # On-demand
            optimized_reserved_pct * 0.06 +                         # Reserved
            spot_percentage * 0.03                                   # Spot
        )
        
        savings_amount = current_cost - optimized_cost
        savings_percentage = (savings_amount / current_cost * 100) if current_cost > 0 else 0
        
        implementation_steps = []
        if right_sized_instances < instances:
            implementation_steps.append(f"Right-size from {instances} to {right_sized_instances} instances")
        implementation_steps.extend([
            f"Increase reserved instance usage to {optimized_reserved_pct*100:.0f}%",
            f"Implement spot instances for {spot_percentage*100:.0f}% of non-critical workloads",
            "Set up auto-scaling policies based on actual usage patterns"
        ])
        
        return CostOptimization(
            strategy=OptimizationStrategy.RIGHT_SIZING,
            category=CostCategory.COMPUTE,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings_amount=savings_amount,
            savings_percentage=savings_percentage,
            implementation_steps=implementation_steps,
            risk_assessment="Medium - Requires careful monitoring of performance",
            implementation_time="2-4 weeks",
            confidence=0.80
        )
    
    async def _optimize_storage_costs(self, storage_data: Dict[str, Any]) -> CostOptimization:
        """Optimize storage costs"""
        ssd_gb = storage_data.get('ssd_gb', 3000)
        hdd_gb = storage_data.get('hdd_gb', 2000)
        access_patterns = storage_data.get('access_patterns', {'hot': 0.3, 'warm': 0.4, 'cold': 0.3})
        
        # Current cost
        current_cost = ssd_gb * 0.10 + hdd_gb * 0.045
        
        # Optimization: Move cold data to archive storage
        cold_percentage = access_patterns.get('cold', 0.3)
        
        # Move 80% of cold data to archive storage
        data_to_archive = (ssd_gb + hdd_gb) * cold_percentage * 0.8
        
        # Optimized storage allocation
        optimized_ssd = max(1000, ssd_gb - data_to_archive * 0.6)  # Keep some SSD for hot data
        optimized_hdd = max(500, hdd_gb - data_to_archive * 0.4)   # Keep some HDD for warm data
        optimized_archive = data_to_archive
        
        # Calculate optimized cost
        optimized_cost = (
            optimized_ssd * 0.10 +      # SSD cost
            optimized_hdd * 0.045 +     # HDD cost
            optimized_archive * 0.004   # Archive cost
        )
        
        savings_amount = current_cost - optimized_cost
        savings_percentage = (savings_amount / current_cost * 100) if current_cost > 0 else 0
        
        return CostOptimization(
            strategy=OptimizationStrategy.INTELLIGENT_CACHING,
            category=CostCategory.STORAGE,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings_amount=savings_amount,
            savings_percentage=savings_percentage,
            implementation_steps=[
                "Analyze data access patterns over 30-day period",
                f"Migrate {data_to_archive:.0f}GB of cold data to archive storage",
                "Implement automated data lifecycle policies",
                "Set up monitoring for access pattern changes"
            ],
            risk_assessment="Low - Archive storage provides same durability",
            implementation_time="1-2 weeks",
            confidence=0.85
        )
    
    async def _optimize_database_costs(self, db_data: Dict[str, Any]) -> CostOptimization:
        """Optimize database costs"""
        instance_hours = db_data.get('instance_hours_monthly', 720)
        storage_gb = db_data.get('storage_gb', 1000)
        avg_cpu = db_data.get('avg_cpu_utilization', 0.35)
        iops_provisioned = db_data.get('iops_provisioned', 3000)
        iops_utilized = db_data.get('iops_utilized', 1200)
        
        # Current cost
        current_cost = instance_hours * 0.15 + storage_gb * 0.20 + iops_provisioned * 0.10 / 1000
        
        # Optimization strategies
        
        # Right-size instance if CPU utilization is low
        if avg_cpu < 0.4:
            optimized_instance_cost = instance_hours * 0.10  # Smaller instance type
        else:
            optimized_instance_cost = instance_hours * 0.15
        
        # Optimize IOPS provisioning
        optimized_iops = max(1000, int(iops_utilized * 1.3))  # 30% buffer above utilization
        
        # Storage optimization (move old data to cheaper storage)
        optimized_storage_cost = storage_gb * 0.15  # Mix of regular and cheaper storage
        
        optimized_cost = (
            optimized_instance_cost +
            optimized_storage_cost +
            optimized_iops * 0.10 / 1000
        )
        
        savings_amount = current_cost - optimized_cost
        savings_percentage = (savings_amount / current_cost * 100) if current_cost > 0 else 0
        
        implementation_steps = []
        if avg_cpu < 0.4:
            implementation_steps.append("Right-size database instance to smaller type")
        implementation_steps.extend([
            f"Reduce provisioned IOPS from {iops_provisioned} to {optimized_iops}",
            "Implement data archiving for historical records",
            "Set up database performance monitoring"
        ])
        
        return CostOptimization(
            strategy=OptimizationStrategy.RIGHT_SIZING,
            category=CostCategory.DATABASE,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings_amount=savings_amount,
            savings_percentage=savings_percentage,
            implementation_steps=implementation_steps,
            risk_assessment="Medium - Requires performance testing",
            implementation_time="1-3 weeks",
            confidence=0.75
        )
    
    async def _optimize_network_costs(self, network_data: Dict[str, Any]) -> CostOptimization:
        """Optimize network costs"""
        data_transfer = network_data.get('data_transfer_gb_monthly', 2000)
        lb_hours = network_data.get('load_balancer_hours', 720)
        cdn_usage = network_data.get('cdn_usage', 0.3)
        
        # Current cost
        current_cost = data_transfer * 0.09 + lb_hours * 0.025
        
        # Optimization: Increase CDN usage to reduce data transfer costs
        optimized_cdn_usage = min(0.7, cdn_usage + 0.3)  # Increase CDN usage
        
        # CDN reduces data transfer costs by 60% for cached content
        cdn_savings = (optimized_cdn_usage - cdn_usage) * data_transfer * 0.09 * 0.6
        
        # Optimize load balancer usage (use application load balancer more efficiently)
        optimized_lb_cost = lb_hours * 0.020  # 20% reduction through optimization
        
        optimized_cost = (
            data_transfer * 0.09 - cdn_savings +
            optimized_lb_cost
        )
        
        savings_amount = current_cost - optimized_cost
        savings_percentage = (savings_amount / current_cost * 100) if current_cost > 0 else 0
        
        return CostOptimization(
            strategy=OptimizationStrategy.INTELLIGENT_CACHING,
            category=CostCategory.NETWORK,
            current_cost=current_cost,
            optimized_cost=optimized_cost,
            savings_amount=savings_amount,
            savings_percentage=savings_percentage,
            implementation_steps=[
                f"Increase CDN usage from {cdn_usage*100:.0f}% to {optimized_cdn_usage*100:.0f}%",
                "Optimize load balancer configuration for efficiency",
                "Implement intelligent caching strategies",
                "Monitor data transfer patterns and costs"
            ],
            risk_assessment="Low - CDN improves performance while reducing costs",
            implementation_time="1-2 weeks",
            confidence=0.90
        )
    
    def _prioritize_optimizations(self, optimizations: Dict[str, CostOptimization]) -> List[str]:
        """Prioritize optimizations by impact and ease of implementation"""
        priority_scores = {}
        
        for service, optimization in optimizations.items():
            # Score based on savings percentage and confidence
            impact_score = optimization.savings_percentage * optimization.confidence
            
            # Adjust for implementation complexity (lower complexity = higher priority)
            complexity_factor = {
                "1-2 weeks": 1.0,
                "2-3 weeks": 0.9,
                "1-3 weeks": 0.8,
                "2-4 weeks": 0.7
            }.get(optimization.implementation_time, 0.6)
            
            priority_scores[service] = impact_score * complexity_factor
        
        # Sort by priority score (highest first)
        return sorted(priority_scores.keys(), key=lambda x: priority_scores[x], reverse=True)
    
    async def _calculate_payback_period(self, optimizations: Dict[str, CostOptimization], total_savings: float) -> str:
        """Calculate payback period for optimizations"""
        # Estimate implementation costs
        implementation_cost = 0
        for optimization in optimizations.values():
            # Rough estimate based on implementation time
            weeks = 2  # Average implementation time
            developer_cost_per_week = 2000  # $2000 per developer week
            implementation_cost += weeks * developer_cost_per_week
        
        if total_savings > 0:
            payback_months = implementation_cost / (total_savings / 12)  # Monthly savings
            if payback_months < 1:
                return "Less than 1 month"
            elif payback_months < 12:
                return f"{payback_months:.1f} months"
            else:
                return f"{payback_months/12:.1f} years"
        else:
            return "No payback - costs exceed savings"
    
    async def _get_fallback_infrastructure_optimization(self) -> InfrastructureOptimization:
        """Get fallback infrastructure optimization"""
        fallback_opt = CostOptimization(
            strategy=OptimizationStrategy.RIGHT_SIZING,
            category=CostCategory.COMPUTE,
            current_cost=5000.0,
            optimized_cost=4000.0,
            savings_amount=1000.0,
            savings_percentage=20.0,
            implementation_steps=["Review resource utilization"],
            risk_assessment="Low",
            implementation_time="2-3 weeks",
            confidence=0.6
        )
        
        return InfrastructureOptimization(
            service_optimizations={'compute': fallback_opt},
            total_current_cost=5000.0,
            total_optimized_cost=4000.0,
            total_savings=1000.0,
            payback_period="2-3 months",
            implementation_priority=['compute']
        )

class CostForecastingEngine:
    """Predictive cost forecasting with trend analysis"""
    
    def __init__(self):
        self.cost_history = []
        self.usage_predictors = {}
        self.seasonal_factors = {}
    
    async def predict_and_budget_costs(self, forecast_months: int = 12) -> CostForecast:
        """Predict future costs and provide budget recommendations"""
        try:
            # Load historical cost data
            cost_history = await self._load_cost_history()
            
            # Analyze cost trends
            cost_trends = await self._analyze_cost_trends(cost_history)
            
            # Predict baseline costs
            baseline_costs = await self._predict_baseline_costs(cost_history, forecast_months)
            
            # Identify optimization opportunities
            optimization_opportunities = await self._identify_future_optimizations(baseline_costs)
            
            # Calculate optimized forecast
            optimized_costs = await self._calculate_optimized_forecast(baseline_costs, optimization_opportunities)
            
            # Calculate projected savings
            projected_savings = sum(baseline_costs.values()) - sum(optimized_costs.values())
            
            # Calculate confidence interval
            confidence_interval = await self._calculate_confidence_interval(baseline_costs)
            
            return CostForecast(
                forecast_period=f"{forecast_months} months",
                baseline_cost=sum(baseline_costs.values()),
                optimized_cost=sum(optimized_costs.values()),
                projected_savings=projected_savings,
                cost_trends=cost_trends,
                optimization_opportunities=optimization_opportunities,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"Cost forecasting failed: {e}")
            return await self._get_fallback_forecast()
    
    async def _load_cost_history(self) -> List[Dict[str, Any]]:
        """Load historical cost data"""
        # Simulated historical data - in production, this would load real cost data
        history = []
        base_date = datetime.now() - timedelta(days=365)
        
        for month in range(12):
            month_date = base_date + timedelta(days=month * 30)
            
            # Simulate cost growth and seasonal patterns
            growth_factor = 1 + (month * 0.02)  # 2% monthly growth
            seasonal_factor = 1 + 0.1 * np.sin(month * np.pi / 6)  # Seasonal variation
            
            history.append({
                'date': month_date,
                'compute': 3000 * growth_factor * seasonal_factor,
                'storage': 1500 * growth_factor,
                'network': 800 * growth_factor * seasonal_factor,
                'ai_services': 2000 * growth_factor * (1 + month * 0.05),  # Higher AI growth
                'database': 1200 * growth_factor,
                'total': 0  # Will be calculated
            })
            
            # Calculate total
            history[-1]['total'] = sum(v for k, v in history[-1].items() if k not in ['date', 'total'])
        
        return history
    
    async def _analyze_cost_trends(self, history: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Analyze cost trends from historical data"""
        trends = {}
        categories = ['compute', 'storage', 'network', 'ai_services', 'database', 'total']
        
        for category in categories:
            values = [month[category] for month in history]
            
            # Calculate month-over-month growth rates
            growth_rates = []
            for i in range(1, len(values)):
                growth_rate = (values[i] - values[i-1]) / values[i-1] if values[i-1] > 0 else 0
                growth_rates.append(growth_rate)
            
            trends[category] = values  # Return actual values for trend visualization
        
        return trends
    
    async def _predict_baseline_costs(self, history: List[Dict[str, Any]], months: int) -> Dict[str, float]:
        """Predict baseline costs using trend analysis"""
        categories = ['compute', 'storage', 'network', 'ai_services', 'database']
        predictions = {}
        
        for category in categories:
            values = [month[category] for month in history]
            
            # Simple linear trend prediction
            x = np.arange(len(values)).reshape(-1, 1)
            y = np.array(values)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(x, y)
            
            # Predict future value
            future_x = np.array([[len(values) + months - 1]])
            predicted_value = model.predict(future_x)[0]
            
            # Add some growth factor for AI services (higher growth expected)
            if category == 'ai_services':
                predicted_value *= 1.5  # 50% additional growth for AI
            
            predictions[category] = max(0, predicted_value)  # Ensure non-negative
        
        return predictions
    
    async def _identify_future_optimizations(self, baseline_costs: Dict[str, float]) -> List[CostOptimization]:
        """Identify future optimization opportunities"""
        optimizations = []
        
        # AI services optimization
        ai_cost = baseline_costs.get('ai_services', 0)
        if ai_cost > 1000:
            optimizations.append(CostOptimization(
                strategy=OptimizationStrategy.PROVIDER_OPTIMIZATION,
                category=CostCategory.AI_SERVICES,
                current_cost=ai_cost,
                optimized_cost=ai_cost * 0.7,  # 30% savings
                savings_amount=ai_cost * 0.3,
                savings_percentage=30.0,
                implementation_steps=["Implement intelligent AI provider routing"],
                risk_assessment="Low",
                implementation_time="2-3 weeks",
                confidence=0.8
            ))
        
        # Compute optimization
        compute_cost = baseline_costs.get('compute', 0)
        if compute_cost > 2000:
            optimizations.append(CostOptimization(
                strategy=OptimizationStrategy.AUTO_SCALING,
                category=CostCategory.COMPUTE,
                current_cost=compute_cost,
                optimized_cost=compute_cost * 0.75,  # 25% savings
                savings_amount=compute_cost * 0.25,
                savings_percentage=25.0,
                implementation_steps=["Implement advanced auto-scaling policies"],
                risk_assessment="Medium",
                implementation_time="3-4 weeks",
                confidence=0.75
            ))
        
        # Storage optimization
        storage_cost = baseline_costs.get('storage', 0)
        if storage_cost > 1000:
            optimizations.append(CostOptimization(
                strategy=OptimizationStrategy.INTELLIGENT_CACHING,
                category=CostCategory.STORAGE,
                current_cost=storage_cost,
                optimized_cost=storage_cost * 0.8,  # 20% savings
                savings_amount=storage_cost * 0.2,
                savings_percentage=20.0,
                implementation_steps=["Implement intelligent data lifecycle management"],
                risk_assessment="Low",
                implementation_time="2-3 weeks",
                confidence=0.85
            ))
        
        return optimizations
    
    async def _calculate_optimized_forecast(self, baseline_costs: Dict[str, float], optimizations: List[CostOptimization]) -> Dict[str, float]:
        """Calculate optimized cost forecast"""
        optimized_costs = baseline_costs.copy()
        
        for optimization in optimizations:
            category = optimization.category.value
            if category in optimized_costs:
                optimized_costs[category] = optimization.optimized_cost
        
        return optimized_costs
    
    async def _calculate_confidence_interval(self, baseline_costs: Dict[str, float]) -> Tuple[float, float]:
        """Calculate confidence interval for cost predictions"""
        total_cost = sum(baseline_costs.values())
        
        # Use 20% margin for confidence interval
        lower_bound = total_cost * 0.8
        upper_bound = total_cost * 1.2
        
        return (lower_bound, upper_bound)
    
    async def _get_fallback_forecast(self) -> CostForecast:
        """Get fallback cost forecast"""
        return CostForecast(
            forecast_period="12 months",
            baseline_cost=10000.0,
            optimized_cost=8000.0,
            projected_savings=2000.0,
            cost_trends={},
            optimization_opportunities=[],
            confidence_interval=(8000.0, 12000.0)
        )

class AIOperationsCostOptimizer:
    """Main AI-powered cost optimization system"""
    
    def __init__(self):
        self.ai_optimizer = AIProviderCostOptimizer()
        self.infrastructure_optimizer = InfrastructureCostOptimizer()
        self.forecasting_engine = CostForecastingEngine()
        self.optimization_history = []
    
    async def optimize_ai_provider_usage(self) -> CostOptimization:
        """Optimize AI provider costs"""
        return await self.ai_optimizer.optimize_ai_provider_usage()
    
    async def optimize_infrastructure_costs(self) -> InfrastructureOptimization:
        """Optimize infrastructure costs"""
        return await self.infrastructure_optimizer.optimize_infrastructure_costs()
    
    async def predict_and_budget_costs(self, forecast_months: int = 12) -> CostForecast:
        """Predict and budget future costs"""
        return await self.forecasting_engine.predict_and_budget_costs(forecast_months)
    
    async def comprehensive_cost_optimization(self) -> Dict[str, Any]:
        """Perform comprehensive cost optimization across all categories"""
        try:
            # Get all optimizations
            ai_optimization = await self.optimize_ai_provider_usage()
            infrastructure_optimization = await self.optimize_infrastructure_costs()
            cost_forecast = await self.predict_and_budget_costs()
            
            # Calculate total potential savings
            total_current_cost = (
                ai_optimization.current_cost +
                infrastructure_optimization.total_current_cost
            )
            
            total_optimized_cost = (
                ai_optimization.optimized_cost +
                infrastructure_optimization.total_optimized_cost
            )
            
            total_savings = total_current_cost - total_optimized_cost
            savings_percentage = (total_savings / total_current_cost * 100) if total_current_cost > 0 else 0
            
            # Generate comprehensive recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                ai_optimization, infrastructure_optimization, cost_forecast
            )
            
            return {
                'ai_optimization': ai_optimization,
                'infrastructure_optimization': infrastructure_optimization,
                'cost_forecast': cost_forecast,
                'total_current_cost': total_current_cost,
                'total_optimized_cost': total_optimized_cost,
                'total_savings': total_savings,
                'savings_percentage': savings_percentage,
                'recommendations': recommendations,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive cost optimization failed: {e}")
            return {
                'error': str(e),
                'optimization_timestamp': datetime.now().isoformat()
            }
    
    async def _generate_comprehensive_recommendations(self, ai_opt: CostOptimization, infra_opt: InfrastructureOptimization, forecast: CostForecast) -> List[str]:
        """Generate comprehensive optimization recommendations"""
        recommendations = []
        
        # High-impact recommendations first
        if ai_opt.savings_percentage > 20:
            recommendations.append(f"HIGH PRIORITY: AI optimization can save {ai_opt.savings_percentage:.1f}% (${ai_opt.savings_amount:.0f}/month)")
        
        if infra_opt.total_savings > 1000:
            recommendations.append(f"Infrastructure optimization can save ${infra_opt.total_savings:.0f}/month")
        
        # Implementation priority recommendations
        recommendations.append(f"Implement optimizations in this order: {', '.join(infra_opt.implementation_priority)}")
        
        # Long-term recommendations based on forecast
        if forecast.projected_savings > forecast.baseline_cost * 0.15:
            recommendations.append(f"Long-term optimization potential: ${forecast.projected_savings:.0f}/year")
        
        # Specific action items
        recommendations.extend([
            "Set up automated cost monitoring and alerting",
            "Review cost optimization opportunities monthly",
            "Implement cost allocation tags for better tracking"
        ])
        
        return recommendations

# Global cost optimizer instance
cost_optimizer = AIOperationsCostOptimizer()

async def get_cost_optimizer() -> AIOperationsCostOptimizer:
    """Get cost optimizer instance"""
    return cost_optimizer
