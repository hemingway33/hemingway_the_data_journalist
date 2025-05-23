"""
Decision Framework for Rejection Inference Method Selection

This module provides structured guidance on when and which rejection inference
methods to use based on business context, data availability, and constraints.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class Priority(Enum):
    """Priority levels for different business objectives"""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"


@dataclass
class BusinessContext:
    """
    Represents the business context for method selection
    """
    rejection_rate: float  # 0.0 to 1.0
    sample_size: int
    external_data_available: bool
    external_data_quality: DataQuality
    interpretability_priority: Priority
    performance_priority: Priority
    regulatory_constraints: Priority
    computational_budget: Priority
    model_stability_requirement: Priority
    time_to_deploy: Priority  # HIGH = urgent, LOW = flexible


@dataclass
class MethodRecommendation:
    """
    Represents a method recommendation with rationale
    """
    primary_method: str
    backup_methods: List[str]
    confidence_level: str  # "High", "Medium", "Low"
    rationale: str
    considerations: List[str]
    expected_performance: str
    implementation_complexity: str


class RejectionInferenceDecisionFramework:
    """
    Main decision framework for rejection inference method selection
    """
    
    def __init__(self):
        self.method_profiles = self._initialize_method_profiles()
        self.scenario_rubrics = self._initialize_scenario_rubrics()
        
    def _initialize_method_profiles(self) -> Dict:
        """
        Initialize profiles for each rejection inference method
        """
        return {
            'approved_only': {
                'name': 'Approved Only (Traditional)',
                'interpretability': 'High',
                'performance': 'Low',
                'complexity': 'Low',
                'external_data_required': False,
                'best_for_rejection_rate': 'Low (<30%)',
                'regulatory_friendly': True,
                'stability': 'High',
                'computational_cost': 'Low'
            },
            'regularized_approved': {
                'name': 'Regularized Approved Only',
                'interpretability': 'High',
                'performance': 'Low-Medium',
                'complexity': 'Low',
                'external_data_required': False,
                'best_for_rejection_rate': 'Low-Medium (<40%)',
                'regulatory_friendly': True,
                'stability': 'High',
                'computational_cost': 'Low'
            },
            'conservative_propensity': {
                'name': 'Conservative Propensity Weighting',
                'interpretability': 'Medium-High',
                'performance': 'Medium',
                'complexity': 'Medium',
                'external_data_required': False,
                'best_for_rejection_rate': 'Medium (30-60%)',
                'regulatory_friendly': True,
                'stability': 'Medium-High',
                'computational_cost': 'Medium'
            },
            'ensemble_averaging': {
                'name': 'Ensemble Averaging',
                'interpretability': 'Medium',
                'performance': 'Medium',
                'complexity': 'Medium',
                'external_data_required': False,
                'best_for_rejection_rate': 'Medium (30-60%)',
                'regulatory_friendly': True,
                'stability': 'High',
                'computational_cost': 'Medium'
            },
            'simple_rejection_inference': {
                'name': 'Simple Rejection Inference',
                'interpretability': 'Low',
                'performance': 'Medium-High',
                'complexity': 'Medium',
                'external_data_required': False,
                'best_for_rejection_rate': 'High (50-80%)',
                'regulatory_friendly': False,
                'stability': 'Low',
                'computational_cost': 'Low'
            },
            'propensity_weighting': {
                'name': 'Propensity Weighting',
                'interpretability': 'Medium',
                'performance': 'Medium-High',
                'complexity': 'Medium',
                'external_data_required': False,
                'best_for_rejection_rate': 'Medium-High (40-70%)',
                'regulatory_friendly': True,
                'stability': 'Medium',
                'computational_cost': 'Medium'
            },
            'external_predictor': {
                'name': 'External Predictor Enhanced',
                'interpretability': 'Low-Medium',
                'performance': 'High',
                'complexity': 'High',
                'external_data_required': True,
                'best_for_rejection_rate': 'High (60-90%)',
                'regulatory_friendly': False,
                'stability': 'Medium',
                'computational_cost': 'High'
            },
            'hybrid_external_propensity': {
                'name': 'Hybrid External + Propensity',
                'interpretability': 'Medium',
                'performance': 'High',
                'complexity': 'High',
                'external_data_required': True,
                'best_for_rejection_rate': 'High (60-90%)',
                'regulatory_friendly': True,
                'stability': 'Medium',
                'computational_cost': 'High'
            }
        }
    
    def _initialize_scenario_rubrics(self) -> Dict:
        """
        Initialize scenario-based decision rubrics
        """
        return {
            'low_rejection_conservative': {
                'conditions': {
                    'rejection_rate': (0.0, 0.3),
                    'interpretability_priority': [Priority.HIGH, Priority.MEDIUM],
                    'regulatory_constraints': [Priority.HIGH, Priority.MEDIUM]
                },
                'recommended_methods': ['approved_only', 'regularized_approved'],
                'rationale': 'Low rejection rate minimizes bias; interpretability and regulatory requirements favor simple approaches'
            },
            'low_rejection_performance': {
                'conditions': {
                    'rejection_rate': (0.0, 0.3),
                    'performance_priority': [Priority.HIGH],
                    'interpretability_priority': [Priority.LOW, Priority.MEDIUM]
                },
                'recommended_methods': ['conservative_propensity', 'regularized_approved'],
                'rationale': 'Even with low rejection rate, propensity weighting can boost performance with acceptable interpretability trade-off'
            },
            'medium_rejection_balanced': {
                'conditions': {
                    'rejection_rate': (0.3, 0.6),
                    'interpretability_priority': [Priority.MEDIUM, Priority.HIGH],
                    'performance_priority': [Priority.MEDIUM, Priority.HIGH]
                },
                'recommended_methods': ['conservative_propensity', 'ensemble_averaging'],
                'rationale': 'Moderate rejection rate requires bias correction; balanced approaches offer good trade-offs'
            },
            'medium_rejection_performance': {
                'conditions': {
                    'rejection_rate': (0.3, 0.6),
                    'performance_priority': [Priority.HIGH],
                    'interpretability_priority': [Priority.LOW, Priority.MEDIUM]
                },
                'recommended_methods': ['propensity_weighting', 'simple_rejection_inference'],
                'rationale': 'Higher bias at medium rejection rates justifies more aggressive approaches for performance'
            },
            'high_rejection_interpretability': {
                'conditions': {
                    'rejection_rate': (0.6, 1.0),
                    'interpretability_priority': [Priority.HIGH],
                    'regulatory_constraints': [Priority.HIGH]
                },
                'recommended_methods': ['conservative_propensity', 'ensemble_averaging'],
                'rationale': 'High rejection creates severe bias, but interpretability constraints limit method choices'
            },
            'high_rejection_performance': {
                'conditions': {
                    'rejection_rate': (0.6, 1.0),
                    'performance_priority': [Priority.HIGH],
                    'interpretability_priority': [Priority.LOW, Priority.MEDIUM],
                    'external_data_available': False
                },
                'recommended_methods': ['simple_rejection_inference', 'propensity_weighting'],
                'rationale': 'High rejection rate requires aggressive bias correction; performance priority justifies complexity'
            },
            'external_data_available': {
                'conditions': {
                    'rejection_rate': (0.4, 1.0),
                    'external_data_available': True,
                    'external_data_quality': [DataQuality.GOOD, DataQuality.EXCELLENT],
                    'performance_priority': [Priority.HIGH]
                },
                'recommended_methods': ['hybrid_external_propensity', 'external_predictor'],
                'rationale': 'High-quality external data enables sophisticated methods with superior performance'
            },
            'startup_resource_constrained': {
                'conditions': {
                    'computational_budget': [Priority.LOW],
                    'time_to_deploy': [Priority.HIGH],
                    'sample_size': (0, 10000)
                },
                'recommended_methods': ['approved_only', 'regularized_approved'],
                'rationale': 'Resource constraints and urgency favor simple, proven approaches'
            },
            'enterprise_comprehensive': {
                'conditions': {
                    'computational_budget': [Priority.HIGH],
                    'sample_size': (50000, float('inf')),
                    'model_stability_requirement': [Priority.HIGH]
                },
                'recommended_methods': ['ensemble_averaging', 'conservative_propensity'],
                'rationale': 'Large scale and stability requirements favor robust ensemble approaches'
            }
        }
    
    def recommend_method(self, context: BusinessContext) -> MethodRecommendation:
        """
        Recommend the best rejection inference method based on business context
        
        Parameters:
        -----------
        context : BusinessContext
            Business context and constraints
            
        Returns:
        --------
        MethodRecommendation
            Detailed recommendation with rationale
        """
        # Score methods based on context
        method_scores = self._score_methods(context)
        
        # Find best matching scenarios
        matching_scenarios = self._find_matching_scenarios(context)
        
        # Combine scenario-based and scoring-based recommendations
        final_recommendation = self._synthesize_recommendation(
            method_scores, matching_scenarios, context
        )
        
        return final_recommendation
    
    def _score_methods(self, context: BusinessContext) -> Dict[str, float]:
        """
        Score each method based on how well it fits the context
        """
        scores = {}
        
        for method_id, profile in self.method_profiles.items():
            score = 0.0
            
            # Rejection rate compatibility
            if context.rejection_rate < 0.3:
                score += 3 if 'Low' in profile['best_for_rejection_rate'] else 1
            elif context.rejection_rate < 0.6:
                score += 3 if 'Medium' in profile['best_for_rejection_rate'] else 2
            else:
                score += 3 if 'High' in profile['best_for_rejection_rate'] else 1
            
            # External data requirement
            if profile['external_data_required'] and not context.external_data_available:
                score -= 5  # Heavy penalty
            elif profile['external_data_required'] and context.external_data_available:
                if context.external_data_quality in [DataQuality.GOOD, DataQuality.EXCELLENT]:
                    score += 2
                else:
                    score -= 1
            
            # Interpretability priority
            if context.interpretability_priority == Priority.HIGH:
                if profile['interpretability'] == 'High':
                    score += 3
                elif profile['interpretability'] in ['Medium-High', 'Medium']:
                    score += 1
                else:
                    score -= 2
            
            # Performance priority
            if context.performance_priority == Priority.HIGH:
                if profile['performance'] == 'High':
                    score += 3
                elif profile['performance'] in ['Medium-High', 'Medium']:
                    score += 1
                else:
                    score -= 1
            
            # Regulatory constraints
            if context.regulatory_constraints == Priority.HIGH:
                if profile['regulatory_friendly']:
                    score += 2
                else:
                    score -= 3
            
            # Computational budget
            if context.computational_budget == Priority.LOW:
                if profile['computational_cost'] == 'Low':
                    score += 2
                elif profile['computational_cost'] == 'High':
                    score -= 2
            
            # Stability requirement
            if context.model_stability_requirement == Priority.HIGH:
                if profile['stability'] == 'High':
                    score += 2
                elif profile['stability'] == 'Low':
                    score -= 2
            
            scores[method_id] = score
        
        return scores
    
    def _find_matching_scenarios(self, context: BusinessContext) -> List[str]:
        """
        Find scenario rubrics that match the given context
        """
        matching_scenarios = []
        
        for scenario_id, rubric in self.scenario_rubrics.items():
            conditions = rubric['conditions']
            matches = True
            
            # Check rejection rate
            if 'rejection_rate' in conditions:
                min_rate, max_rate = conditions['rejection_rate']
                if not (min_rate <= context.rejection_rate <= max_rate):
                    matches = False
            
            # Check categorical conditions
            categorical_checks = [
                ('interpretability_priority', context.interpretability_priority),
                ('performance_priority', context.performance_priority),
                ('regulatory_constraints', context.regulatory_constraints),
                ('computational_budget', context.computational_budget),
                ('time_to_deploy', context.time_to_deploy),
                ('model_stability_requirement', context.model_stability_requirement),
                ('external_data_quality', context.external_data_quality)
            ]
            
            for condition_name, context_value in categorical_checks:
                if condition_name in conditions:
                    if context_value not in conditions[condition_name]:
                        matches = False
                        break
            
            # Check boolean conditions
            if 'external_data_available' in conditions:
                if conditions['external_data_available'] != context.external_data_available:
                    matches = False
            
            # Check sample size
            if 'sample_size' in conditions:
                min_size, max_size = conditions['sample_size']
                if not (min_size <= context.sample_size <= max_size):
                    matches = False
            
            if matches:
                matching_scenarios.append(scenario_id)
        
        return matching_scenarios
    
    def _synthesize_recommendation(self, method_scores: Dict[str, float], 
                                 matching_scenarios: List[str], 
                                 context: BusinessContext) -> MethodRecommendation:
        """
        Synthesize final recommendation from scores and scenarios
        """
        # Get scenario-recommended methods
        scenario_methods = set()
        scenario_rationales = []
        
        for scenario_id in matching_scenarios:
            rubric = self.scenario_rubrics[scenario_id]
            scenario_methods.update(rubric['recommended_methods'])
            scenario_rationales.append(rubric['rationale'])
        
        # Combine scoring and scenario recommendations
        if scenario_methods:
            # Filter scores to only scenario-recommended methods
            filtered_scores = {k: v for k, v in method_scores.items() if k in scenario_methods}
            if filtered_scores:
                method_scores = filtered_scores
        
        # Sort by score
        sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_methods:
            # Fallback recommendation
            primary_method = 'approved_only'
            backup_methods = ['regularized_approved']
            confidence = "Low"
            rationale = "Fallback recommendation due to no clear matches"
        else:
            primary_method = sorted_methods[0][0]
            backup_methods = [method[0] for method in sorted_methods[1:3]]
            
            # Determine confidence based on score gap and scenario matches
            best_score = sorted_methods[0][1]
            if len(sorted_methods) > 1:
                second_best_score = sorted_methods[1][1]
                score_gap = best_score - second_best_score
            else:
                score_gap = best_score
            
            if score_gap >= 3 and matching_scenarios:
                confidence = "High"
            elif score_gap >= 2 or matching_scenarios:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            # Create rationale
            rationale_parts = []
            if scenario_rationales:
                rationale_parts.append("Scenario analysis: " + "; ".join(set(scenario_rationales)))
            
            method_profile = self.method_profiles[primary_method]
            rationale_parts.append(f"Method profile: {method_profile['interpretability']} interpretability, "
                                 f"{method_profile['performance']} performance, "
                                 f"{method_profile['complexity']} complexity")
            
            rationale = ". ".join(rationale_parts)
        
        # Generate considerations
        considerations = self._generate_considerations(primary_method, context)
        
        # Expected performance and complexity
        method_profile = self.method_profiles[primary_method]
        expected_performance = method_profile['performance']
        implementation_complexity = method_profile['complexity']
        
        return MethodRecommendation(
            primary_method=method_profile['name'],
            backup_methods=[self.method_profiles[m]['name'] for m in backup_methods],
            confidence_level=confidence,
            rationale=rationale,
            considerations=considerations,
            expected_performance=expected_performance,
            implementation_complexity=implementation_complexity
        )
    
    def _generate_considerations(self, method_id: str, context: BusinessContext) -> List[str]:
        """
        Generate method-specific considerations and warnings
        """
        considerations = []
        method_profile = self.method_profiles[method_id]
        
        # Rejection rate warnings
        if context.rejection_rate > 0.7:
            considerations.append("High rejection rate (>70%) creates severe sample selection bias")
        elif context.rejection_rate > 0.5:
            considerations.append("Moderate rejection rate requires careful bias correction")
        
        # External data considerations
        if method_profile['external_data_required']:
            if context.external_data_quality == DataQuality.FAIR:
                considerations.append("External data quality is only fair - monitor performance carefully")
            elif context.external_data_quality == DataQuality.POOR:
                considerations.append("Poor external data quality may hurt more than help")
            
            considerations.append("Ensure external data remains available and stable in production")
        
        # Interpretability warnings
        if (context.interpretability_priority == Priority.HIGH and 
            method_profile['interpretability'] in ['Low', 'Low-Medium']):
            considerations.append("Method has limited interpretability despite high business requirement")
        
        # Regulatory considerations
        if (context.regulatory_constraints == Priority.HIGH and 
            not method_profile['regulatory_friendly']):
            considerations.append("Method may face regulatory scrutiny due to artificial label assignment")
        
        # Computational considerations
        if (context.computational_budget == Priority.LOW and 
            method_profile['computational_cost'] == 'High'):
            considerations.append("High computational cost may strain available resources")
        
        # Stability considerations
        if (context.model_stability_requirement == Priority.HIGH and 
            method_profile['stability'] == 'Low'):
            considerations.append("Method may produce unstable coefficients - monitor over time")
        
        # Sample size considerations
        if context.sample_size < 5000:
            considerations.append("Small sample size may limit method effectiveness")
        elif context.sample_size > 100000:
            considerations.append("Large sample size enables more sophisticated methods")
        
        return considerations
    
    def create_decision_matrix(self) -> pd.DataFrame:
        """
        Create a decision matrix showing method suitability for different conditions
        """
        conditions = [
            'Low Rejection (<30%)',
            'Medium Rejection (30-60%)',
            'High Rejection (>60%)',
            'High Interpretability Need',
            'High Performance Need',
            'Strong Regulatory Constraints',
            'Limited Computational Budget',
            'External Data Available',
            'Small Sample Size (<10K)',
            'Large Sample Size (>50K)'
        ]
        
        matrix_data = []
        
        for method_id, profile in self.method_profiles.items():
            row = {
                'Method': profile['name'],
                'Low Rejection (<30%)': self._get_suitability_score(method_id, 'low_rejection'),
                'Medium Rejection (30-60%)': self._get_suitability_score(method_id, 'medium_rejection'),
                'High Rejection (>60%)': self._get_suitability_score(method_id, 'high_rejection'),
                'High Interpretability Need': self._get_interpretability_score(profile),
                'High Performance Need': self._get_performance_score(profile),
                'Strong Regulatory Constraints': self._get_regulatory_score(profile),
                'Limited Computational Budget': self._get_budget_score(profile),
                'External Data Available': self._get_external_data_score(profile),
                'Small Sample Size (<10K)': self._get_small_sample_score(profile),
                'Large Sample Size (>50K)': self._get_large_sample_score(profile)
            }
            matrix_data.append(row)
        
        return pd.DataFrame(matrix_data)
    
    def _get_suitability_score(self, method_id: str, rejection_level: str) -> str:
        """Get suitability score for rejection rate level"""
        profile = self.method_profiles[method_id]
        best_for = profile['best_for_rejection_rate']
        
        if rejection_level == 'low_rejection':
            return "Excellent" if "Low" in best_for else "Good" if "Medium" in best_for else "Poor"
        elif rejection_level == 'medium_rejection':
            return "Excellent" if "Medium" in best_for else "Good"
        else:  # high_rejection
            return "Excellent" if "High" in best_for else "Good" if "Medium" in best_for else "Fair"
    
    def _get_interpretability_score(self, profile: Dict) -> str:
        """Get interpretability suitability score"""
        interp = profile['interpretability']
        if interp == 'High':
            return "Excellent"
        elif interp in ['Medium-High', 'Medium']:
            return "Good"
        else:
            return "Poor"
    
    def _get_performance_score(self, profile: Dict) -> str:
        """Get performance suitability score"""
        perf = profile['performance']
        if perf == 'High':
            return "Excellent"
        elif perf in ['Medium-High', 'Medium']:
            return "Good"
        else:
            return "Poor"
    
    def _get_regulatory_score(self, profile: Dict) -> str:
        """Get regulatory suitability score"""
        return "Excellent" if profile['regulatory_friendly'] else "Poor"
    
    def _get_budget_score(self, profile: Dict) -> str:
        """Get budget suitability score"""
        cost = profile['computational_cost']
        if cost == 'Low':
            return "Excellent"
        elif cost == 'Medium':
            return "Good"
        else:
            return "Poor"
    
    def _get_external_data_score(self, profile: Dict) -> str:
        """Get external data suitability score"""
        if profile['external_data_required']:
            return "Excellent"
        else:
            return "Good"
    
    def _get_small_sample_score(self, profile: Dict) -> str:
        """Get small sample suitability score"""
        complexity = profile['complexity']
        if complexity == 'Low':
            return "Excellent"
        elif complexity == 'Medium':
            return "Good"
        else:
            return "Fair"
    
    def _get_large_sample_score(self, profile: Dict) -> str:
        """Get large sample suitability score"""
        return "Good"  # All methods can benefit from large samples


def create_example_contexts() -> Dict[str, BusinessContext]:
    """
    Create example business contexts for different scenarios
    """
    contexts = {
        'fintech_startup': BusinessContext(
            rejection_rate=0.4,
            sample_size=8000,
            external_data_available=False,
            external_data_quality=DataQuality.FAIR,
            interpretability_priority=Priority.MEDIUM,
            performance_priority=Priority.HIGH,
            regulatory_constraints=Priority.MEDIUM,
            computational_budget=Priority.LOW,
            model_stability_requirement=Priority.MEDIUM,
            time_to_deploy=Priority.HIGH
        ),
        'traditional_bank': BusinessContext(
            rejection_rate=0.6,
            sample_size=50000,
            external_data_available=True,
            external_data_quality=DataQuality.GOOD,
            interpretability_priority=Priority.HIGH,
            performance_priority=Priority.MEDIUM,
            regulatory_constraints=Priority.HIGH,
            computational_budget=Priority.HIGH,
            model_stability_requirement=Priority.HIGH,
            time_to_deploy=Priority.LOW
        ),
        'online_lender': BusinessContext(
            rejection_rate=0.7,
            sample_size=25000,
            external_data_available=True,
            external_data_quality=DataQuality.EXCELLENT,
            interpretability_priority=Priority.LOW,
            performance_priority=Priority.HIGH,
            regulatory_constraints=Priority.MEDIUM,
            computational_budget=Priority.HIGH,
            model_stability_requirement=Priority.MEDIUM,
            time_to_deploy=Priority.MEDIUM
        ),
        'credit_union': BusinessContext(
            rejection_rate=0.3,
            sample_size=5000,
            external_data_available=False,
            external_data_quality=DataQuality.FAIR,
            interpretability_priority=Priority.HIGH,
            performance_priority=Priority.MEDIUM,
            regulatory_constraints=Priority.HIGH,
            computational_budget=Priority.LOW,
            model_stability_requirement=Priority.HIGH,
            time_to_deploy=Priority.MEDIUM
        )
    }
    
    return contexts 