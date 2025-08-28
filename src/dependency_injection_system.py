#!/usr/bin/env python3
"""
Dependency Injection System and Main Application
===============================================

This module implements the dependency injection container and main application
that demonstrates low coupling through interface-based dependencies.

Key Features:
- Dependency Injection Container for loose coupling
- Factory pattern for service creation
- Configuration-driven service setup
- Easy testing through mock implementations
- Clean separation of concerns

Author: Enhanced by Manus AI with System Design Principles
"""

import os
import sys
import logging
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import asdict

# Import architecture and implementations
from modular_architecture_design import (
    ServiceContainer, SACGRUProcessingSystem, ProcessingResult
)
from core_modules_implementation import (
    DataIngestionService, DataPartitioningService, MLModelService,
    MLInferenceService, FeatureExtractionService, ProcessingOrchestrator,
    SimpleStorageService, SimpleLoadBalancer, SimpleNotificationService,
    SimpleConfigurationService, SimpleCacheService
)

# ============================================================================
# SERVICE FACTORY (Factory Pattern for Service Creation)
# ============================================================================

class ServiceFactory:
    """
    Factory for creating service instances with proper dependencies
    Implements Factory Pattern for loose coupling
    """
    
    @staticmethod
    def create_configuration_service(config_file: Optional[str] = None) -> SimpleConfigurationService:
        """Create configuration service"""
        config_service = SimpleConfigurationService()
        
        if config_file and os.path.exists(config_file):
            config_service.load_from_file(config_file)
        
        return config_service
    
    @staticmethod
    def create_cache_service() -> SimpleCacheService:
        """Create cache service"""
        return SimpleCacheService()
    
    @staticmethod
    def create_data_ingestion_service(config_service) -> DataIngestionService:
        """Create data ingestion service with dependencies"""
        return DataIngestionService(config_service)
    
    @staticmethod
    def create_data_partitioning_service(config_service) -> DataPartitioningService:
        """Create data partitioning service with dependencies"""
        return DataPartitioningService(config_service)
    
    @staticmethod
    def create_ml_model_service(config_service, cache_service) -> MLModelService:
        """Create ML model service with dependencies"""
        return MLModelService(config_service, cache_service)
    
    @staticmethod
    def create_feature_extraction_service(config_service) -> FeatureExtractionService:
        """Create feature extraction service with dependencies"""
        return FeatureExtractionService(config_service)
    
    @staticmethod
    def create_ml_inference_service(model_service, feature_service, config_service) -> MLInferenceService:
        """Create ML inference service with dependencies"""
        return MLInferenceService(model_service, feature_service, config_service)
    
    @staticmethod
    def create_storage_service(storage_dir: str = "results") -> SimpleStorageService:
        """Create storage service"""
        return SimpleStorageService(storage_dir)
    
    @staticmethod
    def create_load_balancer_service(max_workers: int = 4) -> SimpleLoadBalancer:
        """Create load balancer service"""
        return SimpleLoadBalancer(max_workers)
    
    @staticmethod
    def create_notification_service() -> SimpleNotificationService:
        """Create notification service"""
        return SimpleNotificationService()
    
    @staticmethod
    def create_processing_orchestrator(container: ServiceContainer) -> ProcessingOrchestrator:
        """Create processing orchestrator with all dependencies"""
        return ProcessingOrchestrator(
            data_ingestion=container.get_service('data_ingestion'),
            data_partitioning=container.get_service('data_partitioning'),
            ml_inference=container.get_service('ml_inference'),
            load_balancer=container.get_service('load_balancer'),
            storage=container.get_service('storage'),
            notification=container.get_service('notification'),
            config=container.get_service('configuration')
        )

# ============================================================================
# DEPENDENCY INJECTION SETUP (Configuration-Driven)
# ============================================================================

class DIContainer:
    """
    Enhanced Dependency Injection Container
    Provides configuration-driven service setup with lazy initialization
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self._container = ServiceContainer()
        self._logger = logging.getLogger(__name__)
        self._setup_services(config_file)
    
    def _setup_services(self, config_file: Optional[str]) -> None:
        """Setup all services with proper dependency injection"""
        self._logger.info("Setting up dependency injection container...")
        
        # Core services (no dependencies)
        self._container.register_factory(
            'configuration',
            lambda: ServiceFactory.create_configuration_service(config_file)
        )
        
        self._container.register_factory(
            'cache',
            lambda: ServiceFactory.create_cache_service()
        )
        
        self._container.register_factory(
            'notification',
            lambda: ServiceFactory.create_notification_service()
        )
        
        # Data services (depend on configuration)
        self._container.register_factory(
            'data_ingestion',
            lambda: ServiceFactory.create_data_ingestion_service(
                self._container.get_service('configuration')
            )
        )
        
        self._container.register_factory(
            'data_partitioning',
            lambda: ServiceFactory.create_data_partitioning_service(
                self._container.get_service('configuration')
            )
        )
        
        # ML services (depend on configuration and cache)
        self._container.register_factory(
            'ml_model',
            lambda: ServiceFactory.create_ml_model_service(
                self._container.get_service('configuration'),
                self._container.get_service('cache')
            )
        )
        
        self._container.register_factory(
            'feature_extraction',
            lambda: ServiceFactory.create_feature_extraction_service(
                self._container.get_service('configuration')
            )
        )
        
        self._container.register_factory(
            'ml_inference',
            lambda: ServiceFactory.create_ml_inference_service(
                self._container.get_service('ml_model'),
                self._container.get_service('feature_extraction'),
                self._container.get_service('configuration')
            )
        )
        
        # Infrastructure services
        self._container.register_factory(
            'storage',
            lambda: ServiceFactory.create_storage_service(
                self._container.get_service('configuration').get_config('storage_dir', 'results')
            )
        )
        
        self._container.register_factory(
            'load_balancer',
            lambda: ServiceFactory.create_load_balancer_service(
                self._container.get_service('configuration').get_config('max_workers', 4)
            )
        )
        
        # Orchestrator (depends on all other services)
        self._container.register_factory(
            'processing_orchestrator',
            lambda: ServiceFactory.create_processing_orchestrator(self._container)
        )
        
        self._logger.info("Dependency injection container setup complete")
    
    def get_container(self) -> ServiceContainer:
        """Get the service container"""
        return self._container
    
    def validate_dependencies(self) -> bool:
        """Validate that all dependencies can be resolved"""
        try:
            # Try to get all services to validate dependencies
            services_to_validate = [
                'configuration', 'cache', 'notification', 'data_ingestion',
                'data_partitioning', 'ml_model', 'feature_extraction',
                'ml_inference', 'storage', 'load_balancer', 'processing_orchestrator'
            ]
            
            for service_name in services_to_validate:
                service = self._container.get_service(service_name)
                if service is None:
                    self._logger.error(f"Failed to create service: {service_name}")
                    return False
            
            self._logger.info("All dependencies validated successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Dependency validation failed: {e}")
            return False

# ============================================================================
# MAIN APPLICATION (API Gateway Pattern)
# ============================================================================

class SACGRUApplication:
    """
    Main application class implementing API Gateway pattern
    Single entry point for all system operations
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self._setup_logging()
        self._logger = logging.getLogger(__name__)
        
        # Initialize dependency injection
        self._di_container = DIContainer(config_file)
        
        # Validate dependencies
        if not self._di_container.validate_dependencies():
            raise RuntimeError("Failed to setup dependencies")
        
        # Create main processing system
        self._processing_system = SACGRUProcessingSystem(
            self._di_container.get_container()
        )
        
        self._logger.info("SAC-GRU Application initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sac_gru_application.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def process_pcap_archive(self, archive_path: str, 
                           output_format: str = 'json',
                           output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point for PCAP archive processing
        
        Args:
            archive_path: Path to PCAP archive
            output_format: Output format ('json' or 'csv')
            output_path: Optional output path
            
        Returns:
            Processing summary
        """
        self._logger.info(f"Starting PCAP archive processing: {archive_path}")
        start_time = time.time()
        
        try:
            # Process the archive
            results = self._processing_system.process_pcap_archive(archive_path)
            
            # Export results if output path specified
            if output_path:
                success = self._processing_system.export_results(
                    results, output_format, output_path
                )
                if not success:
                    self._logger.warning(f"Failed to export results to {output_path}")
            
            # Calculate summary statistics
            processing_time = time.time() - start_time
            summary = self._calculate_processing_summary(results, processing_time)
            
            self._logger.info(f"Processing completed in {processing_time:.2f} seconds")
            return summary
            
        except Exception as e:
            self._logger.error(f"Processing failed: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_status': self._processing_system.get_system_status(),
            'dependency_container': {
                'services_registered': len(self._di_container.get_container()._services) + 
                                     len(self._di_container.get_container()._factories)
            },
            'application_info': {
                'version': '1.0.0',
                'architecture': 'SAC-GRU with System Design Principles',
                'design_patterns': ['Dependency Injection', 'Factory', 'API Gateway', 'Observer']
            }
        }
    
    def train_new_model(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a new SAC-GRU model"""
        self._logger.info("Starting model training...")
        
        ml_model_service = self._di_container.get_container().get_service('ml_model')
        training_stats = ml_model_service.train_model(training_config)
        
        self._logger.info("Model training completed")
        return training_stats
    
    def _calculate_processing_summary(self, results: List[ProcessingResult], 
                                    processing_time: float) -> Dict[str, Any]:
        """Calculate processing summary statistics"""
        if not results:
            return {
                'total_sequences': 0,
                'processing_time': processing_time,
                'classification_summary': {},
                'average_confidence': 0.0,
                'success_rate': 0.0
            }
        
        # Classification summary
        classification_counts = {}
        total_confidence = 0.0
        successful_predictions = 0
        
        for result in results:
            label = result.traffic_label
            classification_counts[label] = classification_counts.get(label, 0) + 1
            total_confidence += result.confidence_score
            
            if result.status.value == 'completed':
                successful_predictions += 1
        
        return {
            'total_sequences': len(results),
            'processing_time': processing_time,
            'processing_rate': len(results) / processing_time if processing_time > 0 else 0,
            'classification_summary': classification_counts,
            'average_confidence': total_confidence / len(results),
            'success_rate': successful_predictions / len(results),
            'architecture_benefits': {
                'high_cohesion': 'Each module has single responsibility',
                'low_coupling': 'Dependencies through interfaces only',
                'maintainability': 'Easy to modify individual components',
                'testability': 'Each module can be tested in isolation',
                'scalability': 'Components can be scaled independently'
            }
        }

# ============================================================================
# CONFIGURATION EXAMPLES
# ============================================================================

def create_sample_config_file(config_path: str = "sac_gru_config.json") -> None:
    """Create a sample configuration file"""
    sample_config = {
        "sequence_length": 20,
        "feature_dim": 5,
        "hidden_units": 64,
        "batch_size": 32,
        "max_workers": 4,
        "storage_dir": "results",
        "model_path": "sac_gru_model",
        "training": {
            "num_episodes": 10000,
            "batch_size": 64,
            "update_frequency": 4,
            "confidence_threshold": 0.7
        },
        "processing": {
            "chunk_size_mb": 512,
            "progress_interval": 50000,
            "max_memory_usage_percent": 80
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print(f"Sample configuration created: {config_path}")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC-GRU PCAP Processor with System Design Principles')
    parser.add_argument('archive_path', help='Path to PCAP archive file')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    parser.add_argument('--train', action='store_true', help='Train new model before processing')
    parser.add_argument('--status', action='store_true', help='Show system status only')
    parser.add_argument('--create-config', help='Create sample configuration file')
    
    args = parser.parse_args()
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config_file(args.create_config)
        return
    
    try:
        # Initialize application
        app = SACGRUApplication(args.config)
        
        # Show status if requested
        if args.status:
            status = app.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        # Train model if requested
        if args.train:
            training_config = {
                'num_episodes': 5000,
                'batch_size': 32,
                'sequence_length': 20,
                'feature_dim': 5,
                'hidden_units': 64
            }
            training_stats = app.train_new_model(training_config)
            print(f"Model training completed. Accuracy: {training_stats['final_accuracy']:.4f}")
        
        # Process archive
        summary = app.process_pcap_archive(
            args.archive_path,
            args.format,
            args.output
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total sequences processed: {summary['total_sequences']:,}")
        print(f"Processing time: {summary['processing_time']:.2f} seconds")
        print(f"Processing rate: {summary['processing_rate']:.0f} sequences/second")
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Average confidence: {summary['average_confidence']:.3f}")
        
        print("\nClassification Results:")
        for label, count in summary['classification_summary'].items():
            percentage = (count / summary['total_sequences']) * 100
            print(f"  {label}: {count:,} ({percentage:.1f}%)")
        
        print("\nArchitecture Benefits Demonstrated:")
        for benefit, description in summary['architecture_benefits'].items():
            print(f"  {benefit.replace('_', ' ').title()}: {description}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

