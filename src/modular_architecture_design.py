#!/usr/bin/env python3
"""
Modular Architecture Design for SAC-GRU PCAP Processor
=====================================================

This module defines the high-level architecture with clear separation of concerns,
high cohesion within modules, and low coupling between modules.

Key Design Principles:
- Single Responsibility Principle
- Dependency Inversion Principle  
- Interface Segregation Principle
- Open/Closed Principle

Author: Enhanced by Manus AI with System Design Principles
"""

from abc import ABC, abstractmethod
from typing import Protocol, List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum
import logging

# ============================================================================
# CORE INTERFACES (Low Coupling through Abstractions)
# ============================================================================

class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingResult:
    """Standardized processing result"""
    file_id: str
    status: ProcessingStatus
    traffic_label: str
    confidence_score: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class PacketSequence:
    """Standardized packet sequence data"""
    file_id: str
    sequence_data: List[List[float]]
    metadata: Dict[str, Any]

# ============================================================================
# DATA LAYER INTERFACES (High Cohesion)
# ============================================================================

class IDataIngestionService(Protocol):
    """Interface for data ingestion - Single Responsibility: File Reading"""
    
    def extract_archive(self, archive_path: str) -> List[str]:
        """Extract archive and return list of PCAP file paths"""
        ...
    
    def read_pcap_file(self, file_path: str) -> Iterator[PacketSequence]:
        """Read PCAP file and yield packet sequences"""
        ...
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files"""
        ...

class IDataPartitioningService(Protocol):
    """Interface for data partitioning - Single Responsibility: Data Chunking"""
    
    def partition_large_file(self, file_path: str, chunk_size: int) -> Iterator[str]:
        """Partition large file into manageable chunks"""
        ...
    
    def estimate_memory_requirements(self, file_path: str) -> Dict[str, float]:
        """Estimate memory requirements for file processing"""
        ...

# ============================================================================
# MACHINE LEARNING LAYER INTERFACES (High Cohesion)
# ============================================================================

class IMLModelService(Protocol):
    """Interface for ML model operations - Single Responsibility: Model Management"""
    
    def load_model(self, model_path: str) -> bool:
        """Load trained SAC-GRU model"""
        ...
    
    def train_model(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train new SAC-GRU model"""
        ...
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for inference"""
        ...

class IMLInferenceService(Protocol):
    """Interface for ML inference - Single Responsibility: Prediction"""
    
    def predict(self, packet_sequence: PacketSequence) -> ProcessingResult:
        """Perform Reel vs Non-Reel classification"""
        ...
    
    def batch_predict(self, sequences: List[PacketSequence]) -> List[ProcessingResult]:
        """Perform batch predictions"""
        ...

class IFeatureExtractionService(Protocol):
    """Interface for feature extraction - Single Responsibility: Feature Engineering"""
    
    def extract_features(self, raw_packets: List[Dict]) -> List[List[float]]:
        """Extract features from raw packet data"""
        ...
    
    def normalize_features(self, features: List[List[float]]) -> List[List[float]]:
        """Normalize feature values"""
        ...

# ============================================================================
# STORAGE LAYER INTERFACES (High Cohesion)
# ============================================================================

class IStorageService(Protocol):
    """Interface for storage operations - Single Responsibility: Data Persistence"""
    
    def save_results(self, results: List[ProcessingResult]) -> bool:
        """Save processing results"""
        ...
    
    def load_results(self, file_id: str) -> Optional[ProcessingResult]:
        """Load processing results by file ID"""
        ...
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        ...

class ICacheService(Protocol):
    """Interface for caching - Single Responsibility: Cache Management"""
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value with optional TTL"""
        ...
    
    def invalidate(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        ...

# ============================================================================
# PROCESSING LAYER INTERFACES (High Cohesion)
# ============================================================================

class IProcessingOrchestrator(Protocol):
    """Interface for processing orchestration - Single Responsibility: Workflow Management"""
    
    def process_archive(self, archive_path: str, config: Dict[str, Any]) -> List[ProcessingResult]:
        """Orchestrate complete archive processing"""
        ...
    
    def get_processing_status(self, job_id: str) -> ProcessingStatus:
        """Get status of processing job"""
        ...

class ILoadBalancer(Protocol):
    """Interface for load balancing - Single Responsibility: Resource Management"""
    
    def distribute_work(self, work_items: List[Any]) -> List[List[Any]]:
        """Distribute work across available workers"""
        ...
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        ...

# ============================================================================
# EXPORT LAYER INTERFACES (High Cohesion)
# ============================================================================

class IExportService(Protocol):
    """Interface for export operations - Single Responsibility: Output Generation"""
    
    def export_csv(self, results: List[ProcessingResult], output_path: str) -> bool:
        """Export results to CSV format"""
        ...
    
    def export_json(self, results: List[ProcessingResult], output_path: str) -> bool:
        """Export results to JSON format"""
        ...
    
    def export_mobile_model(self, model_path: str, output_path: str) -> str:
        """Export model for mobile deployment"""
        ...

# ============================================================================
# NOTIFICATION LAYER INTERFACES (High Cohesion)
# ============================================================================

class INotificationService(Protocol):
    """Interface for notifications - Single Responsibility: Event Communication"""
    
    def notify_processing_started(self, job_id: str) -> None:
        """Notify that processing has started"""
        ...
    
    def notify_processing_completed(self, job_id: str, results: List[ProcessingResult]) -> None:
        """Notify that processing has completed"""
        ...
    
    def notify_error(self, job_id: str, error: str) -> None:
        """Notify about processing errors"""
        ...

# ============================================================================
# CONFIGURATION LAYER INTERFACES (High Cohesion)
# ============================================================================

class IConfigurationService(Protocol):
    """Interface for configuration management - Single Responsibility: Config Management"""
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        ...
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        ...
    
    def load_from_file(self, config_path: str) -> bool:
        """Load configuration from file"""
        ...

# ============================================================================
# DEPENDENCY INJECTION CONTAINER (Low Coupling)
# ============================================================================

class ServiceContainer:
    """Dependency injection container for loose coupling"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
    
    def register_service(self, interface_name: str, implementation: Any) -> None:
        """Register a service implementation"""
        self._services[interface_name] = implementation
    
    def register_factory(self, interface_name: str, factory: callable) -> None:
        """Register a factory function for lazy initialization"""
        self._factories[interface_name] = factory
    
    def get_service(self, interface_name: str) -> Any:
        """Get service implementation"""
        if interface_name in self._services:
            return self._services[interface_name]
        elif interface_name in self._factories:
            service = self._factories[interface_name]()
            self._services[interface_name] = service
            return service
        else:
            raise ValueError(f"Service not registered: {interface_name}")
    
    def has_service(self, interface_name: str) -> bool:
        """Check if service is registered"""
        return interface_name in self._services or interface_name in self._factories

# ============================================================================
# MAIN SYSTEM FACADE (API Gateway Pattern)
# ============================================================================

class SACGRUProcessingSystem:
    """
    Main system facade implementing API Gateway pattern
    Single Responsibility: System Coordination
    """
    
    def __init__(self, container: ServiceContainer):
        self._container = container
        self._logger = logging.getLogger(__name__)
    
    def process_pcap_archive(self, archive_path: str, config: Optional[Dict[str, Any]] = None) -> List[ProcessingResult]:
        """
        Main entry point for PCAP archive processing
        Coordinates all services through dependency injection
        """
        try:
            # Get required services (Low Coupling through DI)
            orchestrator = self._container.get_service('processing_orchestrator')
            notification_service = self._container.get_service('notification_service')
            
            # Start processing
            job_id = f"job_{archive_path.split('/')[-1]}_{int(time.time())}"
            notification_service.notify_processing_started(job_id)
            
            # Delegate to orchestrator
            results = orchestrator.process_archive(archive_path, config or {})
            
            # Notify completion
            notification_service.notify_processing_completed(job_id, results)
            
            return results
            
        except Exception as e:
            self._logger.error(f"Processing failed: {e}")
            notification_service.notify_error(job_id, str(e))
            raise
    
    def export_results(self, results: List[ProcessingResult], format_type: str, output_path: str) -> bool:
        """Export processing results in specified format"""
        export_service = self._container.get_service('export_service')
        
        if format_type.lower() == 'csv':
            return export_service.export_csv(results, output_path)
        elif format_type.lower() == 'json':
            return export_service.export_json(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'ml_model_ready': self._container.get_service('ml_model_service').is_model_ready(),
            'storage_stats': self._container.get_service('storage_service').get_storage_stats(),
            'worker_stats': self._container.get_service('load_balancer').get_worker_stats(),
            'services_registered': len(self._container._services) + len(self._container._factories)
        }

# ============================================================================
# ARCHITECTURE BENEFITS SUMMARY
# ============================================================================

"""
HIGH COHESION ACHIEVED:
- Each interface has a single, well-defined responsibility
- Data ingestion only handles file operations
- ML services only handle model operations
- Storage services only handle persistence
- Export services only handle output generation

LOW COUPLING ACHIEVED:
- All dependencies are through interfaces (Protocol)
- Dependency injection eliminates hard dependencies
- Services can be swapped without affecting others
- Configuration is externalized
- Communication through events/notifications

SYSTEM DESIGN PRINCIPLES APPLIED:
1. Microservices: Each service is independent and focused
2. API Gateway: Single entry point through SACGRUProcessingSystem
3. Caching: ICacheService for performance optimization
4. Load Balancing: ILoadBalancer for resource management
5. Data Partitioning: IDataPartitioningService for large files
6. Notification System: INotificationService for event communication

SCALABILITY & MAINTAINABILITY:
- Easy to add new services
- Easy to modify existing implementations
- Easy to test individual components
- Easy to deploy services independently
- Clear separation of concerns
"""

