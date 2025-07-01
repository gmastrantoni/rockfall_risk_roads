"""
Progress Tracking Utilities

This module provides utilities for tracking progress during long-running operations
in the rockfall risk assessment workflow, particularly for exposure assessment.
"""

import time
import psutil
import logging
import sys
from typing import Optional, Callable, Dict, Any, Union
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress bar implementation
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None, disable=False, **kwargs):
            self.total = total
            self.desc = desc or ""
            self.unit = unit or "it"
            self.disable = disable
            self.n = 0
            self.start_time = time.time()
            self._last_print_time = self.start_time
            
        def update(self, n=1):
            self.n += n
            current_time = time.time()
            # Update every 2 seconds to avoid spam
            if current_time - self._last_print_time >= 2.0 or self.n >= self.total:
                self._print_progress(current_time)
                self._last_print_time = current_time
                
        def _print_progress(self, current_time):
            if self.disable:
                return
            elapsed = current_time - self.start_time
            if self.total:
                percentage = (self.n / self.total) * 100
                rate = self.n / elapsed if elapsed > 0 else 0
                eta = (self.total - self.n) / rate if rate > 0 else 0
                print(f"\r{self.desc}: {percentage:.1f}% ({self.n}/{self.total}) "
                      f"[{elapsed:.1f}s<{eta:.1f}s, {rate:.2f}{self.unit}/s]", end="", flush=True)
            else:
                rate = self.n / elapsed if elapsed > 0 else 0
                print(f"\r{self.desc}: {self.n} [{elapsed:.1f}s, {rate:.2f}{self.unit}/s]", end="", flush=True)
        
        def close(self):
            if not self.disable:
                print()  # New line
                
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            self.close()


class ProgressTracker:
    """
    A comprehensive progress tracking utility for long-running operations.
    
    This class provides progress bars, time estimation, memory monitoring,
    and logging capabilities for tracking the progress of computational tasks.
    """
    
    def __init__(
        self,
        total_items: Optional[int] = None,
        description: str = "Processing",
        enable_progress_bar: bool = True,
        enable_memory_monitoring: bool = True,
        memory_check_interval: float = 5.0,
        memory_warning_threshold: float = 80.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ProgressTracker.
        
        Parameters
        ----------
        total_items : int, optional
            Total number of items to process
        description : str
            Description for the progress display
        enable_progress_bar : bool
            Whether to show progress bar
        enable_memory_monitoring : bool
            Whether to monitor memory usage
        memory_check_interval : float
            Interval in seconds between memory checks
        memory_warning_threshold : float
            Memory usage percentage threshold for warnings
        logger : logging.Logger, optional
            Logger instance for progress messages
        """
        self.total_items = total_items
        self.description = description
        self.enable_progress_bar = enable_progress_bar
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_check_interval = memory_check_interval
        self.memory_warning_threshold = memory_warning_threshold
        self.logger = logger or logging.getLogger(__name__)
        
        # Progress tracking variables
        self.start_time = None
        self.current_item = 0
        self.progress_bar = None
        
        # Memory monitoring
        self.memory_monitor_active = False
        self.memory_monitor_thread = None
        self.max_memory_usage = 0
        self.memory_warnings = 0
        
        # Timing statistics
        self.batch_times = []
        self.estimated_total_time = None
        
    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()
        self.current_item = 0
        self.logger.info(f"Starting {self.description}")
        
        if self.enable_progress_bar:
            self.progress_bar = tqdm(
                total=self.total_items,
                desc=self.description,
                unit="items",
                disable=not self.enable_progress_bar
            )
        
        if self.enable_memory_monitoring:
            self._start_memory_monitoring()
            
        self.logger.info(f"Progress tracking initialized for {self.total_items or 'unknown'} items")
    
    def update(self, increment: int = 1, message: Optional[str] = None):
        """
        Update progress.
        
        Parameters
        ----------
        increment : int
            Number of items completed
        message : str, optional
            Additional message to log
        """
        self.current_item += increment
        
        if self.progress_bar:
            self.progress_bar.update(increment)
            
        # Log periodic updates
        if message or (self.current_item % max(1, (self.total_items or 100) // 10) == 0):
            elapsed_time = time.time() - self.start_time
            rate = self.current_item / elapsed_time if elapsed_time > 0 else 0
            
            if self.total_items:
                progress_percent = (self.current_item / self.total_items) * 100
                eta = (self.total_items - self.current_item) / rate if rate > 0 else 0
                
                log_msg = (f"Progress: {self.current_item}/{self.total_items} "
                          f"({progress_percent:.1f}%) - Rate: {rate:.2f} items/s - "
                          f"ETA: {self._format_time(eta)}")
            else:
                log_msg = (f"Progress: {self.current_item} items - "
                          f"Rate: {rate:.2f} items/s - "
                          f"Elapsed: {self._format_time(elapsed_time)}")
                
            if message:
                log_msg += f" - {message}"
                
            self.logger.info(log_msg)
    
    def finish(self):
        """Finish progress tracking and log summary."""
        if self.progress_bar:
            self.progress_bar.close()
            
        if self.memory_monitor_thread:
            self.memory_monitor_active = False
            self.memory_monitor_thread.join()
        
        total_time = time.time() - self.start_time
        final_rate = self.current_item / total_time if total_time > 0 else 0
        
        self.logger.info(f"Completed {self.description}")
        self.logger.info(f"Total items processed: {self.current_item}")
        self.logger.info(f"Total time: {self._format_time(total_time)}")
        self.logger.info(f"Average rate: {final_rate:.2f} items/s")
        
        if self.enable_memory_monitoring:
            self.logger.info(f"Peak memory usage: {self.max_memory_usage:.1f}%")
            if self.memory_warnings > 0:
                self.logger.warning(f"Memory warnings issued: {self.memory_warnings}")
    
    def _start_memory_monitoring(self):
        """Start memory monitoring in a separate thread."""
        self.memory_monitor_active = True
        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self.memory_monitor_thread.daemon = True
        self.memory_monitor_thread.start()
    
    def _monitor_memory(self):
        """Monitor memory usage in background thread."""
        while self.memory_monitor_active:
            try:
                memory_percent = psutil.virtual_memory().percent
                self.max_memory_usage = max(self.max_memory_usage, memory_percent)
                
                if memory_percent > self.memory_warning_threshold:
                    self.memory_warnings += 1
                    self.logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    
            except Exception as e:
                self.logger.error(f"Error monitoring memory: {e}")
                
            time.sleep(self.memory_check_interval)
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @contextmanager
    def batch_timer(self, batch_size: int):
        """Context manager for timing batches and estimating completion time."""
        batch_start = time.time()
        yield
        batch_time = time.time() - batch_start
        
        self.batch_times.append(batch_time)
        
        # Keep only recent batch times for better estimation
        if len(self.batch_times) > 10:
            self.batch_times = self.batch_times[-10:]
        
        # Estimate total time based on recent batch performance
        if self.total_items and len(self.batch_times) >= 3:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            items_per_batch = batch_size
            batches_remaining = (self.total_items - self.current_item) / items_per_batch
            self.estimated_total_time = batches_remaining * avg_batch_time
            
            self.logger.debug(f"Batch completed in {batch_time:.2f}s. "
                            f"Estimated remaining time: {self._format_time(self.estimated_total_time)}")


class BatchProcessor:
    """
    Utility for processing large datasets in batches with progress tracking.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        progress_tracker: Optional[ProgressTracker] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize BatchProcessor.
        
        Parameters
        ----------
        batch_size : int
            Number of items to process per batch
        progress_tracker : ProgressTracker, optional
            Progress tracker instance
        logger : logging.Logger, optional
            Logger instance
        """
        self.batch_size = batch_size
        self.progress_tracker = progress_tracker
        self.logger = logger or logging.getLogger(__name__)
    
    def process_in_batches(
        self,
        items: list,
        process_func: Callable,
        *args,
        **kwargs
    ) -> list:
        """
        Process items in batches with progress tracking.
        
        Parameters
        ----------
        items : list
            List of items to process
        process_func : callable
            Function to apply to each batch
        *args, **kwargs
            Additional arguments for process_func
            
        Returns
        -------
        list
            Combined results from all batches
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        self.logger.info(f"Processing {len(items)} items in {total_batches} batches of {self.batch_size}")
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            
            self.logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
            
            if self.progress_tracker:
                with self.progress_tracker.batch_timer(len(batch)):
                    batch_result = process_func(batch, *args, **kwargs)
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                    self.progress_tracker.update(len(batch), f"Batch {batch_num}/{total_batches}")
            else:
                batch_result = process_func(batch, *args, **kwargs)
                results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
        
        return results


def create_progress_tracker(
    total_items: Optional[int] = None,
    description: str = "Processing",
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> ProgressTracker:
    """
    Factory function to create a ProgressTracker with configuration.
    
    Parameters
    ----------
    total_items : int, optional
        Total number of items to process
    description : str
        Description for progress tracking
    config : dict, optional
        Configuration dictionary
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    ProgressTracker
        Configured progress tracker instance
    """
    if config is None:
        config = {}
    
    return ProgressTracker(
        total_items=total_items,
        description=description,
        enable_progress_bar=config.get('enable_progress_bar', True),
        enable_memory_monitoring=config.get('enable_memory_monitoring', True),
        memory_check_interval=config.get('memory_check_interval', 5.0),
        memory_warning_threshold=config.get('memory_warning_threshold', 80.0),
        logger=logger
    )
