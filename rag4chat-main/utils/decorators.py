"""
Decorator utility module
"""
import functools
import logging
import streamlit as st
from typing import Callable, Any

logger = logging.getLogger(__name__)

def error_handler(show_error: bool = True) -> Callable:
    """
    Unified error handling decorator
    
    @param {bool} show_error - Whether to display error information in UI
    @returns {Callable} - Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{func.__name__} execution failed: {str(e)}")
                if show_error:
                    st.error(f"Operation failed: {str(e)}")
                raise
        return wrapper
    return decorator



def log_execution(func: Callable) -> Callable:
    """
    Decorator to log function execution
    
    @param {Callable} func - Function to be decorated
    @returns {Callable} - Decorator function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger.info(f"Starting execution of {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} executed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} execution failed: {str(e)}")
            raise
    return wrapper 