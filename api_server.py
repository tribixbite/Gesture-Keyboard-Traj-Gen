#!/usr/bin/env python
"""
Production API server for gesture trajectory generation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import numpy as np
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Import generators
from optimized_swype_generator import OptimizedSwypeGenerator
from improved_swype_generator import ImprovedSwypeGenerator
from unified_swype_api import UnifiedSwypeAPI, TrajectoryConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Gesture Trajectory Generation API",
    description="Generate synthetic swype gesture trajectories",
    version="1.0.0"
)

# Global generators
generators = {}
api = None

# Request/Response models
class GenerationRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)
    method: str = Field(default="optimized", pattern="^(enhanced|improved|optimized|rnn|jerk-min)$")
    style: Optional[str] = Field(default="natural", pattern="^(precise|natural|fast|sloppy)$")
    sampling_rate: Optional[int] = Field(default=100, ge=10, le=1000)
    user_speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0)
    precision: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    
class BatchGenerationRequest(BaseModel):
    words: List[str] = Field(..., min_items=1, max_items=1000)
    method: str = Field(default="optimized")
    style: Optional[str] = Field(default="natural")
    sampling_rate: Optional[int] = Field(default=100)
    
class GenerationResponse(BaseModel):
    word: str
    trajectory: List[List[float]]
    metrics: Dict[str, Any]
    generation_time: float
    
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    generators_loaded: List[str]
    total_requests: int

# Metrics tracking
metrics = {
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'total_generation_time': 0.0,
    'words_generated': []
}

def convert_to_python_types(obj):
    """Convert numpy types to Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    return obj

@app.on_event("startup")
async def startup_event():
    """Initialize generators on startup"""
    global generators, api
    
    logger.info("Initializing trajectory generators...")
    
    try:
        # Initialize generators
        generators['optimized'] = OptimizedSwypeGenerator()
        generators['improved'] = ImprovedSwypeGenerator()
        
        # Initialize unified API
        api = UnifiedSwypeAPI()
        
        logger.info("✅ All generators initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize generators: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        generators_loaded=list(generators.keys()),
        total_requests=metrics['total_requests']
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_trajectory(request: GenerationRequest):
    """Generate a single trajectory"""
    metrics['total_requests'] += 1
    
    try:
        start_time = time.time()
        
        # Create config for unified API
        config = TrajectoryConfig(
            word=request.word,
            method=request.method,
            style=request.style,
            sampling_rate=request.sampling_rate,
            user_speed=request.user_speed,
            precision=request.precision
        )
        
        # Generate trajectory
        result = api.generate(config)
        
        generation_time = time.time() - start_time
        
        # Track metrics
        metrics['successful_requests'] += 1
        metrics['total_generation_time'] += generation_time
        metrics['words_generated'].append(request.word)
        
        # Log request
        logger.info(f"Generated trajectory for '{request.word}' using {request.method} in {generation_time:.3f}s")
        
        # Convert numpy arrays to lists
        response_data = convert_to_python_types({
            'word': request.word,
            'trajectory': result['trajectory'].tolist(),
            'metrics': result.get('metrics', {}),
            'generation_time': generation_time
        })
        
        return GenerationResponse(**response_data)
        
    except Exception as e:
        metrics['failed_requests'] += 1
        logger.error(f"Failed to generate trajectory for '{request.word}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_batch")
async def generate_batch(request: BatchGenerationRequest, background_tasks: BackgroundTasks):
    """Generate multiple trajectories"""
    metrics['total_requests'] += 1
    
    try:
        start_time = time.time()
        results = []
        
        for word in request.words:
            config = TrajectoryConfig(
                word=word,
                method=request.method,
                style=request.style,
                sampling_rate=request.sampling_rate
            )
            
            result = api.generate(config)
            
            results.append({
                'word': word,
                'trajectory': result['trajectory'].tolist(),
                'metrics': result.get('metrics', {})
            })
        
        generation_time = time.time() - start_time
        
        # Track metrics
        metrics['successful_requests'] += 1
        metrics['total_generation_time'] += generation_time
        metrics['words_generated'].extend(request.words)
        
        logger.info(f"Generated {len(results)} trajectories in {generation_time:.3f}s")
        
        # Save batch results asynchronously
        background_tasks.add_task(save_batch_results, results)
        
        return JSONResponse(content={
            'status': 'success',
            'count': len(results),
            'generation_time': generation_time,
            'results': convert_to_python_types(results)
        })
        
    except Exception as e:
        metrics['failed_requests'] += 1
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get API metrics"""
    avg_time = metrics['total_generation_time'] / max(metrics['successful_requests'], 1)
    
    return {
        'total_requests': metrics['total_requests'],
        'successful_requests': metrics['successful_requests'],
        'failed_requests': metrics['failed_requests'],
        'average_generation_time': avg_time,
        'unique_words': len(set(metrics['words_generated'])),
        'total_words_generated': len(metrics['words_generated'])
    }

@app.get("/methods")
async def get_available_methods():
    """Get available generation methods"""
    return {
        'methods': ['enhanced', 'improved', 'optimized', 'rnn', 'jerk-min'],
        'styles': ['precise', 'natural', 'fast', 'sloppy'],
        'default_method': 'optimized',
        'default_style': 'natural'
    }

def save_batch_results(results: List[Dict]):
    """Save batch results to file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_traces/batch_{timestamp}.json"
        
        Path("synthetic_traces").mkdir(exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved batch results to {filename}")
    except Exception as e:
        logger.error(f"Failed to save batch results: {e}")

if __name__ == "__main__":
    # Run server
    port = 8080
    logger.info(f"Starting API server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )