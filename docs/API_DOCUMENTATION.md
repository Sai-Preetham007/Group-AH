# Medical Knowledge RAG Chatbot - API Documentation

## Overview
The Medical Knowledge RAG Chatbot provides a RESTful API for querying medical information using Retrieval-Augmented Generation (RAG) technology.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently no authentication is required. Future versions will include API key authentication.

## Endpoints

### Health Check
- **GET** `/health`
- **Description**: Check if the service is running
- **Response**: 
  ```json
  {
    "status": "healthy",
    "message": "Service is running"
  }
  ```

### Root Endpoint
- **GET** `/`
- **Description**: Main health check with version info
- **Response**:
  ```json
  {
    "status": "healthy",
    "message": "Medical Knowledge RAG Chatbot is running",
    "version": "1.0.0"
  }
  ```

### System Information
- **GET** `/info`
- **Description**: Get system information and available endpoints
- **Response**:
  ```json
  {
    "name": "Medical Knowledge RAG Chatbot",
    "version": "1.0.0",
    "description": "A healthcare-focused Retrieval-Augmented Generation system",
    "endpoints": {
      "health": "/health",
      "docs": "/docs",
      "redoc": "/redoc"
    }
  }
  ```

## Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Error Responses
All endpoints return appropriate HTTP status codes:
- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Endpoint not found
- `500 Internal Server Error`: Server error

## Rate Limiting
Currently no rate limiting is implemented. Future versions will include rate limiting for production use.

## CORS
Cross-Origin Resource Sharing (CORS) is enabled for all origins to support web applications.
