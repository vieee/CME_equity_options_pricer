#!/bin/bash

# CME Equity Options Pricer - Docker Build and Run Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to build the Docker image
build_image() {
    print_status "Building CME Equity Options Pricer Docker image..."
    
    if docker build -t cme-equity-options-pricer:latest .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run the container
run_container() {
    print_status "Starting CME Equity Options Pricer container..."
    
    # Stop any existing container
    if docker ps -q -f name=cme-equity-options-pricer | grep -q .; then
        print_warning "Stopping existing container..."
        docker stop cme-equity-options-pricer
        docker rm cme-equity-options-pricer
    fi
    
    # Run the new container
    if docker run -d \
        --name cme-equity-options-pricer \
        -p 8501:8501 \
        --restart unless-stopped \
        cme-equity-options-pricer:latest; then
        print_success "Container started successfully"
        print_status "Application available at: http://localhost:8501"
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to run with docker-compose
run_compose() {
    print_status "Starting CME Equity Options Pricer with Docker Compose..."
    
    if docker-compose up -d; then
        print_success "Services started successfully with Docker Compose"
        print_status "Application available at: http://localhost:8501"
    else
        print_error "Failed to start services with Docker Compose"
        exit 1
    fi
}

# Function to show logs
show_logs() {
    print_status "Showing container logs..."
    docker logs -f cme-equity-options-pricer
}

# Function to stop services
stop_services() {
    print_status "Stopping CME Equity Options Pricer services..."
    
    if docker ps -q -f name=cme-equity-options-pricer | grep -q .; then
        docker stop cme-equity-options-pricer
        docker rm cme-equity-options-pricer
        print_success "Container stopped"
    fi
    
    if command -v docker-compose &> /dev/null; then
        docker-compose down
        print_success "Docker Compose services stopped"
    fi
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    print_success "Cleanup completed"
}

# Main script logic
case "${1:-}" in
    "build")
        check_docker
        build_image
        ;;
    "run")
        check_docker
        build_image
        run_container
        ;;
    "compose")
        check_docker
        run_compose
        ;;
    "logs")
        show_logs
        ;;
    "stop")
        stop_services
        ;;
    "cleanup")
        cleanup
        ;;
    "restart")
        check_docker
        stop_services
        build_image
        run_container
        ;;
    *)
        echo "CME Equity Options Pricer - Docker Management Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  run       - Build and run the container"
        echo "  compose   - Start services with Docker Compose"
        echo "  logs      - Show container logs"
        echo "  stop      - Stop all services"
        echo "  cleanup   - Clean up Docker resources"
        echo "  restart   - Stop, build, and restart container"
        echo ""
        echo "Examples:"
        echo "  $0 run      # Build and start the application"
        echo "  $0 compose  # Start with Docker Compose"
        echo "  $0 logs     # View application logs"
        echo "  $0 stop     # Stop the application"
        ;;
esac
