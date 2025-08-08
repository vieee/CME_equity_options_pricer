# CME Equity Options Pricer - Docker Build and Run Script (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        Write-Success "Docker is running"
        return $true
    }
    catch {
        Write-Error "Docker is not running. Please start Docker and try again."
        return $false
    }
}

# Function to build the Docker image
function Build-Image {
    Write-Status "Building CME Equity Options Pricer Docker image..."
    
    try {
        docker build -t cme-equity-options-pricer:latest .
        Write-Success "Docker image built successfully"
        return $true
    }
    catch {
        Write-Error "Failed to build Docker image"
        return $false
    }
}

# Function to run the container
function Start-Container {
    Write-Status "Starting CME Equity Options Pricer container..."
    
    # Stop any existing container
    $existingContainer = docker ps -q -f name=cme-equity-options-pricer
    if ($existingContainer) {
        Write-Warning "Stopping existing container..."
        docker stop cme-equity-options-pricer
        docker rm cme-equity-options-pricer
    }
    
    # Run the new container
    try {
        docker run -d `
            --name cme-equity-options-pricer `
            -p 8501:8501 `
            --restart unless-stopped `
            cme-equity-options-pricer:latest
        
        Write-Success "Container started successfully"
        Write-Status "Application available at: http://localhost:8501"
        return $true
    }
    catch {
        Write-Error "Failed to start container"
        return $false
    }
}

# Function to run with docker-compose
function Start-Compose {
    Write-Status "Starting CME Equity Options Pricer with Docker Compose..."
    
    try {
        docker-compose up -d
        Write-Success "Services started successfully with Docker Compose"
        Write-Status "Application available at: http://localhost:8501"
        return $true
    }
    catch {
        Write-Error "Failed to start services with Docker Compose"
        return $false
    }
}

# Function to show logs
function Show-Logs {
    Write-Status "Showing container logs..."
    docker logs -f cme-equity-options-pricer
}

# Function to stop services
function Stop-Services {
    Write-Status "Stopping CME Equity Options Pricer services..."
    
    $existingContainer = docker ps -q -f name=cme-equity-options-pricer
    if ($existingContainer) {
        docker stop cme-equity-options-pricer
        docker rm cme-equity-options-pricer
        Write-Success "Container stopped"
    }
    
    if (Get-Command docker-compose -ErrorAction SilentlyContinue) {
        docker-compose down
        Write-Success "Docker Compose services stopped"
    }
}

# Function to clean up Docker resources
function Invoke-Cleanup {
    Write-Status "Cleaning up Docker resources..."
    
    # Remove stopped containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    Write-Success "Cleanup completed"
}

# Main script logic
switch ($Command.ToLower()) {
    "build" {
        if (Test-Docker) {
            Build-Image
        }
    }
    "run" {
        if (Test-Docker) {
            if (Build-Image) {
                Start-Container
            }
        }
    }
    "compose" {
        if (Test-Docker) {
            Start-Compose
        }
    }
    "logs" {
        Show-Logs
    }
    "stop" {
        Stop-Services
    }
    "cleanup" {
        Invoke-Cleanup
    }
    "restart" {
        if (Test-Docker) {
            Stop-Services
            if (Build-Image) {
                Start-Container
            }
        }
    }
    default {
        Write-Host "CME Equity Options Pricer - Docker Management Script" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Usage: .\docker-run.ps1 [command]" -ForegroundColor White
        Write-Host ""
        Write-Host "Commands:" -ForegroundColor White
        Write-Host "  build     - Build the Docker image" -ForegroundColor Gray
        Write-Host "  run       - Build and run the container" -ForegroundColor Gray
        Write-Host "  compose   - Start services with Docker Compose" -ForegroundColor Gray
        Write-Host "  logs      - Show container logs" -ForegroundColor Gray
        Write-Host "  stop      - Stop all services" -ForegroundColor Gray
        Write-Host "  cleanup   - Clean up Docker resources" -ForegroundColor Gray
        Write-Host "  restart   - Stop, build, and restart container" -ForegroundColor Gray
        Write-Host ""
        Write-Host "Examples:" -ForegroundColor White
        Write-Host "  .\docker-run.ps1 run      # Build and start the application" -ForegroundColor Gray
        Write-Host "  .\docker-run.ps1 compose  # Start with Docker Compose" -ForegroundColor Gray
        Write-Host "  .\docker-run.ps1 logs     # View application logs" -ForegroundColor Gray
        Write-Host "  .\docker-run.ps1 stop     # Stop the application" -ForegroundColor Gray
    }
}
