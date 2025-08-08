# 🐳 Docker Deployment Guide

This guide explains how to deploy the CME Equity Options Pricer using Docker.

## 📋 Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)
- At least 2GB of available RAM
- Internet connection for downloading dependencies

## 🚀 Quick Start

### Option 1: Using the Management Script (Recommended)

**Windows (PowerShell):**
```powershell
# Build and run the application
.\docker-run.ps1 run

# Or start with Docker Compose
.\docker-run.ps1 compose
```

**Linux/macOS (Bash):**
```bash
# Make the script executable
chmod +x docker-run.sh

# Build and run the application
./docker-run.sh run

# Or start with Docker Compose
./docker-run.sh compose
```

### Option 2: Manual Docker Commands

**Build the image:**
```bash
docker build -t cme-equity-options-pricer:latest .
```

**Run the container:**
```bash
docker run -d \
  --name cme-equity-options-pricer \
  -p 8501:8501 \
  --restart unless-stopped \
  cme-equity-options-pricer:latest
```

### Option 3: Docker Compose

**Development environment:**
```bash
docker-compose up -d
```

**Production environment:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🌐 Access the Application

Once running, access the application at:
- **Local:** http://localhost:8501
- **Network:** http://YOUR_IP:8501

## 📊 Management Commands

### Using Management Scripts

**Windows PowerShell:**
```powershell
.\docker-run.ps1 build     # Build the image
.\docker-run.ps1 run       # Build and run
.\docker-run.ps1 logs      # View logs
.\docker-run.ps1 stop      # Stop services
.\docker-run.ps1 restart   # Restart services
.\docker-run.ps1 cleanup   # Clean up resources
```

**Linux/macOS Bash:**
```bash
./docker-run.sh build     # Build the image
./docker-run.sh run       # Build and run
./docker-run.sh logs      # View logs
./docker-run.sh stop      # Stop services
./docker-run.sh restart   # Restart services
./docker-run.sh cleanup   # Clean up resources
```

### Manual Docker Commands

**View logs:**
```bash
docker logs -f cme-equity-options-pricer
```

**Stop the container:**
```bash
docker stop cme-equity-options-pricer
docker rm cme-equity-options-pricer
```

**Stop Docker Compose services:**
```bash
docker-compose down
```

## 🔧 Configuration

### Environment Variables

You can customize the application by setting environment variables:

```bash
docker run -d \
  --name cme-equity-options-pricer \
  -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e ALPHA_VANTAGE_API_KEY=your_api_key \
  -e FRED_API_KEY=your_fred_key \
  cme-equity-options-pricer:latest
```

### Persistent Data

To persist data between container restarts:

```bash
docker run -d \
  --name cme-equity-options-pricer \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  cme-equity-options-pricer:latest
```

## 🏗️ Image Details

### Multi-stage Build
The Dockerfile uses a multi-stage build process:
1. **Builder stage:** Installs dependencies with build tools
2. **Production stage:** Creates a lightweight runtime image

### Security Features
- Runs as non-root user (`streamlit`)
- Minimal base image (Python slim)
- No unnecessary packages in production image
- Health checks enabled

### Performance Optimizations
- Layer caching for faster rebuilds
- Optimized dependency installation
- Resource limits configured
- Efficient file copying

## 🔍 Troubleshooting

### Container Won't Start
1. Check Docker is running: `docker info`
2. Check port availability: `netstat -an | grep 8501`
3. View container logs: `docker logs cme-equity-options-pricer`

### Application Errors
1. Check application logs: `docker logs -f cme-equity-options-pricer`
2. Ensure all required environment variables are set
3. Verify network connectivity for API calls

### Build Failures
1. Clear Docker cache: `docker system prune -a`
2. Ensure sufficient disk space
3. Check internet connectivity for package downloads

### Performance Issues
1. Increase Docker memory allocation (Docker Desktop settings)
2. Adjust resource limits in docker-compose.yml
3. Monitor container resources: `docker stats cme-equity-options-pricer`

## 📈 Production Deployment

### Using Production Compose
The `docker-compose.prod.yml` includes additional services:
- **Nginx:** Reverse proxy with SSL support
- **Redis:** Caching layer for improved performance
- **PostgreSQL:** Optional database for data persistence

```bash
# Start production environment
docker-compose -f docker-compose.prod.yml up -d

# Monitor services
docker-compose -f docker-compose.prod.yml ps
```

### SSL Configuration
For HTTPS in production:
1. Place SSL certificates in `./nginx/ssl/`
2. Update `./nginx/nginx.conf` with SSL configuration
3. Restart services

### Resource Monitoring
Monitor resource usage:
```bash
# View resource consumption
docker stats

# View container health
docker ps
```

## 🧹 Cleanup

### Remove Everything
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi cme-equity-options-pricer:latest

# Clean up all unused resources
docker system prune -a
```

### Selective Cleanup
```bash
# Remove only stopped containers
docker container prune

# Remove only unused images
docker image prune

# Remove only unused volumes
docker volume prune
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review container logs for error messages
3. Ensure all prerequisites are met
4. Verify network connectivity and firewall settings
