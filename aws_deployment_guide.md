# AI QA Agent - AWS Cloud Deployment Guide

## Overview
This guide provides step-by-step instructions to deploy the complete AI QA Agent system on AWS using production-ready infrastructure including EKS, RDS, ElastiCache, CloudWatch, and other AWS services.

## Prerequisites

### AWS Account Setup
- **AWS Account**: Active AWS account with appropriate permissions
- **AWS CLI**: Configured with access keys
- **kubectl**: Kubernetes CLI tool installed
- **eksctl**: Amazon EKS CLI tool installed
- **Docker**: For building and pushing container images
- **Terraform** (optional): For Infrastructure as Code

### Required Permissions
Your AWS user/role needs the following permissions:
- EKS (Amazon Elastic Kubernetes Service)
- EC2 (Elastic Compute Cloud)
- RDS (Relational Database Service)
- ElastiCache (Redis)
- VPC (Virtual Private Cloud)
- IAM (Identity and Access Management)
- CloudWatch (Monitoring and Logging)
- ECR (Elastic Container Registry)
- ALB (Application Load Balancer)

## Step 1: AWS Environment Setup

### 1.1 Configure AWS CLI
```bash
# Configure AWS CLI with your credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region (e.g., us-west-2)

# Verify AWS access
aws sts get-caller-identity
# Should return your AWS account information
```

### 1.2 Set Environment Variables
```bash
# Set deployment variables
export AWS_REGION=us-west-2
export CLUSTER_NAME=ai-qa-agent-cluster
export PROJECT_NAME=ai-qa-agent
export ENVIRONMENT=production

# Verify variables
echo "Region: $AWS_REGION"
echo "Cluster: $CLUSTER_NAME"
echo "Project: $PROJECT_NAME"
```

### 1.3 Install Required Tools
```bash
# Install eksctl (if not already installed)
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install kubectl (if not already installed)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify installations
eksctl version
kubectl version --client
```

## Step 2: Infrastructure Provisioning

### 2.1 Create EKS Cluster
```bash
# Create EKS cluster configuration
cat > cluster.yaml << 'EOF'
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: ai-qa-agent-cluster
  region: us-west-2
  version: "1.28"

nodeGroups:
  - name: ai-qa-agent-workers
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 100
    ssh:
      enableSsm: true
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enable: ["audit", "authenticator", "controllerManager"]
EOF

# Create the EKS cluster
eksctl create cluster -f cluster.yaml

# This will take 15-20 minutes. Verify cluster creation
kubectl get nodes
```

### 2.2 Create RDS PostgreSQL Database
```bash
# Create DB subnet group
aws rds create-db-subnet-group \
    --db-subnet-group-name ai-qa-agent-db-subnet-group \
    --db-subnet-group-description "Subnet group for AI QA Agent database" \
    --subnet-ids $(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$(aws eks describe-cluster --name $CLUSTER_NAME --query 'cluster.resourcesVpcConfig.vpcId' --output text)" \
        --query 'Subnets[?AvailabilityZone!=`us-west-2d`].SubnetId' \
        --output text)

# Create security group for RDS
VPC_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query 'cluster.resourcesVpcConfig.vpcId' --output text)
DB_SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name ai-qa-agent-db-sg \
    --description "Security group for AI QA Agent database" \
    --vpc-id $VPC_ID \
    --query 'GroupId' --output text)

# Allow PostgreSQL access from EKS nodes
NODE_SECURITY_GROUP_ID=$(aws eks describe-cluster --name $CLUSTER_NAME --query 'cluster.resourcesVpcConfig.securityGroupIds[0]' --output text)
aws ec2 authorize-security-group-ingress \
    --group-id $DB_SECURITY_GROUP_ID \
    --protocol tcp \
    --port 5432 \
    --source-group $NODE_SECURITY_GROUP_ID

# Create RDS PostgreSQL instance
aws rds create-db-instance \
    --db-instance-identifier ai-qa-agent-db \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --engine-version 15.3 \
    --master-username postgres \
    --master-user-password 'SecurePassword123!' \
    --allocated-storage 100 \
    --storage-type gp3 \
    --db-subnet-group-name ai-qa-agent-db-subnet-group \
    --vpc-security-group-ids $DB_SECURITY_GROUP_ID \
    --backup-retention-period 7 \
    --multi-az \
    --storage-encrypted

# Wait for RDS instance to be available (10-15 minutes)
aws rds wait db-instance-available --db-instance-identifier ai-qa-agent-db

# Get RDS endpoint
RDS_ENDPOINT=$(aws rds describe-db-instances --db-instance-identifier ai-qa-agent-db --query 'DBInstances[0].Endpoint.Address' --output text)
echo "RDS Endpoint: $RDS_ENDPOINT"
```

### 2.3 Create ElastiCache Redis Cluster
```bash
# Create cache subnet group
aws elasticache create-cache-subnet-group \
    --cache-subnet-group-name ai-qa-agent-cache-subnet-group \
    --cache-subnet-group-description "Subnet group for AI QA Agent cache" \
    --subnet-ids $(aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[?AvailabilityZone!=`us-west-2d`].SubnetId' \
        --output text)

# Create security group for ElastiCache
CACHE_SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name ai-qa-agent-cache-sg \
    --description "Security group for AI QA Agent cache" \
    --vpc-id $VPC_ID \
    --query 'GroupId' --output text)

# Allow Redis access from EKS nodes
aws ec2 authorize-security-group-ingress \
    --group-id $CACHE_SECURITY_GROUP_ID \
    --protocol tcp \
    --port 6379 \
    --source-group $NODE_SECURITY_GROUP_ID

# Create ElastiCache Redis cluster
aws elasticache create-replication-group \
    --replication-group-id ai-qa-agent-redis \
    --description "Redis cluster for AI QA Agent" \
    --num-cache-clusters 2 \
    --cache-node-type cache.t3.medium \
    --engine redis \
    --engine-version 7.0 \
    --port 6379 \
    --cache-subnet-group-name ai-qa-agent-cache-subnet-group \
    --security-group-ids $CACHE_SECURITY_GROUP_ID \
    --at-rest-encryption-enabled \
    --transit-encryption-enabled

# Wait for Redis cluster to be available (10-15 minutes)
aws elasticache wait replication-group-available --replication-group-id ai-qa-agent-redis

# Get Redis endpoint
REDIS_ENDPOINT=$(aws elasticache describe-replication-groups --replication-group-id ai-qa-agent-redis --query 'ReplicationGroups[0].RedisEndpoint.Address' --output text)
echo "Redis Endpoint: $REDIS_ENDPOINT"
```

## Step 3: Container Registry Setup

### 3.1 Create ECR Repositories
```bash
# Create ECR repositories for each service
aws ecr create-repository --repository-name ai-qa-agent/main
aws ecr create-repository --repository-name ai-qa-agent/orchestrator
aws ecr create-repository --repository-name ai-qa-agent/specialists
aws ecr create-repository --repository-name ai-qa-agent/conversation
aws ecr create-repository --repository-name ai-qa-agent/learning

# Get ECR login token
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com
```

### 3.2 Build and Push Container Images
```bash
# Set ECR repository URLs
export ECR_REGISTRY=$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com
export MAIN_IMAGE=$ECR_REGISTRY/ai-qa-agent/main:latest
export ORCHESTRATOR_IMAGE=$ECR_REGISTRY/ai-qa-agent/orchestrator:latest
export SPECIALISTS_IMAGE=$ECR_REGISTRY/ai-qa-agent/specialists:latest

# Build main application image
docker build -f docker/agent-system/Dockerfile.agent-base -t $MAIN_IMAGE .
docker push $MAIN_IMAGE

# Build orchestrator image
docker build -f docker/agent-system/Dockerfile.orchestrator -t $ORCHESTRATOR_IMAGE .
docker push $ORCHESTRATOR_IMAGE

# Build specialists image
docker build -f docker/agent-system/Dockerfile.specialists -t $SPECIALISTS_IMAGE .
docker push $SPECIALISTS_IMAGE

# Verify images are pushed
aws ecr describe-images --repository-name ai-qa-agent/main
```

## Step 4: Kubernetes Configuration

### 4.1 Create Namespace and Secrets
```bash
# Create namespace
kubectl create namespace ai-qa-agent

# Create secrets for database and API keys
kubectl create secret generic db-credentials \
    --from-literal=host=$RDS_ENDPOINT \
    --from-literal=username=postgres \
    --from-literal=password='SecurePassword123!' \
    --from-literal=database=qaagent \
    --namespace=ai-qa-agent

kubectl create secret generic redis-credentials \
    --from-literal=host=$REDIS_ENDPOINT \
    --from-literal=port=6379 \
    --namespace=ai-qa-agent

kubectl create secret generic api-keys \
    --from-literal=openai-api-key='your_openai_api_key_here' \
    --from-literal=anthropic-api-key='your_anthropic_api_key_here' \
    --namespace=ai-qa-agent

# IMPORTANT: Replace with your actual API keys
echo "⚠️  IMPORTANT: Update the api-keys secret with your actual API keys"
```

### 4.2 Deploy Application Components
```bash
# Create ConfigMap for application configuration
cat > k8s-configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-qa-agent-config
  namespace: ai-qa-agent
data:
  ENVIRONMENT: "production"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  PROMETHEUS_ENABLED: "true"
  JAEGER_ENABLED: "true"
EOF

kubectl apply -f k8s-configmap.yaml

# Deploy main application
cat > k8s-main-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-qa-agent-main
  namespace: ai-qa-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-qa-agent-main
  template:
    metadata:
      labels:
        app: ai-qa-agent-main
    spec:
      containers:
      - name: main
        image: $MAIN_IMAGE
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USERNAME):\$(DB_PASSWORD)@\$(DB_HOST):5432/\$(DB_DATABASE)"
        - name: REDIS_URL
          value: "redis://\$(REDIS_HOST):\$(REDIS_PORT)/0"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: anthropic-api-key
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: database
        - name: REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: host
        - name: REDIS_PORT
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: port
        envFrom:
        - configMapRef:
            name: ai-qa-agent-config
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-qa-agent-main-service
  namespace: ai-qa-agent
spec:
  selector:
    app: ai-qa-agent-main
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
EOF

kubectl apply -f k8s-main-deployment.yaml
```

### 4.3 Deploy Agent Services
```bash
# Deploy agent orchestrator
cat > k8s-agent-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-qa-agent-orchestrator
  namespace: ai-qa-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ai-qa-agent-orchestrator
  template:
    metadata:
      labels:
        app: ai-qa-agent-orchestrator
    spec:
      containers:
      - name: orchestrator
        image: $ORCHESTRATOR_IMAGE
        ports:
        - containerPort: 8000
        env:
        - name: AGENT_MODE
          value: "orchestrator"
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USERNAME):\$(DB_PASSWORD)@\$(DB_HOST):5432/\$(DB_DATABASE)"
        - name: REDIS_URL
          value: "redis://\$(REDIS_HOST):\$(REDIS_PORT)/0"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: database
        - name: REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: host
        - name: REDIS_PORT
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: port
        envFrom:
        - configMapRef:
            name: ai-qa-agent-config
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-qa-agent-specialists
  namespace: ai-qa-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-qa-agent-specialists
  template:
    metadata:
      labels:
        app: ai-qa-agent-specialists
    spec:
      containers:
      - name: specialists
        image: $SPECIALISTS_IMAGE
        env:
        - name: AGENT_MODE
          value: "specialist"
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USERNAME):\$(DB_PASSWORD)@\$(DB_HOST):5432/\$(DB_DATABASE)"
        - name: REDIS_URL
          value: "redis://\$(REDIS_HOST):\$(REDIS_PORT)/0"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: database
        - name: REDIS_HOST
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: host
        - name: REDIS_PORT
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: port
        envFrom:
        - configMapRef:
            name: ai-qa-agent-config
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1500m
            memory: 3Gi
EOF

kubectl apply -f k8s-agent-deployment.yaml
```

## Step 5: Load Balancer and Ingress Setup

### 5.1 Install AWS Load Balancer Controller
```bash
# Create IAM role for AWS Load Balancer Controller
eksctl create iamserviceaccount \
    --cluster=$CLUSTER_NAME \
    --namespace=kube-system \
    --name=aws-load-balancer-controller \
    --role-name=AmazonEKSLoadBalancerControllerRole \
    --attach-policy-arn=arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess \
    --approve

# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName=$CLUSTER_NAME \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller
```

### 5.2 Create Application Load Balancer
```bash
# Create ALB Ingress
cat > k8s-ingress.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-qa-agent-ingress
  namespace: ai-qa-agent
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health/
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: '30'
    alb.ingress.kubernetes.io/healthy-threshold-count: '2'
    alb.ingress.kubernetes.io/unhealthy-threshold-count: '3'
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-qa-agent-main-service
            port:
              number: 80
EOF

kubectl apply -f k8s-ingress.yaml

# Wait for ALB to be provisioned and get URL
sleep 60
ALB_URL=$(kubectl get ingress ai-qa-agent-ingress -n ai-qa-agent -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
echo "Application Load Balancer URL: http://$ALB_URL"
```

## Step 6: Monitoring and Logging Setup

### 6.1 Deploy Prometheus and Grafana
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --set prometheus.prometheusSpec.retention=30d \
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
    --set grafana.adminPassword=admin123

# Wait for Prometheus to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus -n monitoring --timeout=300s
```

### 6.2 Configure CloudWatch Integration
```bash
# Create CloudWatch agent configuration
cat > cloudwatch-config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: cwagentconfig
  namespace: amazon-cloudwatch
data:
  cwagentconfig.json: |
    {
      "logs": {
        "metrics_collected": {
          "kubernetes": {
            "cluster_name": "$CLUSTER_NAME",
            "metrics_collection_interval": 60
          }
        },
        "force_flush_interval": 5
      }
    }
EOF

# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cloudwatch-namespace.yaml
kubectl apply -f cloudwatch-config.yaml
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-daemonset.yaml
```

## Step 7: Auto-Scaling Configuration

### 7.1 Install Cluster Autoscaler
```bash
# Create IAM role for Cluster Autoscaler
eksctl create iamserviceaccount \
    --cluster=$CLUSTER_NAME \
    --namespace=kube-system \
    --name=cluster-autoscaler \
    --attach-policy-arn=arn:aws:iam::aws:policy/AutoScalingFullAccess \
    --approve

# Deploy Cluster Autoscaler
cat > cluster-autoscaler.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/$CLUSTER_NAME
        env:
        - name: AWS_REGION
          value: $AWS_REGION
EOF

kubectl apply -f cluster-autoscaler.yaml
```

### 7.2 Configure Horizontal Pod Autoscaler
```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Create HPA for main application
cat > k8s-hpa.yaml << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-qa-agent-main-hpa
  namespace: ai-qa-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-qa-agent-main
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

kubectl apply -f k8s-hpa.yaml
```

## Step 8: Security Configuration

### 8.1 Network Policies
```bash
# Create network policies for security
cat > k8s-network-policies.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-qa-agent-network-policy
  namespace: ai-qa-agent
spec:
  podSelector:
    matchLabels:
      app: ai-qa-agent-main
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS
    - protocol: TCP
      port: 53    # DNS
    - protocol: UDP
      port: 53    # DNS
EOF

kubectl apply -f k8s-network-policies.yaml
```

### 8.2 Pod Security Standards
```bash
# Create Pod Security Policy
cat > k8s-pod-security.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ai-qa-agent
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
EOF

kubectl apply -f k8s-pod-security.yaml
```

## Step 9: Database Migration and Initialization

### 9.1 Initialize Database
```bash
# Create a one-time job to initialize the database
cat > k8s-db-init-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: db-init-job
  namespace: ai-qa-agent
spec:
  template:
    spec:
      containers:
      - name: db-init
        image: $MAIN_IMAGE
        command: ["/bin/sh"]
        args:
        - -c
        - |
          python -c "
          from src.core.database import engine, Base
          Base.metadata.create_all(bind=engine)
          print('Database initialized successfully')
          "
        env:
        - name: DATABASE_URL
          value: "postgresql://\$(DB_USERNAME):\$(DB_PASSWORD)@\$(DB_HOST):5432/\$(DB_DATABASE)"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        - name: DB_DATABASE
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: database
        envFrom:
        - configMapRef:
            name: ai-qa-agent-config
      restartPolicy: Never
  backoffLimit: 4
EOF

kubectl apply -f k8s-db-init-job.yaml

# Wait for job to complete
kubectl wait --for=condition=complete job/db-init-job -n ai-qa-agent --timeout=300s
```

## Step 10: Verification and Testing

### 10.1 Check Deployment Status
```bash
# Check all pods are running
kubectl get pods -n ai-qa-agent
kubectl get pods -n monitoring

# Check services
kubectl get services -n ai-qa-agent

# Check ingress
kubectl get ingress -n ai-qa-agent

# Check logs
kubectl logs -f deployment/ai-qa-agent-main -n ai-qa-agent
```

### 10.2 Test Application Health
```bash
# Test health endpoints
curl http://$ALB_URL/health/
curl http://$ALB_URL/health/detailed
curl http://$ALB_URL/api/v1/analysis/status
curl http://$ALB_URL/api/v1/agent/status

# All should return successful responses
```

### 10.3 Test Web Interface
```bash
# Test web interface
echo "🌐 AI QA Agent URLs:"
echo "Main Application: http://$ALB_URL/"
echo "Agent Chat: http://$ALB_URL/agent-chat"
echo "Analytics Dashboard: http://$ALB_URL/analytics"
echo "Demo Platform: http://$ALB_URL/demos"
echo "API Documentation: http://$ALB_URL/docs"
```

## Step 11: Production Optimizations

### 11.1 Enable SSL/TLS
```bash
# Request SSL certificate from ACM
aws acm request-certificate \
    --domain-name "ai-qa-agent.yourdomain.com" \
    --validation-method DNS \
    --region $AWS_REGION

# Update ingress with SSL
# (You'll need to update the domain and certificate ARN)
```

### 11.2 Configure Backup Strategy
```bash
# Enable automated RDS backups (already configured)
# Configure automated snapshots for persistent volumes
cat > k8s-backup-policy.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: backup-policy
  namespace: ai-qa-agent
data:
  backup-schedule: "0 2 * * *"  # Daily at 2 AM
  retention-days: "30"
EOF

kubectl apply -f k8s-backup-policy.yaml
```

## Step 12: Cost Optimization

### 12.1 Configure Spot Instances
```bash
# Update node group to use spot instances for non-critical workloads
eksctl create nodegroup \
    --cluster=$CLUSTER_NAME \
    --name=spot-workers \
    --instance-types=m5.large,m5.xlarge,m4.large \
    --spot \
    --nodes=2 \
    --nodes-min=1 \
    --nodes-max=10
```

### 12.2 Set Resource Quotas
```bash
# Create resource quotas to control costs
cat > k8s-resource-quota.yaml << EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ai-qa-agent-quota
  namespace: ai-qa-agent
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    pods: "50"
EOF

kubectl apply -f k8s-resource-quota.yaml
```

## Troubleshooting

### Common Issues and Solutions

#### 1. EKS Cluster Creation Fails
```bash
# Check IAM permissions
aws iam get-user
aws iam list-attached-user-policies --user-name <your-username>

# Check service limits
aws service-quotas get-service-quota --service-code eks --quota-code L-1194D53C
```

#### 2. RDS Connection Issues
```bash
# Check security groups
aws ec2 describe-security-groups --group-ids $DB_SECURITY_GROUP_ID

# Test connection from EKS
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- psql -h $RDS_ENDPOINT -U postgres
```

#### 3. Container Image Pull Errors
```bash
# Check ECR permissions
aws ecr describe-repositories

# Ensure images are pushed
aws ecr describe-images --repository-name ai-qa-agent/main
```

#### 4. LoadBalancer Not Accessible
```bash
# Check ALB status
kubectl describe ingress ai-qa-agent-ingress -n ai-qa-agent

# Check target groups
aws elbv2 describe-target-groups
```

## Monitoring and Maintenance

### CloudWatch Dashboards
```bash
# Key metrics to monitor:
# - EKS cluster health
# - RDS performance
# - ElastiCache performance
# - Application response times
# - Error rates
# - Cost metrics
```

### Automated Scaling
```bash
# Monitor auto-scaling events
kubectl get hpa -n ai-qa-agent
kubectl describe hpa ai-qa-agent-main-hpa -n ai-qa-agent

# Check cluster autoscaler logs
kubectl logs -f deployment/cluster-autoscaler -n kube-system
```

## Cost Estimates

Estimated monthly costs for production deployment:
- **EKS Cluster**: $75/month (control plane)
- **EC2 Instances**: $300-800/month (3-10 m5.xlarge instances)
- **RDS PostgreSQL**: $150-300/month (db.t3.medium with Multi-AZ)
- **ElastiCache Redis**: $100-200/month (cache.t3.medium with replication)
- **ALB**: $25/month
- **Data Transfer**: $50-200/month (varies by usage)
- **CloudWatch**: $50-100/month
- **Total**: ~$750-1,750/month

## Security Best Practices

1. **Secrets Management**: Use AWS Secrets Manager or Kubernetes secrets
2. **Network Security**: Implement VPC security groups and network policies
3. **Access Control**: Use IAM roles and RBAC
4. **Encryption**: Enable encryption at rest and in transit
5. **Monitoring**: Set up CloudTrail and comprehensive logging
6. **Updates**: Keep EKS, node groups, and container images updated

---

**🎉 Congratulations! Your AI QA Agent system is now deployed on AWS with enterprise-grade infrastructure, monitoring, and auto-scaling capabilities.**