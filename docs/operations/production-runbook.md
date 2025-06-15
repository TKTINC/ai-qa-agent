# Production Operations Runbook: AI QA Agent System

## Overview

This runbook provides comprehensive operational procedures for maintaining the AI QA Agent System in production environments. It covers monitoring, troubleshooting, maintenance, and emergency response procedures for ensuring 99.9% uptime and optimal performance.

## System Architecture Overview

### Core Components
- **Agent Orchestrator**: Central coordination service (Port 8000)
- **Specialist Agents**: Domain expert services (Port 8001)
- **Conversation Manager**: Real-time communication service (Port 8080, 8081)
- **Learning Engine**: Continuous learning service (Port 8002)
- **Redis**: State management and caching (Port 6379)
- **PostgreSQL**: Persistent data storage (Port 5432)
- **Monitoring Stack**: Prometheus (9090), Grafana (3000), Jaeger (16686)

### Service Dependencies
```
User Requests → Load Balancer → Agent Orchestrator → Specialist Agents
                                      ↓
                           Conversation Manager ↔ Redis ↔ Learning Engine
                                      ↓
                                PostgreSQL
```

## Health Monitoring and Alerting

### Key Health Endpoints

#### Service Health Checks
```bash
# Agent Orchestrator Health
curl -f http://agent-orchestrator:8000/health/live || echo "CRITICAL: Orchestrator down"

# Specialist Agents Health  
curl -f http://specialist-agents:8001/health || echo "WARNING: Specialists unavailable"

# Conversation Manager Health
curl -f http://conversation-manager:8081/health || echo "CRITICAL: Conversations down"

# Learning Engine Health
curl -f http://learning-engine:8002/health || echo "WARNING: Learning disabled"
```

#### Database Health Checks
```bash
# PostgreSQL Health
pg_isready -h postgres-agent -p 5432 -U agent_user || echo "CRITICAL: Database down"

# Redis Health
redis-cli -h redis-agent-state ping || echo "CRITICAL: Redis down"
```

#### Intelligence Metrics Health
```bash
# Check reasoning quality
curl -s http://agent-orchestrator:8000/metrics | grep "agent_reasoning_quality_score" || echo "WARNING: No reasoning metrics"

# Check learning velocity
curl -s http://learning-engine:8002/metrics | grep "agent_learning_velocity" || echo "WARNING: No learning metrics"
```

### Critical Alert Conditions

#### Immediate Response Required (P0)
- **System Down**: Any core service completely unavailable for >1 minute
- **Database Failure**: PostgreSQL or Redis complete failure
- **Authentication Failure**: Unable to authenticate users
- **Data Loss**: Any indication of data corruption or loss
- **Security Breach**: Unauthorized access or security incident

#### Urgent Response Required (P1)
- **Performance Degradation**: >10 second response times or >20% error rate
- **Agent Intelligence Failure**: Reasoning quality score <0.5 for >10 minutes
- **Learning System Failure**: No learning events for >30 minutes
- **High Memory Usage**: >90% memory utilization for >15 minutes
- **Disk Space Critical**: <10% disk space remaining

#### Monitor and Plan (P2)
- **Elevated Response Times**: 5-10 second response times
- **Reasoning Quality Decline**: Quality score 0.5-0.7 for >1 hour
- **User Satisfaction Drop**: Satisfaction score <3.0 for >2 hours
- **Resource Usage High**: >80% CPU or memory for >1 hour

## Troubleshooting Procedures

### Agent Orchestrator Issues

#### Symptom: High Response Times
```bash
# Check system resources
kubectl top pods -n qa-agent | grep orchestrator

# Check reasoning queue length
curl -s http://agent-orchestrator:8000/metrics | grep "reasoning_queue_length"

# Check database connections
curl -s http://agent-orchestrator:8000/health/detailed | jq '.database'

# Actions:
1. Scale orchestrator pods: kubectl scale deployment agent-orchestrator --replicas=5 -n qa-agent
2. Check database performance: pg_stat_activity queries
3. Review recent reasoning complexity trends
4. Consider temporary load shedding if critical
```

#### Symptom: Agent Reasoning Quality Degradation
```bash
# Check recent reasoning metrics
curl -s http://agent-orchestrator:8000/metrics | grep "agent_reasoning_quality_score"

# Check learning system health
curl -s http://learning-engine:8002/health/detailed

# Check collaboration success rates
curl -s http://agent-orchestrator:8000/metrics | grep "collaboration_effectiveness"

# Actions:
1. Review recent learning events and feedback quality
2. Check for data quality issues in training pipeline
3. Verify specialist agent availability and performance
4. Consider rolling back recent model updates
5. Manually trigger learning system refresh
```

### Specialist Agents Issues

#### Symptom: Specialist Unavailable
```bash
# Check specialist pod status
kubectl get pods -n qa-agent | grep specialist

# Check specialist registration
curl -s http://agent-orchestrator:8000/api/v1/agent/specialists

# Check tool execution metrics
curl -s http://specialist-agents:8001/metrics | grep "tool_execution"

# Actions:
1. Restart failed specialist pods: kubectl delete pod -l app=specialist-agents -n qa-agent
2. Verify tool dependencies and external service connectivity
3. Check specialist configuration and environment variables
4. Scale specialist pool if capacity issues: kubectl scale deployment specialist-agents --replicas=8
```

#### Symptom: Tool Execution Failures
```bash
# Check tool success rates
curl -s http://specialist-agents:8001/metrics | grep "tool_success_rate"

# Check recent tool errors
kubectl logs -l app=specialist-agents -n qa-agent --tail=100 | grep ERROR

# Check external dependencies
curl -f https://api.external-tool.com/health || echo "External dependency down"

# Actions:
1. Identify failing tools and check configurations
2. Verify external service availability and API keys
3. Review tool timeout settings and resource limits
4. Implement tool-specific circuit breakers
5. Consider disabling problematic tools temporarily
```

### Conversation Manager Issues

#### Symptom: WebSocket Connection Failures
```bash
# Check active WebSocket connections
curl -s http://conversation-manager:8081/metrics | grep "websocket_connections"

# Check connection error rates
curl -s http://conversation-manager:8081/metrics | grep "websocket_errors"

# Check Redis connectivity
redis-cli -h redis-agent-state ping

# Actions:
1. Check load balancer WebSocket configuration
2. Verify Redis connectivity and performance
3. Review WebSocket timeout and keepalive settings
4. Scale conversation manager replicas if needed
5. Check for network connectivity issues
```

#### Symptom: Session State Loss
```bash
# Check Redis memory usage and eviction
redis-cli -h redis-agent-state info memory

# Check session persistence metrics
curl -s http://conversation-manager:8081/metrics | grep "session_persistence"

# Check for Redis failover events
kubectl logs -l app=redis-agent-state -n qa-agent --tail=50

# Actions:
1. Verify Redis persistence configuration
2. Check Redis memory limits and eviction policies
3. Review session cleanup and timeout settings
4. Consider increasing Redis memory allocation
5. Implement session backup and recovery procedures
```

### Learning Engine Issues

#### Symptom: Learning Stagnation
```bash
# Check learning event rates
curl -s http://learning-engine:8002/metrics | grep "learning_events_total"

# Check learning velocity trends
curl -s http://learning-engine:8002/metrics | grep "learning_velocity"

# Check feedback processing
curl -s http://learning-engine:8002/health/detailed | jq '.feedback_processing'

# Actions:
1. Verify feedback data quality and volume
2. Check learning algorithm parameters and thresholds
3. Review recent user interaction patterns
4. Restart learning engine to clear cached models
5. Trigger manual learning cycle with known good data
```

#### Symptom: Memory Leak in Learning Engine
```bash
# Check memory usage trends
kubectl top pods -n qa-agent | grep learning-engine

# Check for memory allocation patterns
curl -s http://learning-engine:8002/metrics | grep "memory_usage"

# Check garbage collection metrics
kubectl exec -it deployment/learning-engine -n qa-agent -- python -c "import gc; print(gc.get_stats())"

# Actions:
1. Restart learning engine pod to clear memory
2. Review recent learning data size and complexity
3. Implement memory limits and garbage collection tuning
4. Consider batch processing for large learning events
5. Scale learning engine horizontally if needed
```

### Database Issues

#### Symptom: PostgreSQL Performance Degradation
```bash
# Check active connections
psql -h postgres-agent -U agent_user -c "SELECT count(*) FROM pg_stat_activity;"

# Check slow queries
psql -h postgres-agent -U agent_user -c "SELECT query, state, query_start FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# Check database locks
psql -h postgres-agent -U agent_user -c "SELECT * FROM pg_locks WHERE NOT granted;"

# Actions:
1. Identify and kill long-running queries if safe
2. Check for missing indexes on frequently queried columns
3. Update table statistics: ANALYZE;
4. Consider increasing connection pool size
5. Review query patterns and optimize problematic queries
```

#### Symptom: Redis Memory Pressure
```bash
# Check memory usage and fragmentation
redis-cli -h redis-agent-state info memory

# Check eviction statistics
redis-cli -h redis-agent-state info stats | grep evicted

# Check key expiration patterns
redis-cli -h redis-agent-state info keyspace

# Actions:
1. Increase Redis memory allocation if needed
2. Review key expiration policies and TTL settings
3. Implement key cleanup procedures for old sessions
4. Consider Redis cluster for horizontal scaling
5. Optimize data structures and serialization
```

## Maintenance Procedures

### Routine Maintenance Tasks

#### Daily Operations
```bash
#!/bin/bash
# Daily maintenance script

# Check system health
./scripts/health_check.sh

# Verify backup completion
./scripts/verify_backups.sh

# Clean up old logs
find /var/log/qa-agent -name "*.log" -mtime +7 -delete

# Update metrics dashboard
curl -X POST http://grafana:3000/api/dashboards/home/refresh
```

#### Weekly Maintenance
```bash
#!/bin/bash
# Weekly maintenance script

# Database maintenance
psql -h postgres-agent -U agent_user -c "VACUUM ANALYZE;"

# Redis maintenance
redis-cli -h redis-agent-state BGREWRITEAOF

# Learning model optimization
curl -X POST http://learning-engine:8002/api/v1/maintenance/optimize

# Performance report generation
./scripts/generate_weekly_report.sh
```

#### Monthly Maintenance
```bash
#!/bin/bash
# Monthly maintenance script

# Full system backup
./scripts/full_backup.sh

# Security certificate renewal
./scripts/renew_certificates.sh

# Performance baseline update
./scripts/update_performance_baselines.sh

# Capacity planning analysis
./scripts/capacity_planning_analysis.sh
```

### Deployment Procedures

#### Rolling Updates
```bash
#!/bin/bash
# Rolling update procedure

# 1. Update staging environment
kubectl apply -f k8s/staging/ -n qa-agent-staging

# 2. Run integration tests
./tests/integration/run_all_tests.sh staging

# 3. Deploy to production with rolling update
kubectl set image deployment/agent-orchestrator orchestrator=qa-agent/orchestrator:v2.1.0 -n qa-agent

# 4. Monitor deployment progress
kubectl rollout status deployment/agent-orchestrator -n qa-agent

# 5. Verify health after deployment
./scripts/post_deployment_health_check.sh

# 6. Update monitoring dashboards
./scripts/update_dashboards_for_new_version.sh
```

#### Rollback Procedures
```bash
#!/bin/bash
# Emergency rollback procedure

# 1. Identify last known good version
kubectl rollout history deployment/agent-orchestrator -n qa-agent

# 2. Rollback to previous version
kubectl rollout undo deployment/agent-orchestrator -n qa-agent

# 3. Verify rollback success
kubectl rollout status deployment/agent-orchestrator -n qa-agent

# 4. Validate system health
./scripts/health_check.sh

# 5. Notify stakeholders
./scripts/send_rollback_notification.sh
```

### Backup and Recovery

#### Backup Procedures
```bash
#!/bin/bash
# Comprehensive backup script

# PostgreSQL backup
pg_dump -h postgres-agent -U agent_user -Fc agent_db > backups/agent_db_$(date +%Y%m%d_%H%M%S).backup

# Redis backup
redis-cli -h redis-agent-state --rdb backups/redis_$(date +%Y%m%d_%H%M%S).rdb

# Learning model backup
kubectl exec deployment/learning-engine -n qa-agent -- tar -czf - /app/models | gzip > backups/models_$(date +%Y%m%d_%H%M%S).tar.gz

# Configuration backup
kubectl get configmaps,secrets -n qa-agent -o yaml > backups/config_$(date +%Y%m%d_%H%M%S).yaml
```

#### Recovery Procedures
```bash
#!/bin/bash
# Emergency recovery script

# 1. Stop all services
kubectl scale deployment --all --replicas=0 -n qa-agent

# 2. Restore PostgreSQL
pg_restore -h postgres-agent -U agent_user -d agent_db backups/latest_db.backup

# 3. Restore Redis
redis-cli -h redis-agent-state --rdb backups/latest_redis.rdb

# 4. Restore learning models
kubectl exec deployment/learning-engine -n qa-agent -- tar -xzf - -C /app < backups/latest_models.tar.gz

# 5. Restart services
kubectl scale deployment --all --replicas=1 -n qa-agent

# 6. Verify recovery
./scripts/recovery_validation.sh
```

## Performance Optimization

### Performance Monitoring
```bash
# Monitor key performance indicators
curl -s http://prometheus:9090/api/v1/query?query=agent_response_time_seconds | jq '.data.result'

# Check resource utilization
kubectl top nodes
kubectl top pods -n qa-agent

# Monitor intelligence metrics
curl -s http://agent-orchestrator:8000/metrics | grep "reasoning_quality_score"
```

### Optimization Procedures
```bash
# Database query optimization
psql -h postgres-agent -U agent_user -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"

# Redis performance tuning
redis-cli -h redis-agent-state CONFIG SET maxmemory-policy allkeys-lru

# Agent reasoning optimization
curl -X POST http://learning-engine:8002/api/v1/optimize/reasoning_cache

# Connection pool optimization
kubectl patch deployment agent-orchestrator -n qa-agent -p '{"spec":{"template":{"spec":{"containers":[{"name":"orchestrator","env":[{"name":"DB_POOL_SIZE","value":"20"}]}]}}}}'
```

## Emergency Response Procedures

### Service Outage Response

#### Complete System Outage
1. **Immediate Assessment** (0-5 minutes)
   - Check infrastructure status (cloud provider, network)
   - Verify database and Redis connectivity
   - Check load balancer and ingress controller

2. **Initial Response** (5-15 minutes)
   - Restart failed services using kubectl
   - Check recent deployments and rollback if necessary
   - Activate emergency communication channels

3. **Investigation and Resolution** (15-60 minutes)
   - Analyze logs and metrics for root cause
   - Implement fixes or workarounds
   - Monitor system recovery and stability

4. **Post-Incident** (1-24 hours)
   - Conduct post-mortem analysis
   - Update runbooks and procedures
   - Implement preventive measures

#### Data Corruption Response
1. **Immediate Isolation**
   - Stop all write operations to affected databases
   - Preserve current state for forensic analysis
   - Activate backup restoration procedures

2. **Assessment and Recovery**
   - Determine scope and extent of corruption
   - Restore from latest known good backup
   - Validate data integrity after restoration

3. **Prevention**
   - Implement additional data validation checks
   - Enhance backup and monitoring procedures
   - Review access controls and security measures

## Contact Information and Escalation

### On-Call Rotation
- **Primary**: DevOps Engineer (24/7)
- **Secondary**: AI Platform Engineer (24/7)
- **Escalation**: Engineering Manager (business hours)
- **Executive**: CTO (critical incidents only)

### Emergency Contacts
- **PagerDuty**: +1-XXX-XXX-XXXX
- **Slack**: #qa-agent-alerts
- **Email**: alerts@company.com
- **Status Page**: status.qa-agent.com

### Vendor Support
- **Cloud Provider**: 24/7 enterprise support
- **Monitoring**: Grafana Enterprise support
- **Database**: PostgreSQL professional support
- **AI Platform**: OpenAI API enterprise support

---

This runbook should be reviewed and updated monthly to ensure procedures remain current with system evolution and operational experience. All team members should be familiar with these procedures and participate in regular incident response drills.
