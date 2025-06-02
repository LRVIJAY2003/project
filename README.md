
  # NEW: Add this No Response SLO
  - krmName: sphere-no-response-usc1
    displayName: "No Response: 99.5% of requests receive response"
    mode: STATS
    goal: 0.995  # 99.5% must receive a response
    errorBudgetThreshold: 0.80
    rollingPeriod: "604800s"  # 7 days
    serviceLevelIndicator:
      requestBased:
        goodTotalRatio:
          # Good: All requests that receive ANY HTTP response (including errors)
          goodServiceFilter: >-
            metric.type="workload.googleapis.com/http.server.request.duration" 
            resource.type="k8s_container" 
            metric.labels.app_id="2852" 
            metric.labels.component_id="coml-5913"
            metric.labels.namespace="app-2852-default" 
            metric.labels.http_route="monitoring.regex.full_match(\"/.*/\")" 
            resource.labels.location="us-central1"
          # Total: All requests attempted (including timeouts)  
          totalServiceFilter: >-
            metric.type="workload.googleapis.com/http.server.request.count" 
            resource.type="k8s_container" 
            metric.labels.app_id="2852" 
            metric.labels.component_id="coml-5913"
            metric.labels.namespace="app-2852-default" 
            metric.labels.http_route="monitoring.regex.full_match(\"/.*/\")" 
            resource.labels.location="us-central1"





- name: "sphere-no-response-slo-burn"
    displayName: "No Response: Timeout rate exceeding threshold"
    combiner: "OR"
    conditions:
      - displayName: "High timeout rate detected"
        conditionThreshold:
          filter: >-
            metric.type="monitoring.googleapis.com/uptime_check/check_passed" 
            resource.type="k8s_container" 
            metric.labels.app_id="2852" 
            metric.labels.component_id="coml-5913"
            metric.labels.namespace="app-2852-default" 
            metric.labels.http_route="monitoring.regex.full_match(\"/.*/\")" 
            resource.labels.location="us-central1"
          aggregations:
            - alignmentPeriod: "300s"  # 5 minutes
              perSeriesAligner: "ALIGN_RATE"
              crossSeriesReducer: "REDUCE_SUM"
          comparison: "COMPARISON_LT"  # Less than (for response rate)
          thresholdValue: 0.995  # Alert when response rate drops below 99.5%
          duration: "300s"  # Alert after 5 minutes
    alertStrategy:
      notificationRateLimit:
        period: "300s"  # Limit notifications to every 5 minutes
      autoClose: "1800s"  # Auto-close after 30 minutes
    notificationChannels:
      - "spheresupport@cmegroup.com"
      - "refdatasre@cmegroup.com"
    documentation:
      content: >-
        CRITICAL: High number of requests are timing out without any response from the Sphere application. 
        This indicates potential complete service outages, network issues, or backend failures.
        
        Immediate actions:
        1. Check service health and pod status
        2. Verify network connectivity and load balancer health
        3. Review recent deployments or configuration changes
        4. Check resource utilization (CPU, memory, disk)
        5. Examine application logs for errors or performance issues
        
        This alert fires when less than 99.5% of requests receive any response within the timeout period.
      mimeType: "text/markdown"