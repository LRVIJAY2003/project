```yaml
# Project identification
projectRef: ptj-dv-sphere-0721 # application GCP project ID
# component: coml-5046
projectId: "127578600691"

# Dashboard configuration
dashboard:
  enable: true # disable the dashboard if you prefer to roll your own
  name: Sphere SLO Dashboard

# Service configuration
services:
  component: coml-5913
  displayName: "Sphere coml-5913"
  notify: true
  notificationChannels:
    - name: sphere-notification-channel
      email: spheresupport@cmegroup.com
    - name: refsre-notification-channel
      email: refdatasrc@cmegroup.com
  
  # Service Level Objectives
  objectives:
    # Latency SLO
    - krmName: sphere-latency-usc1
      displayName: "SLO Latency: 99% of endpoints usc1"
      mode: STATS
      goal: 0.99
      #errorBudgetThreshold: 0.80
      rollingPeriod: "2419200s" #28 days
      serviceLevelIndicator:
        requestBased:
          distributionCut:
            distributionFilter: "metric.type=\"workload.googleapis.com/http.server.request.duration\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") metric.labels.service_namespace=\"app-2852-default\" resource.labels.location=\"us-central1\""
            range:
              min: 0
              max: 5
    
    # Availability SLO
    - krmName: sphere-availability-usc1
      displayName: "Availability: 99% instrument endpoint usc1" #good/Total Ratio
      mode: STATS
      goal: 0.99
      calendarPeriod: "WEEK"
      serviceLevelIndicator:
        requestBased:
          goodTotalRatio:
            goodServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.http_response_status_code=\"200\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") resource.labels.location=\"us-central1\""
            totalServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.http_response_status_code=monitoring.regex.full_match(\".*\") metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") resource.labels.location=\"us-central1\""
    
    # Surge Alert Implementation - Structured like your working SLOs
    - krmName: sphere-traffic-surge-usc1
      displayName: "Traffic Surge Alert: 2x baseline in 5min"
      mode: STATS
      goal: 0.99
      rollingPeriod: "2419200s" # 28 days - matching your other SLOs
      serviceLevelIndicator:
        requestBased:
          goodTotalRatio:
            goodServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") resource.labels.location=\"us-central1\""
            totalServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") resource.labels.location=\"us-central1\""

# Alerting policies - Separate from SLOs in this configuration
alertPolicies:
  - displayName: "Sphere Traffic Surge Alert"
    combiner: "OR"
    conditions:
      - displayName: "Sudden traffic increase detected"
        conditionThreshold:
          filter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") resource.labels.location=\"us-central1\""
          aggregations:
            - alignmentPeriod: "60s"
              perSeriesAligner: "ALIGN_RATE"
              crossSeriesReducer: "REDUCE_SUM"
          comparison: "COMPARISON_GT"
          thresholdValue: 2.0
          duration: "300s"
    alertStrategy:
      notificationRateLimit:
        period: "300s"
      autoClose: "1800s"
    notificationChannels:
      - "spheresupport@cmegroup.com"
      - "refdatasrc@cmegroup.com"
    documentation:
      content: "A significant increase in traffic has been detected on the Sphere application. Please investigate potential causes including: unexpected user behavior, scheduled events, external system integration issues, or potential DoS activity."
      mimeType: "text/markdown"
```