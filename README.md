```yaml
# Project identification
projectRef: ptj-dv-sphere-0721 # application GCP project ID
# component: coml-5046
projectId: "127578600691"

# Dashboard configuration
dashboard:
  enable: true # disable the dashboard if you prefer to roll your own
  name: Sphere SLO Dashboard

# Notification configuration (commented out section from original)
#notification:
#  - notificationName: sre-test-email
#  notificationEmail: sriharsha.kankatala@cmegroup.com

# Alerts configuration (commented out section from original)
#alerts:
#  - name: "cuapi-availability-slo-burn"
#    displayName: "Bad/Total-Ratio: 99%"
#  - name: "cuapi-latency-slo-burn"
#    dispalyName: "Latency Burn rate on 99%"

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
  
  # Removed notification channel (preserved as comment from original)
  # notificationName: sre-notification-channel
  # notificationEmail: coppergesservicesupport@cmegroup.com
  
  # Service Level Objectives
  objectives:
    # Latency SLO
    - krmName: sphere-latency-usc1
      displayName: "SLO Latency: 99% of endpoints usc1"
      mode: STATS
      goal: 0.99
      #errorBudgetThreshold: 0.80
      rollingPeriod: "2419200s" #28 days
      #"604800s" # 7 days
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
    
    # Surge Alert Implementation
    - krmName: sphere-traffic-surge-usc1
      displayName: "Traffic Surge Alert: 2x baseline in 5min"
      mode: STATS
      goal: 0.95  # Required for schema validation
      serviceLevelIndicator:
        basicSli:
          availability:
            enabled: true
      # Custom alert policy for surge detection
      alertPolicy:
        displayName: "Sphere Traffic Surge Alert"
        conditions:
          - displayName: "Sudden traffic increase detected"
            conditionThreshold:
              filter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=monitoring.regex.full_match(\"/rest/.*\") resource.labels.location=\"us-central1\""
              aggregations:
                - alignmentPeriod: "60s"
                  perSeriesAligner: "ALIGN_RATE"
                  crossSeriesReducer: "REDUCE_SUM"
                  groupByFields:
                    - "resource.labels.pod_name"
              comparison: "COMPARISON_GT"
              thresholdValue: 2.0  # 2x normal traffic
              duration: "300s"  # Sustained for 5 minutes
              trigger:
                count: 1
        combiner: "OR"
        alertStrategy:
          notificationRateLimit:
            period: "300s"  # Don't notify more than once every 5 minutes
          autoClose: "1800s"  # Auto-close after 30 minutes
        notificationChannels:
          - "sphere-notification-channel"
          - "refsre-notification-channel"
        documentation:
          content: "A significant increase in traffic has been detected on the Sphere application. Please investigate potential causes including: unexpected user behavior, scheduled events, external system integration issues, or potential DoS activity."
          mimeType: "text/markdown"
```