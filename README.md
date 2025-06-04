# COMPLETE ADDITION TO YOUR values.yaml FILE
# Add this EXACTLY to your objectives section:

objectives:
  # Your existing objectives remain unchanged
  - krmName: sphere-latency-usc1
    displayName: "SLO Latency: 99% of endpoints usc1"
    mode: STATS
    goal: 0.99
    errorBudgetThreshold: 0.80
    rollingPeriod: "2419200s"
    serviceLevelIndicator:
      requestBased:
        goodTotalRatio:
          goodServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.duration\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_response_status_code!=\"200\" metric.labels.http_route=\"monitoring.regex.full_match(\"/rest/.*\")\" resource.labels.location=\"us-central1\" range: min: -9007199254740991 max: 3"
          totalServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=\"monitoring.regex.full_match(\"/rest/.*\")\" resource.labels.location=\"us-central1\""

  - krmName: sphere-availability-usc1
    displayName: "Availability: 99% instrument endpoint usc1"
    mode: STATS
    goal: 0.99
    calendarPeriod: "WEEK"
    serviceLevelIndicator:
      requestBased:
        goodTotalRatio:
          goodServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_response_status_code!=\"401\" metric.labels.http_response_status_code!=\"403\" resource.labels.location=\"us-central1\""
          totalServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" metric.labels.http_route=\"monitoring.regex.full_match(\"/rest/.*\")\" resource.labels.location=\"us-central1\""

  # ADD THIS NEW NO-RESPONSE OBJECTIVE:
  - krmName: sphere-no-response-usc1
    displayName: "No Response: 99.5% requests receive response"
    mode: STATS
    goal: 0.995
    errorBudgetThreshold: 0.80
    rollingPeriod: "604800s"
    serviceLevelIndicator:
      requestBased:
        goodTotalRatio:
          goodServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.duration\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" resource.labels.location=\"us-central1\""
          totalServiceFilter: "metric.type=\"workload.googleapis.com/http.server.request.count\" resource.type=\"k8s_container\" metric.labels.app_id=\"2852\" metric.labels.component_id=\"coml-5913\" metric.labels.namespace=\"app-2852-default\" resource.labels.location=\"us-central1\""

# ADD THIS TO YOUR alerts section:
alerts:
  # Your existing alerts remain
  - name: "cuapi-availability-slo-burn"
    displayName: "Bad/Total-Ratio: 99%"
  - name: "cuapi-latency-slo-burn"
    displayName: "Latency Burn rate on 99%"

  # ADD THIS NEW ALERT:
  - name: "sphere-no-response-slo-burn"
    displayName: "No Response: Request timeout rate alert"






# FILE: monitoring-alertpolicy.yaml
# ADD THIS SECTION after your existing alert policies (around line 35-40):

{{- range $objective := .Values.services }}
{{- range $objective := $svc.objectives }}
{{- if eq $objective.krmName "sphere-no-response-usc1" }}
---
apiVersion: monitoring.cnrm.cloud.google.com/v1beta1
kind: MonitoringAlertPolicy
metadata:
  name: {{ $objective.krmName }}-alert-policy
  labels:
    app.kubernetes.io/name: {{ include "monitoring.name" $ }}
    app.kubernetes.io/instance: {{ $.Release.Name }}
    app.kubernetes.io/version: {{ $.Chart.AppVersion }}
    app.kubernetes.io/managed-by: {{ $.Release.Service }}
spec:
  projectRef:
    external: "{{ $.Values.projectRef }}"
  displayName: "{{ $objective.displayName }} - Burn Alert"
  enabled: true
  conditions:
    - displayName: "No response burn rate high"
      conditionThreshold:
        filter: 'select_slo_burn_rate("projects/{{ $.Values.projectRef }}/services/{{ $svc.component }}/serviceLevelObjectives/{{ $objective.krmName }}", "600s")'
        comparison: "COMPARISON_GT"
        thresholdValue: 3.0
        duration: "300s"
        aggregations:
          - alignmentPeriod: "300s"
            perSeriesAligner: "ALIGN_MEAN"
  combiner: "OR"
  alertStrategy:
    notificationRateLimit:
      period: "300s"
    autoClose: "1800s"
  notificationChannels:
    {{- range $svc.notificationChannels }}
    - name: {{ .name }}
      email: {{ .email }}
    {{- end }}
  documentation:
    content: |
      ## No Response Alert - {{ $svc.component }}
      
      **CRITICAL**: Requests to {{ $svc.component }} are timing out without receiving any HTTP response.
      
      **Immediate Actions:**
      1. Check pod health: `kubectl get pods -l component={{ $svc.component }} -n app-2852-default`
      2. Verify load balancer status in GCP Console
      3. Check recent deployments or configuration changes
      4. Monitor resource utilization (CPU/Memory)
      
      **This indicates:**
      - Complete service outages
      - Network connectivity issues  
      - Load balancer problems
      - Backend resource exhaustion
      
      **SLO Target:** 99.5% of requests must receive a response
      **Current Error Budget:** Check monitoring dashboard
    mimeType: "text/markdown"
{{- end }}
{{- end }}
{{- end }}

# FILE: monitoring-slo.yaml  
# MODIFY the serviceLevelIndicator section (around line 15-25) to include this condition:

{{- if eq $objective.krmName "sphere-no-response-usc1" }}
  serviceLevelIndicator:
    requestBased:
      goodTotalRatio:
        goodServiceFilter: >-
          metric.type="workload.googleapis.com/http.server.request.duration"
          resource.type="k8s_container"
          metric.labels.app_id="2852"
          metric.labels.component_id="{{ $svc.component }}"
          metric.labels.namespace="app-2852-default"
          resource.labels.location="us-central1"
        totalServiceFilter: >-
          metric.type="workload.googleapis.com/http.server.request.count"
          resource.type="k8s_container"
          metric.labels.app_id="2852"
          metric.labels.component_id="{{ $svc.component }}"
          metric.labels.namespace="app-2852-default"
          resource.labels.location="us-central1"
{{- else }}
  # Your existing serviceLevelIndicator logic for other objectives
  serviceLevelIndicator:
    requestBased:
      goodTotalRatio:
        goodServiceFilter: {{ $objective.serviceLevelIndicator.requestBased.goodTotalRatio.goodServiceFilter | quote }}
        totalServiceFilter: {{ $objective.serviceLevelIndicator.requestBased.goodTotalRatio.totalServiceFilter | quote }}
{{- end }}